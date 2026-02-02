# -*- coding: utf-8 -*-
# @Author   : kiro
# @Time     : 2025/12/14
# @File     : chat_graph.py

"""
闲聊工作流 - LangGraph 版本

直接调用 LLM 进行对话，不依赖外部业务 API。
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from app.core.clients.redis_client import redis_client
from app.core.clients.coze_loop_client import coze_loop_client_provider
from app.core.config.constants import CozePromptHubKey, LlmModelName, LlmProvider, RedisMessageKeyName
from app.core.tools import llm_factory
from app.schemas.entities.message.redis_message import (
    BaseRedisMessage,
    TextMessageContent,
    WithActionContent,
)
from app.schemas.entities.workflow.graph_state import ChatWorkflowState
from app.schemas.response.workflow_response import WorkflowResponse
from app.service.chains.workflow.base_graph import BaseWorkflowGraph


class ChatGraph(BaseWorkflowGraph):
    """闲聊工作流 - LangGraph 版本"""

    span_name = "闲聊工作流"
    run_name = "chat-graph"

    def _build_graph(self) -> CompiledStateGraph:
        """构建工作流图"""
        graph = StateGraph(ChatWorkflowState)

        graph.add_node("init_state", self._init_state_node)
        graph.add_node("chat", self._chat_node)
        graph.add_node("package", self._package_result_node)

        graph.set_entry_point("init_state")
        graph.add_edge("init_state", "chat")
        graph.add_edge("chat", "package")
        graph.add_edge("package", END)

        return graph.compile()

    def _init_state_node(self, state: ChatWorkflowState) -> Dict[str, Any]:
        """初始化状态节点 - 推送开始消息"""
        req = state["request"]

        if getattr(req, "suppress_messages", False):
            return {}

        # 推送开始消息
        start_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="收到任务",
            status="RUNNING",
            content_type=2,
            content=WithActionContent(
                text="收到，正在为您回复...",
                actions=[],
                agent="chat",
            ),
            create_ts=int(round(time.time() * 1000)),
        )
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            start_message.model_dump_json(),
        )
        logger.debug("[闲聊工作流] 推送开始消息")
        return {}

    def _chat_node(self, state: ChatWorkflowState) -> Dict[str, Any]:
        """调用 LLM 进行闲聊"""
        req = state["request"]

        now_str = datetime.now().strftime("%Y-%m-%d")
        prompt_vars = {"current_date": now_str, "user_query": req.user_query}
        try:
            messages = coze_loop_client_provider.get_langchain_messages(
                prompt_key=CozePromptHubKey.MAIN_CHAT_PROMPT.value,
                variables=prompt_vars,
            )
        except Exception as e:
            logger.warning(f"[闲聊工作流] 获取提示词失败，降级为纯用户输入: {e}")
            messages = [HumanMessage(content=req.user_query or "")]

        model_candidates = [
            LlmModelName.HUANXIN_GEMINI_3_FLASH_PREVIEW.value,
            LlmModelName.HUANXIN_GROK_4_1_FAST_NON_REASONING.value,
        ]
        response_text = ""
        last_error: Exception | None = None
        for model in model_candidates:
            try:
                llm: BaseChatModel = llm_factory.get_llm(LlmProvider.HUANXIN.name, model)
                response_text = (llm.with_retry(stop_after_attempt=2) | StrOutputParser()).invoke(messages)
                break
            except Exception as e:
                last_error = e
                logger.warning(f"[闲聊工作流] 模型 {model} 调用失败，尝试下一模型: {e}")
        if not response_text:
            if last_error:
                logger.warning(f"[闲聊工作流] 回退到默认回复: {last_error}")
            response_text = "抱歉，我没有理解你的问题。"

        logger.debug(f"[闲聊工作流] 回复: {response_text[:50]}...")
        return {"chat_response": response_text}

    def _package_result_node(self, state: ChatWorkflowState) -> Dict[str, Any]:
        """封装返回结果节点 - 推送结果消息"""
        req = state["request"]
        chat_response = state.get("chat_response", "抱歉，我没有理解你的问题。")

        if getattr(req, "suppress_messages", False):
            return {"workflow_response": WorkflowResponse(select_result=chat_response, relate_data=None)}

        # 推送结果消息
        result_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="结果",
            status="END",
            content_type=1,
            content=TextMessageContent(text=chat_response),
            create_ts=int(round(time.time() * 1000)),
        )
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            result_message.model_dump_json(),
        )
        logger.debug("[闲聊工作流] 推送结果消息")

        return {"workflow_response": WorkflowResponse(select_result=chat_response, relate_data=None)}


__all__ = ["ChatGraph"]
