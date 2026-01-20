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
from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from app.core.clients.redis_client import redis_client
from app.core.config.constants import LlmModelName, LlmProvider, RedisMessageKeyName
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

        llm: BaseChatModel = llm_factory.get_llm(
            LlmProvider.HUANXIN.name,
            LlmModelName.HUANXIN_GEMINI_2_5_FLASH.value,
        )

        system_prompt = """你是一个友好的 AI 助手，专注于帮助用户进行服装选品和时尚相关的问题。
如果用户的问题与选品无关，你可以友好地回答，但尽量引导用户回到选品相关的话题。
回答要简洁友好，不超过 100 字。"""

        messages = [("system", system_prompt), ("human", req.user_query)]

        chat_chain = llm | StrOutputParser()
        response_text = chat_chain.with_config(run_name="闲聊对话").invoke(messages)

        logger.debug(f"[闲聊工作流] 回复: {response_text[:50]}...")
        return {"chat_response": response_text}

    def _package_result_node(self, state: ChatWorkflowState) -> Dict[str, Any]:
        """封装返回结果节点 - 推送结果消息"""
        req = state["request"]
        chat_response = state.get("chat_response", "抱歉，我没有理解你的问题。")

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
