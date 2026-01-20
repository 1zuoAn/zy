# -*- coding: utf-8 -*-
# @Author   : kiro
# @Time     : 2025/12/14
# @File     : main_orchestrator_graph.py

"""
主编排工作流 - LangGraph 版本

使用 StateGraph 构建，支持图结构导出和可视化。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from sqlalchemy import text

from app.config import settings
from app.core.clients.coze_loop_client import coze_loop_client_provider
from app.core.clients.db_client import pg_session
from app.core.clients.redis_client import redis_client
from app.core.config.constants import (
    CozePromptHubKey,
    DBAlias,
    LlmModelName,
    LlmProvider,
    RedisMessageKeyName,
    WorkflowType,
)
from app.core.tools import llm_factory
from app.schemas.entities.message.redis_message import (
    BaseRedisMessage,
    TextMessageContent,
    WithActionContent,
)
from app.schemas.entities.workflow.context import (
    ConversationMessage,
    format_conversation_history,
)
from app.schemas.entities.workflow.graph_state import MainOrchestratorState
from app.schemas.entities.workflow.llm_output import IntentClassifyResult
from app.schemas.request.workflow_request import MainWorkflowRequest
from app.schemas.response.workflow_response import WorkflowResponse
from app.service.chains.workflow.base_graph import BaseWorkflowGraph
from app.service.chains.workflow.workflow_delegate import WorkflowDelegate, get_delegate
from app.utils import thread_pool

# 意图到工作流类型的映射
INTENT_TO_WORKFLOW_TYPE = {
    "选品": WorkflowType.SELECT,
    "生图改图": WorkflowType.IMAGE_DESIGN,
    "趋势报告": WorkflowType.TRENDS,
    "媒体": WorkflowType.MEDIA,
    "店铺": WorkflowType.SHOP,
    "定时任务": WorkflowType.SCHEDULE,
    "闲聊": WorkflowType.CHAT,
    "other": WorkflowType.CHAT,
}


class MainOrchestratorGraph(BaseWorkflowGraph):
    """主编排工作流 - LangGraph 版本"""

    span_name = "主编排工作流"
    run_name = "main-orchestrator-graph"

    def __init__(self, delegate: Optional[WorkflowDelegate] = None) -> None:
        self._delegate = delegate if delegate is not None else get_delegate()
        super().__init__()

    def _build_graph(self) -> CompiledStateGraph:
        """构建工作流图"""
        graph = StateGraph(MainOrchestratorState)

        # 添加节点
        graph.add_node("init_state", self._init_state_node)
        graph.add_node("pre_think", self._pre_think_node)
        graph.add_node("extract_image", self._extract_image_content_node)
        graph.add_node("skip_image", self._skip_image_node)
        graph.add_node("query_history", self._query_conversation_history_node)
        graph.add_node("classify_intent", self._classify_intent_node)
        graph.add_node("run_sub_workflow", self._run_sub_workflow_node)
        graph.add_node("summary", self._summary_node)
        graph.add_node("publish_result", self._publish_result_node)
        graph.add_node("save_ai_response", self._save_ai_response_node)
        graph.add_node("package", self._package_result_node)

        # 设置入口
        graph.set_entry_point("init_state")

        # 添加边
        graph.add_edge("init_state", "pre_think")

        # 条件分支：是否有图片
        graph.add_conditional_edges(
            "pre_think",
            self._route_image_extraction,
            {"extract": "extract_image", "skip": "skip_image"},
        )

        graph.add_edge("extract_image", "query_history")
        graph.add_edge("skip_image", "query_history")
        graph.add_edge("query_history", "classify_intent")
        graph.add_edge("classify_intent", "run_sub_workflow")
        graph.add_edge("run_sub_workflow", "summary")
        graph.add_edge("summary", "publish_result")
        graph.add_edge("publish_result", "save_ai_response")
        graph.add_edge("save_ai_response", "package")
        graph.add_edge("package", END)

        return graph.compile()

    def _route_image_extraction(self, state: MainOrchestratorState) -> str:
        """条件路由：是否需要提取图片"""
        req = state.get("request")
        if req and hasattr(req, "images") and req.images and len(req.images) > 0:
            return "extract"
        return "skip"

    def _resolve_workflow_type(self, intent: str, preferred_entity: str) -> WorkflowType:
        """根据意图和 preferred_entity 解析工作流类型

        路由规则：
        1. 媒体意图：根据 preferred_entity 路由到对应媒体工作流
        2. 选品意图：
           - 有对应选品工作流的实体 → 选品工作流
           - 媒体平台实体（小红书/知款）→ 媒体工作流（这些平台没有商品选品功能，只能搜索笔记/帖子）
           - 未知实体 → 知衣选品（兜底）
        3. 其他意图：使用意图映射表
        """
        if intent == "媒体":
            if "海外探款" in preferred_entity:
                return WorkflowType.MEDIA_ABROAD_INS
            elif "知款" in preferred_entity:
                return WorkflowType.MEDIA_ZHIKUAN_INS
            else:
                return WorkflowType.MEDIA_ZXH_XHS
        elif intent == "选品":
            # 优先匹配有商品选品功能的实体
            if "抖衣" in preferred_entity or "抖音" in preferred_entity:
                return WorkflowType.SELECT_DOUYI
            elif "知衣" in preferred_entity:
                return WorkflowType.SELECT_ZHIYI
            elif "海外探款" in preferred_entity:
                return WorkflowType.SELECT_ABROAD_GOODS
            # 媒体平台实体 - 这些平台没有商品选品功能，只能搜索内容
            # 例如："帮我在小红书上选品" → 实际是搜索小红书笔记而非商品
            elif "小红" in preferred_entity or "知小红" in preferred_entity:
                logger.info(f"[主编排] 选品意图但实体为小红书，降级到媒体工作流(搜索笔记而非商品)")
                return WorkflowType.MEDIA_ZXH_XHS
            elif "知款" in preferred_entity:
                logger.info(f"[主编排] 选品意图但实体为知款，降级到媒体工作流(搜索帖子而非商品)")
                return WorkflowType.MEDIA_ZHIKUAN_INS
            # 默认返回知衣选品（最通用的选品工作流）
            logger.debug(f"[主编排] 选品意图未指定实体或实体未识别，使用默认知衣选品工作流")
            return WorkflowType.SELECT_ZHIYI
        return INTENT_TO_WORKFLOW_TYPE.get(intent, WorkflowType.CHAT)

    # ==================== 节点实现 ====================

    def _init_state_node(self, state: MainOrchestratorState) -> Dict[str, Any]:
        """初始化状态节点"""
        req = state["request"]

        def insert_start_track() -> None:
            try:
                with pg_session(DBAlias.DB_ABROAD_AI.value) as session:
                    session.execute(
                        text("""
                            INSERT INTO n8n_user_query_message(session_id, user_query)
                            VALUES (:session_id, :user_query)
                        """),
                        {"session_id": req.session_id, "user_query": f"human:{req.user_query}"},
                    )
                    logger.debug(f"[主工作流] 成功记录用户查询: {req.session_id}")
            except Exception as e:
                logger.error(f"[主工作流] 记录用户查询失败: {req.session_id}, error={e}")

        thread_pool.submit_with_context(insert_start_track)
        return {}

    def _pre_think_node(self, state: MainOrchestratorState) -> Dict[str, Any]:
        """预处理节点 - 发送开始消息"""
        req = state["request"]

        start_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="收到任务",
            status="RUNNING",
            content_type=2,
            content=WithActionContent(
                text="收到，正在分析您的需求...",
                actions=["view", "export", "download"],
                agent="search",
            ),
            create_ts=int(round(time.time() * 1000)),
        )
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            start_message.model_dump_json(),
        )
        logger.debug("[主工作流] 推送开始消息")
        return {}

    def _skip_image_node(self, state: MainOrchestratorState) -> Dict[str, Any]:
        """跳过图片提取节点"""
        return {"image_content": None}

    def _extract_single_image(self, image_url: str) -> Optional[str]:
        """提取单张图片的描述"""
        max_retries = settings.main_workflow_image_retry_attempts

        for attempt in range(max_retries):
            try:
                llm = llm_factory.get_llm(
                    LlmProvider.HUANXIN.name,
                    LlmModelName.HUANXIN_GPT_4O.value,
                )
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "请描述这张服装图片的主要特征，包括类目和2-3个关键属性，控制在20字以内。",
                            },
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ]
                result = llm.invoke(messages)
                return str(result.content)
            except Exception as e:
                logger.warning(f"[图片提取] 第{attempt+1}次尝试失败: {image_url}, error={e}")
                if attempt == max_retries - 1:
                    logger.error(f"[图片提取] 所有重试失败，跳过图片: {image_url}")
                    return None
        return None

    def _extract_image_content_node(self, state: MainOrchestratorState) -> Dict[str, Any]:
        """图片内容提取节点"""
        req: MainWorkflowRequest = state["request"]  # type: ignore[assignment]

        if not req.images:
            return {"image_content": None}

        image_descriptions: List[str] = []
        max_workers = min(len(req.images), settings.main_workflow_image_max_workers)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(self._extract_single_image, url): url for url in req.images
            }
            for future in as_completed(future_to_url):
                result = future.result()
                if result:
                    image_descriptions.append(result)

        image_content = " ".join(image_descriptions) if image_descriptions else None
        logger.debug(
            f"[图片提取] 结果: {image_content}, 成功处理 {len(image_descriptions)}/{len(req.images)} 张图片"
        )
        return {"image_content": image_content}

    def _query_conversation_history_node(self, state: MainOrchestratorState) -> Dict[str, Any]:
        """查询会话历史节点"""
        req = state["request"]

        messages: List[ConversationMessage] = []
        try:
            with pg_session(DBAlias.DB_ABROAD_AI.value) as session:
                result = session.execute(
                    text("""
                        SELECT id, session_id, user_query
                        FROM n8n_user_query_message
                        WHERE session_id = :session_id
                        ORDER BY id
                    """),
                    {"session_id": req.session_id},
                )
                for row in result.mappings().all():
                    messages.append(
                        ConversationMessage(
                            id=row["id"],
                            session_id=row["session_id"],
                            user_query=row["user_query"],
                        )
                    )
        except Exception as e:
            logger.warning(f"[会话历史] 查询失败: {e}")

        conversation_history = format_conversation_history(messages)
        logger.debug(f"[会话历史] 共 {len(messages)} 条记录")
        return {"conversation_history": conversation_history}

    def _classify_intent_node(self, state: MainOrchestratorState) -> Dict[str, Any]:
        """意图分类节点"""
        req = state["request"]
        conversation_history = state.get("conversation_history", "")
        image_content = state.get("image_content")

        prompt_params = {
            "user_query": req.user_query,
            "conversation_history": conversation_history,
            "preferred_entity": req.preferred_entity,
            "industry": req.industry,
            "image_content": image_content or "",
        }

        try:
            messages: List[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
                prompt_key=CozePromptHubKey.MAIN_INTENT_CLASSIFY_PROMPT.value,
                variables=prompt_params,
            )
        except Exception as e:
            logger.warning(f"[意图分类] 无法从 CozeLoop 获取提示词，使用内置提示词: {e}")
            messages = self._build_fallback_intent_prompt(prompt_params)

        llm: BaseChatModel = llm_factory.get_llm(
            LlmProvider.HUANXIN.name,
            LlmModelName.HUANXIN_GEMINI_2_5_FLASH.value,
        )
        structured_llm = llm.with_structured_output(IntentClassifyResult).with_retry(stop_after_attempt=2)

        intent_result: IntentClassifyResult = structured_llm.with_config(
            run_name="意图分类"
        ).invoke(messages)

        logger.info(f"[意图分类] 结果: {intent_result.intent}, 置信度: {intent_result.confidence}")
        return {"intent": intent_result.intent, "intent_result": intent_result}

    def _build_fallback_intent_prompt(self, params: Dict[str, Any]) -> List[BaseMessage]:
        """构建内置的意图分类提示词"""
        system_prompt = """你是一个意图分类助手。根据用户的输入，将其分类为以下意图之一：
- 选品: 用户想要搜索、筛选、查找服装款式
- 生图改图: 用户想要生成或编辑图片
- 趋势报告: 用户想要了解时尚趋势、行业报告
- 媒体: 用户想要在社交媒体平台（如 Instagram、小红书）上搜索内容
- 店铺: 用户想要查询店铺信息或排名
- 定时任务: 用户想要设置定时提醒或任务
- other: 其他意图，如闲聊、问候等

请根据用户查询返回最匹配的意图，并给出 0-1 之间的置信度分数。"""

        user_prompt = f"""用户查询: {params.get('user_query', '')}
用户偏好实体: {params.get('preferred_entity', '')}
行业: {params.get('industry', '')}
会话历史: {params.get('conversation_history', '')}
图片内容: {params.get('image_content', '')}

请分析用户意图。"""

        return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    def _run_sub_workflow_node(self, state: MainOrchestratorState) -> Dict[str, Any]:
        """执行子工作流节点"""
        req = state["request"]
        intent = state.get("intent", "other")

        workflow_type = self._resolve_workflow_type(intent, req.preferred_entity)
        logger.info(f"[主工作流] 路由到子工作流: {workflow_type.value}")

        sub_result = self._delegate.execute(workflow_type, req)
        return {"sub_workflow_result": sub_result}

    def _summary_node(self, state: MainOrchestratorState) -> Dict[str, Any]:
        """结果汇总节点"""
        sub_result = state.get("sub_workflow_result")

        if not sub_result:
            return {"summary_text": "抱歉，处理过程中出现问题，请稍后重试。"}

        # 闲聊或非成功场景直接透传子工作流输出，避免因外部提示词缺失/异常导致的意外文案
        if sub_result.workflow_name == WorkflowType.CHAT.value:
            return {"summary_text": sub_result.output}

        if not sub_result.success:
            summary_text = f"抱歉，{sub_result.output}，请稍后重试。"
        else:
            # 对齐 n8n：直接使用子工作流输出，不再额外做 LLM 汇总，防止提示词缺失时出现异常提示
            summary_text = sub_result.output

        logger.debug(f"[结果汇总] {summary_text}")
        return {"summary_text": summary_text}

    def _publish_result_node(self, state: MainOrchestratorState) -> Dict[str, Any]:
        """推送结果消息节点"""
        req = state["request"]
        summary_text = state.get("summary_text", "")

        finish_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="结果",
            status="END",
            content_type=1,
            content=TextMessageContent(text=summary_text),
            create_ts=int(round(time.time() * 1000)),
        )
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            finish_message.model_dump_json(),
        )
        logger.debug("[主工作流] 推送结果消息")
        return {}

    def _save_ai_response_node(self, state: MainOrchestratorState) -> Dict[str, Any]:
        """保存AI回复节点"""
        req = state["request"]
        summary_text = state.get("summary_text", "")

        def _insert_ai_response() -> None:
            try:
                with pg_session(DBAlias.DB_ABROAD_AI.value) as session:
                    session.execute(
                        text("""
                            INSERT INTO n8n_user_query_message(session_id, user_query)
                            VALUES (:session_id, :user_query)
                        """),
                        {"session_id": req.session_id, "user_query": f"ai:{summary_text}"},
                    )
                    logger.debug(f"[主工作流] 成功保存AI回复: {req.session_id}")
            except Exception as e:
                logger.error(f"[主工作流] 保存AI回复失败: {req.session_id}, error={e}")

        thread_pool.submit_with_context(_insert_ai_response)
        return {}

    def _package_result_node(self, state: MainOrchestratorState) -> Dict[str, Any]:
        """封装返回结果节点"""
        summary_text = state.get("summary_text", "")
        sub_result = state.get("sub_workflow_result")

        response = WorkflowResponse(
            select_result=summary_text,
            relate_data=sub_result.relate_data if sub_result else None,
        )
        return {"workflow_response": response}


__all__ = ["MainOrchestratorGraph"]
