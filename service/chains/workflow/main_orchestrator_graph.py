# -*- coding: utf-8 -*-
# @Author   : kiro
# @Time     : 2025/12/14
# @File     : main_orchestrator_graph.py

"""
主编排工作流 - LangGraph 版本

使用 StateGraph 构建，支持图结构导出和可视化。
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.config import settings
from app.core.clients.db_client import pg_session
from app.core.clients.redis_client import redis_client
from app.core.config.constants import (
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
from app.schemas.request.workflow_request import MainWorkflowRequest
from app.schemas.response.workflow_response import WorkflowResponse
from app.service.chains.workflow.base_graph import BaseWorkflowGraph
from app.service.chains.workflow.workflow_delegate import WorkflowDelegate, get_delegate
from app.utils import thread_pool

class AgentToolCall(BaseModel):
    tool: str = Field(description="工具名称")
    reason: str = Field(default="", description="调用理由")
    params: Dict[str, Any] = Field(default_factory=dict, description="工具参数")


class AgentPlan(BaseModel):
    tool_calls: List[AgentToolCall] = Field(default_factory=list, description="工具调用列表")
    final_reply: Optional[str] = Field(default=None, description="无需调用工具时的直接回复")


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
        graph.add_node("run_agent", self._run_agent_node)
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
        graph.add_edge("query_history", "run_agent")
        graph.add_edge("run_agent", "summary")
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

    def _build_agent_prompt(self, state: MainOrchestratorState) -> List[BaseMessage]:
        """构建主Agent提示词"""
        req = state["request"]
        conversation_history = state.get("conversation_history", "")
        image_content = state.get("image_content") or ""
        tool_list = [
            "select_zhiyi", "select_douyi", "select_abroad_goods",
            "media_abroad_ins", "media_zhikuan_ins", "media_zxh_xhs",
            "shop_rank", "image_search",
            "image_create", "image_edit",
            "schedule_task", "inspect_artifact",
        ]
        schema_text = json.dumps(AgentPlan.model_json_schema(), ensure_ascii=False)
        system_prompt = (
            "你是主编排Agent，负责根据用户诉求选择子工作流工具并可调用多个工具完成复杂请求。\n"
            "要求：仅输出JSON，字段为 tool_calls(列表) 与 final_reply(可选)。\n"
            "必须严格符合以下JSON Schema：\n"
            f"{schema_text}\n"
            "工具列表：\n"
            f"- {', '.join(tool_list)}\n"
            "规则：\n"
            "1) 趋势报告忽略，不要选择相关工具。\n"
            "2) 闲聊直接输出 final_reply，不调用工具。\n"
            "3) 图搜仅在用户明确要找同款/相似款且有图片时使用。\n"
            "4) 有图片链接且要求修改时用 image_edit；否则 image_create。\n"
            "5) 如用户要求查看资产详情，使用 inspect_artifact 并提供 artifact_id。\n"
            "6) 可返回多个 tool_calls 以完成组合任务。\n"
            "7) 用户提出多个任务/多个查询（如“再/然后/以及/并且/同时”）时，必须拆成多条 tool_calls。\n"
            "8) 同一工具多次调用时，用 params.query 指定每个子任务的查询文本。\n"
            "9) inspect_artifact 支持 params: artifact_id 或 'latest'，可带 limit(<=20)。\n"
        )
        user_prompt = (
            "用户信息如下：\n"
            f"- 用户问题: {req.user_query}\n"
            f"- 平台偏好: {req.preferred_entity}\n"
            f"- 行业: {req.industry}\n"
            f"- 用户偏好: {req.user_preferences}\n"
            f"- 图片描述: {image_content}\n"
            f"- 图片链接: {req.images or []}\n"
            f"- 图搜链接: {req.image_url or ''}\n"
            f"- 历史上下文: {conversation_history}\n"
            "请输出JSON。\n"
        )
        return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    def _safe_parse_json(self, raw: Optional[str]) -> Any:
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def _parse_agent_plan_from_text(self, raw: str) -> AgentPlan | None:
        content = raw.strip()
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()
        data = self._safe_parse_json(content)
        if not isinstance(data, dict):
            return None
        tool_calls = data.get("tool_calls") or data.get("toolCalls") or []
        final_reply = data.get("final_reply") or data.get("finalReply")
        parsed_calls: List[AgentToolCall] = []
        if isinstance(tool_calls, list):
            for item in tool_calls:
                if not isinstance(item, dict):
                    continue
                parsed_calls.append(
                    AgentToolCall(
                        tool=item.get("tool") or item.get("name") or "",
                        reason=item.get("reason") or "",
                        params=item.get("params") or {},
                    )
                )
        return AgentPlan(tool_calls=parsed_calls, final_reply=final_reply)

    def _register_artifact(
        self,
        artifacts: Dict[str, Any],
        payloads: Dict[str, Any],
        *,
        artifact_type: str,
        description: str,
        content_cache: Optional[str],
        meta: Dict[str, Any],
        payload: Any = None,
    ) -> str:
        artifact_id = f"{artifact_type}_{uuid4().hex[:8]}"
        artifacts[artifact_id] = {
            "id": artifact_id,
            "type": artifact_type,
            "description": description,
            "content_cache": content_cache,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "meta": meta,
        }
        if payload is not None:
            payloads[artifact_id] = payload
        return artifact_id

    def _inspect_artifact(self, state: MainOrchestratorState, artifact_id: str) -> Dict[str, Any]:
        artifacts = state.get("artifacts") or {}
        payloads = state.get("artifact_payloads") or {}
        artifact = artifacts.get(artifact_id)
        if not artifact:
            return {"success": False, "user_output": "未找到对应资产", "artifact_id": artifact_id}
        artifact_type = artifact.get("type")
        if artifact_type == "image_asset":
            detail = artifact.get("content_cache")
        elif artifact_type == "product_list":
            detail = payloads.get(artifact_id)
        else:
            detail = artifact.get("meta")
        return {
            "success": True,
            "user_output": artifact.get("description") or "已获取资产详情",
            "artifact_id": artifact_id,
            "detail": detail,
        }

    def _execute_tool_call(
        self,
        call: AgentToolCall,
        state: MainOrchestratorState,
        artifacts: Dict[str, Any],
        payloads: Dict[str, Any],
    ) -> Dict[str, Any]:
        req = state["request"]
        tool = call.tool
        if tool == "inspect_artifact":
            artifact_id = call.params.get("artifact_id") or call.params.get("artifactId")
            if not artifact_id:
                return {"tool": tool, "success": False, "user_output": "缺少artifact_id"}
            return {"tool": tool, **self._inspect_artifact(state, artifact_id)}

        workflow_map = {
            "select_zhiyi": WorkflowType.SELECT_ZHIYI,
            "select_douyi": WorkflowType.SELECT_DOUYI,
            "select_abroad_goods": WorkflowType.SELECT_ABROAD_GOODS,
            "media_abroad_ins": WorkflowType.MEDIA_ABROAD_INS,
            "media_zhikuan_ins": WorkflowType.MEDIA_ZHIKUAN_INS,
            "media_zxh_xhs": WorkflowType.MEDIA_ZXH_XHS,
            "image_create": WorkflowType.IMAGE_CREATE,
            "image_edit": WorkflowType.IMAGE_EDIT,
            "image_search": WorkflowType.IMAGE_SEARCH,
            "shop_rank": WorkflowType.SHOP,
            "schedule_task": WorkflowType.SCHEDULE,
        }
        workflow_type = workflow_map.get(tool)
        if not workflow_type:
            return {"tool": tool, "success": False, "user_output": "未知工具"}

        result = self._delegate.execute(workflow_type, req)
        payload = self._safe_parse_json(result.relate_data)

        artifact_type = "product_list"
        content_cache = None
        if tool in ("image_create", "image_edit"):
            artifact_type = "image_asset"
            content_cache = req.image_prompt or req.edit_prompt or req.user_query
        elif tool == "schedule_task":
            artifact_type = "task_receipt"

        count = len(payload) if isinstance(payload, list) else None
        base_desc = {
            "select_zhiyi": "选品列表已生成",
            "select_douyi": "选品列表已生成",
            "select_abroad_goods": "选品列表已生成",
            "media_abroad_ins": "媒体内容列表已生成",
            "media_zhikuan_ins": "媒体内容列表已生成",
            "media_zxh_xhs": "媒体内容列表已生成",
            "shop_rank": "店铺排行列表已生成",
            "image_search": "相似款列表已生成",
            "image_create": "图片已生成",
            "image_edit": "图片已编辑",
            "schedule_task": "定时任务已生成",
        }.get(tool, "任务已完成")
        if count is not None and artifact_type == "product_list":
            description = f"{base_desc}，共{count}条"
        else:
            description = base_desc

        artifact_id = self._register_artifact(
            artifacts,
            payloads,
            artifact_type=artifact_type,
            description=description,
            content_cache=content_cache,
            meta={"tool": tool, "workflow": result.workflow_name, "count": count},
            payload=payload,
        )

        return {
            "tool": tool,
            "success": result.success,
            "user_output": result.output,
            "artifact_id": artifact_id,
            "relate_data": result.relate_data,
        }

    def _run_agent_node(self, state: MainOrchestratorState) -> Dict[str, Any]:
        """主Agent节点：选择并执行子工作流工具"""
        messages = self._build_agent_prompt(state)
        llm: BaseChatModel = llm_factory.get_llm(
            LlmProvider.OPENROUTER.name,
            LlmModelName.OPENROUTER_GEMINI_3_PRO_PREVIEW.value,
        )
        try:
            raw_resp = llm.invoke(messages)
            raw_text = raw_resp.content if hasattr(raw_resp, "content") else str(raw_resp)
            logger.info(f"[主Agent] 原始输出: {raw_text}")
            plan = self._parse_agent_plan_from_text(str(raw_text))
            if not plan:
                raise ValueError("Agent输出无法解析为JSON Schema")
        except Exception as e:
            logger.warning(f"[主Agent] 生成计划失败: {e}")
            return {"agent_reply": "抱歉，当前无法完成任务，请稍后重试。"}

        artifacts = state.get("artifacts") or {}
        payloads = state.get("artifact_payloads") or {}
        tool_results: List[Dict[str, Any]] = []
        tool_calls = plan.tool_calls or []

        for call in tool_calls:
            result = self._execute_tool_call(call, state, artifacts, payloads)
            tool_results.append(result)

        if plan.final_reply:
            agent_reply = plan.final_reply
        else:
            outputs = [r.get("user_output") for r in tool_results if r.get("user_output")]
            agent_reply = "\n".join(outputs) if outputs else "任务已完成"

        return {
            "agent_reply": agent_reply,
            "tool_calls": [call.model_dump() for call in tool_calls],
            "tool_results": tool_results,
            "artifacts": artifacts,
            "artifact_payloads": payloads,
        }

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
        return {
            "artifacts": {},
            "artifact_payloads": {},
            "tool_calls": [],
            "tool_results": [],
        }

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


    def _summary_node(self, state: MainOrchestratorState) -> Dict[str, Any]:
        """结果汇总节点"""
        summary_text = state.get("agent_reply")
        if not summary_text:
            summary_text = "抱歉，处理过程中出现问题，请稍后重试。"
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
        relate_data = None
        tool_results = state.get("tool_results") or []
        for item in reversed(tool_results):
            if item.get("relate_data"):
                relate_data = item.get("relate_data")
                break
        response = WorkflowResponse(select_result=summary_text, relate_data=relate_data)
        return {"workflow_response": response}


__all__ = ["MainOrchestratorGraph"]
