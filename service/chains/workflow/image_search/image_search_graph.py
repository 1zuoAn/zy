# -*- coding: utf-8 -*-
"""
图搜工作流 - 对齐 n8n 图搜 workflow
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import requests
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from app.config import settings
from app.core.clients.redis_client import redis_client
from app.core.config.constants import RedisMessageKeyName, WorkflowMessageContentType
from app.schemas.entities.message.redis_message import (
    BaseRedisMessage,
    CustomDataContent,
    ParameterData,
    ParameterDataWithAgentContent,
    TextMessageContent,
    WithActionContent,
)
from app.schemas.entities.workflow.graph_state import ImageSearchWorkflowState
from app.schemas.request.workflow_request import WorkflowRequest
from app.schemas.response.workflow_response import WorkflowResponse
from app.service.chains.workflow.base_graph import BaseWorkflowGraph


class ImageSearchGraph(BaseWorkflowGraph):
    """图搜工作流"""

    span_name = "图搜工作流"
    run_name = "image-search-graph"

    _SUCCESS_TEXT = "已完成图搜检索，找到了商品"
    _FAIL_TEXT = "“很抱歉，本次未完成筛选，当前任务点数已返还，请稍后重试～”"

    _NODE_NAME_MAP = {
        "init_state": "初始化",
        "notify_start": "发送开始消息",
        "fetch_box": "获取图片标签",
        "handle_success": "成功处理",
        "handle_failure": "失败处理",
        "package": "封装结果",
    }

    def _get_trace_name_modifier(self):
        """返回节点名称修改函数（用于 CozeLoop 追踪）"""

        def modifier(node_name: str) -> str:
            if node_name == "init_state":
                return self.span_name
            return self._NODE_NAME_MAP.get(node_name, node_name)

        return modifier

    def _build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(ImageSearchWorkflowState)

        graph.add_node("init_state", self._init_state_node)
        graph.add_node("notify_start", self._notify_start_node)
        graph.add_node("fetch_box", self._fetch_box_node)
        graph.add_node("handle_success", self._handle_success_node)
        graph.add_node("handle_failure", self._handle_failure_node)
        graph.add_node("package", self._package_result_node)

        graph.set_entry_point("init_state")
        graph.add_edge("init_state", "notify_start")
        graph.add_edge("notify_start", "fetch_box")
        graph.add_conditional_edges(
            "fetch_box",
            self._route_after_fetch,
            {"success": "handle_success", "failure": "handle_failure"},
        )
        graph.add_edge("handle_success", "package")
        graph.add_edge("handle_failure", "package")
        graph.add_edge("package", END)

        return graph.compile()

    _ABROAD_TYPES = ("Temu", "独立站", "SHEIN", "亚马逊")

    def _resolve_variant(self, preferred_entity: str, abroad_type: str) -> tuple[bool, str]:
        preferred_lower = preferred_entity.lower()
        abroad_type = abroad_type.strip()

        if abroad_type in self._ABROAD_TYPES:
            return True, abroad_type
        if "海外探款" in preferred_entity:
            if "ins" in preferred_lower or not abroad_type or abroad_type.upper() == "INS":
                return True, "INS"

        if "抖衣" in preferred_entity or "抖音" in preferred_entity or "douyin" in preferred_lower:
            return False, "抖衣"
        if "ins" in preferred_lower or "知款" in preferred_entity:
            return False, "INS"
        return False, "淘宝"

    def _resolve_title(self, is_abroad: bool, platform_type: str) -> str:
        if is_abroad:
            return "abroad-image-search"
        if platform_type == "INS":
            return "INS图搜"
        return "知衣图搜"

    def _resolve_platform(
        self,
        preferred_entity: str,
        is_abroad: bool,
        platform_type: str,
    ) -> str:
        if is_abroad:
            if platform_type == "INS":
                return "海外探款"
            return preferred_entity or ""
        return "知衣"

    def _normalize_images(self, req: WorkflowRequest) -> List[str]:
        images: List[str] = []
        image_url = getattr(req, "image_url", None)
        if image_url is not None:
            return [str(image_url)]

        raw_images = getattr(req, "images", None)
        if isinstance(raw_images, list):
            images = [str(item) for item in raw_images if item is not None]

        if not images:
            input_images = getattr(req, "input_images", None)
            if isinstance(input_images, str):
                parts = [part for part in input_images.split("#")]
                images = parts or [input_images]
            elif isinstance(input_images, list):
                images = [str(item) for item in input_images if item is not None]

        return images

    def _push_message(self, message: BaseRedisMessage) -> None:
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            message.model_dump_json(exclude_none=True),
        )

    def _route_after_fetch(self, state: ImageSearchWorkflowState) -> str:
        return "success" if state.get("box_success") else "failure"

    def _init_state_node(self, state: ImageSearchWorkflowState) -> Dict[str, Any]:
        req: WorkflowRequest = state["request"]

        images = self._normalize_images(req)
        preferred_entity = req.preferred_entity or ""
        abroad_type = req.abroad_type or ""
        is_abroad, platform_type = self._resolve_variant(preferred_entity, abroad_type)
        entity_type = 11 if is_abroad else 10

        platform = self._resolve_platform(preferred_entity, is_abroad, platform_type)
        title = self._resolve_title(is_abroad, platform_type)
        datasource_content = "abroad-image-search" if is_abroad else "zy-image-search"
        cost_id = f"{req.session_id}_{int(time.time() * 1000)}"
        image_url = images[0] if images else ""

        logger.info(f"[图搜] 接收到 {len(images)} 张图片，准备检索")
        return {
            "cost_id": cost_id,
            "image_urls": images,
            "image_url": image_url,
            "entity_type": entity_type,
            "platform": platform,
            "platform_type": platform_type,
            "title": title,
            "datasource_content": datasource_content,
        }

    def _notify_start_node(self, state: ImageSearchWorkflowState) -> Dict[str, Any]:
        req = state["request"]
        cost_id = state.get("cost_id") or ""
        start_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="收到图搜任务",
            status="RUNNING",
            content_type=WorkflowMessageContentType.PRE_TEXT.value,
            content=WithActionContent(
                text="收到，开始图搜任务...",
                actions=["image_search"],
                agent="image_search",
                cost_id=cost_id,
            ),
            create_ts=int(round(time.time() * 1000)),
        )
        self._push_message(start_message)
        return {}

    def _fetch_box_node(self, state: ImageSearchWorkflowState) -> Dict[str, Any]:
        req = state["request"]
        image_url = state.get("image_url") or ""

        base_url = (settings.zhikuan_api_url or "").rstrip("/")
        if not base_url:
            logger.warning("[图搜] zhikuan_api_url 未配置")
            return {"box_success": False, "box_result": None}

        params = {"mainUrl": image_url}
        if state.get("entity_type") == 11:
            params["excludeCategoryList"] = "帽子"

        headers = {
            "USER-ID": str(req.user_id),
            "TEAM-ID": str(req.team_id),
        }

        try:
            resp = requests.get(
                f"{base_url}/image-bus/get-box",
                params=params,
                headers=headers,
                timeout=settings.zhikuan_api_timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.warning(f"[图搜] 获取图片标签失败: {exc}")
            return {"box_success": False, "box_result": None}

        success = bool(payload.get("success"))
        if not success:
            logger.warning(f"[图搜] 图片标签识别失败: {payload}")
        return {"box_success": success, "box_result": payload.get("result")}

    def _handle_success_node(self, state: ImageSearchWorkflowState) -> Dict[str, Any]:
        req = state["request"]
        image_url = state.get("image_url") or ""
        entity_type = state.get("entity_type") or 10
        platform = state.get("platform") or ""
        platform_type = state.get("platform_type") or ""
        datasource_content = state.get("datasource_content") or ""
        title = state.get("title") or ""
        box_result = state.get("box_result")

        searching_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="正在启用图搜",
            status="RUNNING",
            content_type=WorkflowMessageContentType.PROCESSING.value,
            content=TextMessageContent(text="根据图片搜索相似商品中..."),
            create_ts=int(round(time.time() * 1000)),
        )
        self._push_message(searching_message)

        done_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="正在图搜",
            status="RUNNING",
            content_type=WorkflowMessageContentType.PROCESSING.value,
            content=TextMessageContent(text="搜索已完成"),
            create_ts=int(round(time.time() * 1000)),
        )
        self._push_message(done_message)

        result_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="结果",
            status="RUNNING",
            content_type=WorkflowMessageContentType.PROCESSING.value,
            content=TextMessageContent(text="图搜任务已完成。"),
            create_ts=int(round(time.time() * 1000)),
        )
        self._push_message(result_message)

        request_body = {
            "box": box_result,
            "start_time": "",
            "end_time": "",
            "platform": platform,
            "image_url": image_url,
            "platform_type": platform_type,
        }
        query_params = [image_url]

        datasource_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="输出参数",
            status="RUNNING",
            content_type=WorkflowMessageContentType.QUERY_DATA_SOURCE.value,
            content=CustomDataContent(
                data={
                    "entity_type": entity_type,
                    "content": datasource_content,
                    "query_params": query_params,
                }
            ),
            create_ts=int(round(time.time() * 1000)),
        )
        self._push_message(datasource_message)

        parameter_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="输出参数",
            status="END",
            content_type=WorkflowMessageContentType.RESULT.value,
            content=ParameterDataWithAgentContent(
                data=ParameterData(
                    request_body=json.dumps(
                        request_body,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    actions=["image_search"],
                    title=title,
                    entity_type=entity_type,
                ),
                agent="image_search",
            ),
            create_ts=int(round(time.time() * 1000)),
        )
        self._push_message(parameter_message)

        return {"output_text": self._SUCCESS_TEXT}

    def _handle_failure_node(self, state: ImageSearchWorkflowState) -> Dict[str, Any]:
        req = state["request"]
        cost_id = state.get("cost_id") or ""
        failure_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="任务失败",
            status="RUNNING",
            content_type=WorkflowMessageContentType.COST_REFOUND.value,
            content=WithActionContent(
                text="任务失败",
                actions=["image_search"],
                agent="image_search",
                cost_id=cost_id,
            ),
            create_ts=int(round(time.time() * 1000)),
        )
        self._push_message(failure_message)

        return {"output_text": self._FAIL_TEXT}

    def _package_result_node(self, state: ImageSearchWorkflowState) -> Dict[str, Any]:
        output_text = state.get("output_text", "")
        if not output_text:
            output_text = self._SUCCESS_TEXT
        return {"workflow_response": WorkflowResponse(select_result=output_text, relate_data=None)}


__all__ = ["ImageSearchGraph"]
