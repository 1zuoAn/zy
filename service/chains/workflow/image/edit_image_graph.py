# -*- coding: utf-8 -*-
"""
图生图工作流 - LangGraph 版本

对齐 n8n Edit Image 工作流：
1. 接收 image (URL数组) + prompt
2. 根据图片数量选择调用方式（1张/2张）
3. 调用 Gemini 生成图片
4. 上传 OSS
5. 推送 Redis 消息
6. 埋点
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Dict

import requests

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from sqlalchemy import text

from app.core.clients.db_client import pg_session
from app.core.clients.redis_client import redis_client
from app.core.config.constants import (
    DBAlias,
    LlmModelName,
    RedisMessageKeyName,
)
from app.schemas.entities.message.redis_message import (
    BaseRedisMessage,
    CustomDataContent,
    TextMessageContent,
    WithActionContent,
)
from app.schemas.entities.workflow.graph_state import ImageWorkflowState
from app.schemas.response.workflow_response import WorkflowResponse
from app.service.chains.workflow.base_graph import BaseWorkflowGraph
from app.service.rpc.image_api import get_image_api_client


class EditImageGraph(BaseWorkflowGraph):
    """图生图工作流 - LangGraph 版本"""

    span_name = "图生图工作流"
    run_name = "edit-image-graph"

    def _build_graph(self) -> CompiledStateGraph:
        """构建工作流图"""
        graph = StateGraph(ImageWorkflowState)

        # 添加节点
        graph.add_node("init_state", self._init_state_node)
        graph.add_node("notify_start", self._notify_start_node)
        graph.add_node("generate_image", self._generate_image_node)
        graph.add_node("upload_oss", self._upload_oss_node)
        graph.add_node("notify_result", self._notify_result_node)
        graph.add_node("notify_failure", self._notify_failure_node)
        graph.add_node("track_usage", self._track_usage_node)
        graph.add_node("package", self._package_result_node)

        # 设置入口和边
        graph.set_entry_point("init_state")
        graph.add_edge("init_state", "notify_start")
        graph.add_edge("notify_start", "generate_image")

        # 条件分支：生成成功 -> 上传OSS，失败 -> 通知失败
        graph.add_conditional_edges(
            "generate_image",
            self._check_generation_result,
            {
                "success": "upload_oss",
                "failed": "notify_failure",
            },
        )

        graph.add_edge("upload_oss", "notify_result")
        graph.add_edge("notify_result", "track_usage")
        graph.add_edge("notify_failure", "package")
        graph.add_edge("track_usage", "package")
        graph.add_edge("package", END)

        return graph.compile()

    def _check_generation_result(self, state: ImageWorkflowState) -> str:
        """检查图片生成结果"""
        base64_image = state.get("generated_image_base64")
        if base64_image:
            return "success"
        return "failed"

    def _push_message(self, message: BaseRedisMessage, cost_id: str | None = None) -> None:
        payload = message.model_dump(exclude_none=True)
        if cost_id:
            payload["costId"] = cost_id
        redis_client.list_right_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            json.dumps(payload, ensure_ascii=False),
        )

    def _build_message(
        self,
        req: Any,
        operate_id: str,
        status: str,
        content_type: int,
        content: Any,
    ) -> BaseRedisMessage:
        return BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id=operate_id,
            status=status,
            content_type=content_type,
            content=content,
            create_ts=int(round(time.time() * 1000)),
        )

    # ==================== 节点实现 ====================

    def _init_state_node(self, state: ImageWorkflowState) -> Dict[str, Any]:
        """初始化状态"""
        req = state["request"]

        # 从 request 中提取参数
        # n8n 中 image 是用 # 分隔的字符串，需要解析为数组
        raw_image = req.input_images if req.input_images is not None else ""
        if isinstance(raw_image, str):
            input_images = raw_image.split("#")
        elif isinstance(raw_image, list):
            input_images = raw_image
        else:
            raw_image = ""
            input_images = []

        # 编辑提示词（移除引号）
        edit_prompt = (req.edit_prompt or req.user_query or "").replace('"', "").replace("\n", "").replace("\r", "")
        cost_id = f"{req.session_id}_{int(time.time() * 1000)}"

        logger.debug(f"[图生图] 输入图片数量: {len(input_images)}, 提示词: {edit_prompt[:50]}...")

        return {
            "input_images": input_images,
            "raw_input_images": raw_image,
            "edit_prompt": edit_prompt,
            "cost_id": cost_id,
        }

    def _notify_start_node(self, state: ImageWorkflowState) -> Dict[str, Any]:
        """通知任务开始"""
        req = state["request"]
        cost_id = state.get("cost_id")

        # 推送任务开始消息
        start_message = self._build_message(
            req=req,
            operate_id="收到图生图任务",
            status="RUNNING",
            content_type=2,
            content=WithActionContent(
                text="收到，开始图片设计任务...",
                actions=["images"],
                agent="create_image",
            ),
        )
        self._push_message(start_message, cost_id=cost_id)

        # 推送启用助手消息
        assist_message = self._build_message(
            req=req,
            operate_id="正在启用图生图",
            status="RUNNING",
            content_type=1,
            content=TextMessageContent(text="正在启用图片设计助手"),
        )
        self._push_message(assist_message)

        # 推送正在生成消息
        generating_message = self._build_message(
            req=req,
            operate_id="正在图生图",
            status="RUNNING",
            content_type=1,
            content=TextMessageContent(text="正在生成图片"),
        )
        self._push_message(generating_message)

        return {}

    def _generate_image_node(self, state: ImageWorkflowState) -> Dict[str, Any]:
        """生成图片"""
        input_images = state.get("input_images", [])
        edit_prompt = state.get("edit_prompt", "")
        image_client = get_image_api_client()

        # 根据图片数量调用不同接口（对齐 n8n If1 逻辑）
        # n8n: image.length == 2 且第二张不为空时用两张图，否则用一张
        if len(input_images) == 2 and input_images[1]:
            images_to_use = input_images[:2]
        else:
            images_to_use = input_images[:1] if input_images else []

        if not images_to_use:
            logger.error("[图生图] 没有输入图片")
            return {"generated_image_base64": None}

        response = image_client.generate_image_from_images(
            prompt=edit_prompt,
            image_urls=images_to_use,
            model=LlmModelName.OPENROUTER_GEMINI_3_PRO_IMAGE.value,
        )

        if response.success and response.base64_image:
            logger.info(f"[图生图] 图片生成成功")
            return {"generated_image_base64": response.base64_image}
        logger.warning(f"[图生图] 图片生成失败: {response.error_message}")
        return {"generated_image_base64": None}

    def _upload_oss_node(self, state: ImageWorkflowState) -> Dict[str, Any]:
        """上传图片到 OSS"""
        base64_image = state.get("generated_image_base64", "")

        image_client = get_image_api_client()
        response = image_client.upload_to_oss(base64_image)

        if response.success and response.oss_url:
            logger.info(f"[图生图] OSS上传成功: {response.oss_url}")
            return {"output_image_url": response.oss_url}
        else:
            logger.error(f"[图生图] OSS上传失败: {response.error_message}")
            return {"output_image_url": None}

    def _notify_result_node(self, state: ImageWorkflowState) -> Dict[str, Any]:
        """推送结果消息"""
        req = state["request"]
        output_url = state.get("output_image_url")

        if not output_url:
            return {}

        # 推送图片已生成消息
        result_message = self._build_message(
            req=req,
            operate_id="报告图生图结果",
            status="RUNNING",
            content_type=1,
            content=TextMessageContent(text="图片已生成"),
        )
        self._push_message(result_message)

        # 推送输出参数消息 (content_type=5)
        output_message = self._build_message(
            req=req,
            operate_id="图生图",
            status="END",
            content_type=5,
            content=CustomDataContent(
                data={
                    "actions": ["images"],
                    "images": [output_url],
                }
            ),
        )
        self._push_message(output_message)

        return {}

    def _notify_failure_node(self, state: ImageWorkflowState) -> Dict[str, Any]:
        """推送失败消息"""
        req = state["request"]
        cost_id = state.get("cost_id")

        failure_message = self._build_message(
            req=req,
            operate_id="生图任务失败",
            status="RUNNING",
            content_type=10,
            content=WithActionContent(
                text="图片设计任务失败",
                actions=["images"],
                agent="create_image",
            ),
        )
        self._push_message(failure_message, cost_id=cost_id)

        return {}

    def _track_usage_node(self, state: ImageWorkflowState) -> Dict[str, Any]:
        """埋点记录"""
        req = state["request"]
        raw_input_images = state.get("raw_input_images", "")
        edit_prompt = state.get("edit_prompt", "")
        output_url = state.get("output_image_url", "")
        if not output_url:
            return {}

        if isinstance(raw_input_images, list):
            input_link = "#".join(raw_input_images)
        else:
            input_link = raw_input_images or ""

        try:
            resp = requests.get(output_url, timeout=20.0)
            resp.raise_for_status()
            resp.close()
        except Exception as e:
            logger.warning(f"[图生图] 输出图片获取失败: {e}")

        try:
            with pg_session(DBAlias.DB_ABROAD_AI.value) as session:
                session.execute(
                    text("""
                        INSERT INTO zxy_edit_image_user_data_info(
                            session_id, prompt, input_image_link, output_image_link, time
                        )
                        VALUES (:session_id, :prompt, :input_link, :output_link, :time)
                    """),
                    {
                        "session_id": req.session_id,
                        "prompt": edit_prompt,
                        "input_link": input_link,
                        "output_link": output_url,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
        except Exception as e:
            logger.warning(f"[图生图] 埋点失败: {e}")
        return {}

    def _package_result_node(self, state: ImageWorkflowState) -> Dict[str, Any]:
        """封装返回结果"""
        output_url = state.get("output_image_url")

        if output_url:
            response = WorkflowResponse(
                select_result="图片设计任务已完成",
                relate_data=output_url,
            )
        else:
            response = WorkflowResponse(
                select_result="图片生成失败，可能违反安全限制，请调整输入后重试",
                relate_data=None,
            )
        return {"workflow_response": response}


__all__ = ["EditImageGraph"]
