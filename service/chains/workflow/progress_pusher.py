# -*- coding: utf-8 -*-
"""
阶段化进度推送器

提供工作流进度的模板化推送功能，支持变量填充。
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from loguru import logger

from app.core.clients.redis_client import redis_client
from app.core.config.constants import RedisMessageKeyName, WorkflowMessageContentType
from app.schemas.entities.message.redis_message import (
    BaseRedisMessage,
    PhaseProgressContent, WithActionContent,
)
from app.schemas.entities.workflow.progress_template import WorkflowProgressTemplate
from app.schemas.request.workflow_request import WorkflowRequest


class PhaseProgressPusher:
    """
    阶段化进度推送器

    使用方式:
        pusher = PhaseProgressPusher(template=DOUYI_PROGRESS_TEMPLATE, request=req)
        pusher.push("选品任务规划", "start", content=llm_generated_text)
        pusher.push("商品筛选中", "running", variables={"browsed_count": 1000, ...})
        pusher.push("商品筛选中", "success")
        pusher.push("生成列表中", "fail", content="生成失败，原因是没有数据")
    """

    def __init__(
        self,
        template: WorkflowProgressTemplate,
        request: WorkflowRequest,
    ) -> None:
        self.template = template
        self.request = request

    def push(
        self,
        phase_name: str,
        status: str = "running",
        variables: Optional[Dict[str, Any]] = None,
        content: Optional[str] = None,
    ) -> None:
        """
        推送阶段进度

        Args:
            phase_name: 阶段名称
            status: 状态 (completed/failed)
            variables: 变量字典，用于填充模板
            content: 直接传入的内容文本（优先级最高）
        """
        rendered_content = self._render_content(phase_name, status, variables, content)
        self._push_message(phase_name, status, rendered_content)

    # 兼容旧 API
    def complete_phase(
        self,
        phase_name: str,
        variables: Optional[Dict[str, Any]] = None,
        content: Optional[str] = None,
    ) -> None:
        """[兼容] 完成指定阶段"""
        self.push(phase_name, "completed", variables, content)

    def fail_phase(
        self,
        phase_name: str,
        error_message: Optional[str],
    ) -> None:
        """[兼容] 标记阶段失败"""
        self.push(phase_name, "failed", content=error_message)

    def _render_content(
        self,
        phase_name: str,
        status: str,
        variables: Optional[Dict[str, Any]],
        content: Optional[str],
    ) -> str:
        """渲染阶段内容"""
        # content 优先
        if content is not None:
            return content.strip()

        # 使用模板渲染
        template = self.template.get_template(phase_name, status)
        if template is None:
            logger.warning(f"[PhaseProgressPusher] 阶段 '{phase_name}' 状态 '{status}' 无模板且未传入 content")
            return ""

        return self._render_template(template, variables or {})

    @staticmethod
    def _render_template(template: str, variables: Dict[str, Any]) -> str:
        """渲染模板字符串"""
        try:
            return template.format(**variables)
        except KeyError as e:
            logger.warning(f"[PhaseProgressPusher] 模板变量缺失: {e}")
            return template

    def _push_message(
        self,
        phase_name: str,
        status: str,
        content: Optional[str],
    ) -> None:
        """推送消息到 Redis"""
        progress_status = "completed" if status is None else status

        phase_content = PhaseProgressContent(
            task_name=self.template.task_name,
            phase_name=phase_name,
            status=progress_status,
            text=content,
        )

        message = BaseRedisMessage(
            session_id=self.request.session_id,
            reply_message_id=self.request.message_id,
            reply_id=f"reply_{self.request.message_id}",
            reply_seq=0,
            operate_id="阶段进度",
            status="RUNNING",
            content_type=WorkflowMessageContentType.PROCESSING.value,
            content=phase_content,
            create_ts=int(round(time.time() * 1000)),
        )

        logger.debug(f"[Progress Pusher] 发送Redis消息: {message.model_dump_json(ensure_ascii=False)}")
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            message.model_dump_json(),
        )

class ControlMessagePusher:
    """
    控制类消息推送器
    """
    def __init__(self, request: WorkflowRequest) -> None:
        self.request = request

    def push_cost_refund_message(self):
        message = BaseRedisMessage(
            session_id=self.request.session_id,
            reply_message_id=self.request.message_id,
            reply_id=f"reply_{self.request.message_id}",
            reply_seq=0,
            operate_id="扣点退费",
            status="RUNNING",
            content_type=WorkflowMessageContentType.COST_REFOUND.value,
            # 通过将cost_id置为None来让后端通过message_id进行扣点回退
            content=WithActionContent(),
            create_ts=int(round(time.time() * 1000)),
        )
        self._do_push(message)

    def _do_push(self, message: BaseRedisMessage) -> None:
        logger.debug(f"[Control Pusher] 发送Redis消息: {message.model_dump_json(ensure_ascii=False)}")
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            message.model_dump_json(),
        )


__all__ = ["PhaseProgressPusher"]
