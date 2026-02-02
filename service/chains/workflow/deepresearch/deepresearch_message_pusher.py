# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2026/1/13 16:38
# @File     : deepresearch_message_pusher.py
"""
数据洞察专用消息推送器
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from app.core.clients.redis_client import redis_client
from app.core.config.constants import RedisMessageKeyName, WorkflowMessageContentType
from app.schemas.entities.export import ExcelData
from app.schemas.entities.message.redis_message import (
    BaseRedisMessage, PhaseProgressContent, ParameterData, ParameterDataContent, WithActionContent,
)
from app.service.chains.templates.deepresearch.deepresearch_progress_template import (
    DeepresearchProgressTemplate,
)
from app.schemas.request.workflow_request import WorkflowRequest
from app.service.chains.templates.deepresearch import (
    ABROAD_DEEPRESEARCH_TEMPLATE,
    DOUYI_DEEPRESEARCH_TEMPLATE,
    ZHIYI_DEEPRESEARCH_TEMPLATE,
    INS_DEEPRESEARCH_TEMPLATE,
)


def _get_template_by_workflow_type(workflow_type: str) -> DeepresearchProgressTemplate:
    """根据工作流类型获取对应的模板"""
    templates = {
        "douyi": DOUYI_DEEPRESEARCH_TEMPLATE,
        "zhiyi": ZHIYI_DEEPRESEARCH_TEMPLATE,
        "abroad": ABROAD_DEEPRESEARCH_TEMPLATE,
        "ins": INS_DEEPRESEARCH_TEMPLATE,
    }
    return templates.get(workflow_type, DOUYI_DEEPRESEARCH_TEMPLATE)


class DeepresearchMessagePusher:
    """
    数据洞察专用消息推送器
    """

    def __init__(
        self,
        request: WorkflowRequest,
        is_thinking: bool,
        workflow_type: str,
    ) -> None:
        self.request = request
        self.is_thinking = is_thinking
        self.workflow_type = workflow_type
        self.template = _get_template_by_workflow_type(workflow_type)

    def push_phase(
        self,
        phase_name: str,
        variables: Optional[Dict[str, Any]] = None,
        content: Optional[str] = None,
    ) -> None:
        """
        推送阶段进度（单次推送，status 固定为 completed）

        Args:
            phase_name: 阶段名称
            variables: 变量字典，用于填充模板
            content: 直接传入的内容文本（优先级最高）
        """
        rendered_content = self._render_content(phase_name, variables, content)
        self._push_message(phase_name=phase_name, content=rendered_content)

    def push_data_message(
        self,
        data_type: str,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        推送附加数据消息（预留接口）

        Args:
            data_type: 数据类型标识
            data: 数据内容
            metadata: 元数据
        """
        # 预留接口，后续可以实现 content_type=12 的数据消息
        logger.debug(
            f"[DeepresearchMessagePusher] push_data_message called: "
            f"data_type={data_type}, metadata={metadata}"
        )

    def _render_content(
        self,
        phase_name: str,
        variables: Optional[Dict[str, Any]],
        content: Optional[str],
    ) -> str:
        """渲染阶段内容"""
        # content 优先
        if content is not None:
            return content.strip()

        # 使用模板渲染
        template = self.template.get_template(phase_name, self.is_thinking)
        if template is None:
            logger.warning(
                f"[DeepresearchMessagePusher] 阶段 '{phase_name}' 无模板且未传入 content"
            )
            return ""

        return self._render_template(template, variables or {})

    @staticmethod
    def _render_template(template: str, variables: Dict[str, Any]) -> str:
        """渲染模板字符串"""
        try:
            return template.format(**variables)
        except KeyError as e:
            logger.warning(f"[DeepresearchMessagePusher] 模板变量缺失: {e}")
            return template

    def _push_message(self, phase_name: str, content: str) -> None:
        """推送消息到 Redis"""
        progress_content = PhaseProgressContent(
            task_name=self.template.task_name,
            phase_name=phase_name,
            status="completed",
            text=content,
        )

        message = BaseRedisMessage(
            session_id=self.request.session_id,
            reply_message_id=self.request.message_id,
            reply_id=f"reply_{self.request.message_id}",
            reply_seq=0,
            operate_id="数据洞察进度",
            status="RUNNING",
            content_type=WorkflowMessageContentType.PROCESSING.value,
            content=progress_content,
            create_ts=int(round(time.time() * 1000)),
        )

        logger.debug(
            f"[DeepresearchMessagePusher] 发送Redis消息: {message.model_dump_json(ensure_ascii=False)}"
        )
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            message.model_dump_json(),
        )

    def push_task_start_msg(self):
        """
        推送任务开始消息
        主要是需要获取agent枚举类型
        :return:
        """
        content = WithActionContent(
            text="收到，开始趋势分析任务...",
            actions=["report"],
            agent="deepresearch",
        )

        message = BaseRedisMessage(
            session_id=self.request.session_id,
            reply_message_id=self.request.message_id,
            reply_id=f"reply_{self.request.message_id}",
            reply_seq=0,
            operate_id="收到deepresearch任务",
            status="RUNNING",
            content_type=WorkflowMessageContentType.PRE_TEXT.value,
            content=content,
            create_ts=int(round(time.time() * 1000)),
        )
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            message.model_dump_json(ensure_ascii=False),
        )
        logger.debug(f"[DeepresearchMessagePusher] 发送Redis消息: {message.model_dump_json(ensure_ascii=False)}")

    def push_task_finish_status_msg(self):
        parameter_data = ParameterData(task_status=1)

        message = BaseRedisMessage(
            session_id=self.request.session_id,
            reply_message_id=self.request.message_id,
            reply_id=f"reply_{self.request.message_id}",
            reply_seq=0,
            operate_id="任务状态",
            status="RUNNING",
            content=ParameterDataContent(data=parameter_data),
            content_type=WorkflowMessageContentType.TASK_STATUS.value,
            create_ts=int(round(time.time() * 1000)),
        )
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            message.model_dump_json(ensure_ascii=False),
        )
        logger.debug(f"[DeepresearchMessagePusher] 发送Redis消息: {message.model_dump_json(ensure_ascii=False)}")


    def push_report_and_excel_data(self, entity_type: int | None, report_text: str, excel_data_list: List[ExcelData]) -> None:
        """
        推送 Excel 导出数据

        Args:
            entity_type: 实体类型，报告中没有商品卡则不需要，如ins报告
            report_text: 报告的文本内容
            excel_data_list: ExcelData 列表
        """
        if not report_text:
            logger.warning("[DeepresearchMessagePusher]push_excel_data()输入的趋势报告为空！")
            return
        if not excel_data_list:
            logger.warning("[DeepresearchMessagePusher]push_excel_data()输入的excel数据列表为空！")
            excel_data_list = []

        # 将整个列表序列化为 JSON array
        json_array = [data.model_dump(by_alias=True) for data in excel_data_list]
        request_body_json = json.dumps(json_array, ensure_ascii=False)

        # 构建 ParameterDataContent
        param_data = ParameterData(
            entity_type=entity_type,
            request_body=request_body_json,
            actions=["report"],
            title="数据洞察报告",
            markdown=report_text,
        )
        param_content = ParameterDataContent(data=param_data)

        # 构建消息
        message = BaseRedisMessage(
            session_id=self.request.session_id,
            reply_message_id=self.request.message_id,
            reply_id=f"reply_{self.request.message_id}",
            reply_seq=0,
            operate_id="数据洞察导出",
            status="END",
            content_type=WorkflowMessageContentType.RESULT.value,
            content=param_content,
            create_ts=int(round(time.time() * 1000)),
        )

        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            message.model_dump_json(ensure_ascii=False),
        )
        logger.debug(f"[DeepresearchMessagePusher] 发送Redis消息: {message.model_dump_json(ensure_ascii=False)}")
        logger.debug(f"[DeepresearchMessagePusher] 生成报告与Excel导出数据已推送, 导出count={len(excel_data_list)}")


__all__ = ["DeepresearchMessagePusher"]
