# -*- coding: utf-8 -*-
"""
定时任务 API 封装
"""
from __future__ import annotations

import json
from typing import Optional

import requests
from loguru import logger
from pydantic import BaseModel, Field

from app.config import settings
from app.core.errors import AppException, ErrorCode
from app.schemas.response.common import CommonResponse


class ScheduleTaskStartRequest(BaseModel):
    """定时任务启动通知请求"""
    class Config:
        populate_by_name = True

    session_id: str = Field(alias="sessionId", description="会话ID")
    message_id: str = Field(alias="messageId", description="消息ID")


class ScheduleTaskCreateRequest(BaseModel):
    """定时任务创建请求"""
    class Config:
        populate_by_name = True

    task_title: str = Field(alias="taskTitle", description="任务标题")
    task_content: str = Field(alias="taskContent", description="任务内容")
    cron_expression: str = Field(alias="cronExpression", description="5位cron表达式")
    session_id: str = Field(alias="sessionId", description="会话ID")
    user_id: int = Field(alias="userId", description="用户ID")
    team_id: int = Field(alias="teamId", description="团队ID")


class ScheduleTaskResponse(BaseModel):
    """定时任务响应"""
    class Config:
        populate_by_name = True

    task_id: Optional[str] = Field(default=None, alias="taskId", description="任务ID")
    status: Optional[str] = Field(default=None, description="任务状态")
    message: Optional[str] = Field(default=None, description="响应消息")


class ScheduleTaskAPI:
    """定时任务 API"""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self._base_url = base_url.rstrip("/") if base_url else ""
        self._timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        user_id: int,
        team_id: int,
        **kwargs,
    ) -> dict:
        url = f"{self._base_url}{path}"
        headers = kwargs.pop("headers", {})
        headers.update({
            "USER-ID": str(user_id),
            "TEAM-ID": str(team_id),
            "Content-Type": "application/json",
        })

        logger.debug(f"[ScheduleTaskAPI] {method} {url}")
        if "json" in kwargs:
            logger.debug(f"[ScheduleTaskAPI] Request Body: {json.dumps(kwargs['json'], ensure_ascii=False)}")

        try:
            resp = requests.request(
                method=method,
                url=url,
                headers=headers,
                timeout=self._timeout,
                **kwargs,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            logger.error(f"[ScheduleTaskAPI] 请求超时: {url}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, "定时任务服务请求超时")
        except Exception as e:
            logger.error(f"[ScheduleTaskAPI] 请求异常: {e}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, str(e))

    def start_task(
        self,
        user_id: int,
        team_id: int,
        params: ScheduleTaskStartRequest,
    ) -> ScheduleTaskResponse:
        """通知定时任务开始"""
        data = self._request(
            method="POST",
            path="/api/schedule/start",
            user_id=user_id,
            team_id=team_id,
            json=params.model_dump(by_alias=True, exclude_none=True),
        )

        response = CommonResponse[ScheduleTaskResponse].model_validate(data)
        if not response.success:
            logger.warning(f"[ScheduleTaskAPI] 业务错误: {response.error_code} - {response.error_desc}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, response.error_desc or "定时任务启动服务返回错误")

        return response.result or ScheduleTaskResponse()

    def create_task(
        self,
        user_id: int,
        team_id: int,
        params: ScheduleTaskCreateRequest,
    ) -> ScheduleTaskResponse:
        """创建定时任务"""
        data = self._request(
            method="POST",
            path="/api/schedule/create",
            user_id=user_id,
            team_id=team_id,
            json=params.model_dump(by_alias=True, exclude_none=True),
        )

        response = CommonResponse[ScheduleTaskResponse].model_validate(data)
        if not response.success:
            logger.warning(f"[ScheduleTaskAPI] 业务错误: {response.error_code} - {response.error_desc}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, response.error_desc or "定时任务创建服务返回错误")

        return response.result or ScheduleTaskResponse()


# 全局实例
schedule_task_api_client = ScheduleTaskAPI(
    base_url=settings.schedule_task_api_url,
    timeout=settings.schedule_task_api_timeout,
)
