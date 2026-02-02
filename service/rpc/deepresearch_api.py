# -*- coding: utf-8 -*-
"""
趋势报告(DeepResearch) API 封装
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests
from loguru import logger
from pydantic import BaseModel, Field

from app.config import settings
from app.core.errors import AppException, ErrorCode
from app.schemas.response.common import CommonResponse


class TrendsReportRequest(BaseModel):
    """趋势报告请求参数"""
    class Config:
        populate_by_name = True

    platform: str = Field(description="平台")
    industry: str = Field(description="行业")
    date_start: str = Field(alias="dateStart", description="开始日期")
    date_end: str = Field(alias="dateEnd", description="结束日期")
    root_category_id: Optional[str] = Field(default=None, alias="rootCategoryId", description="根类目ID")
    abroad_type: Optional[str] = Field(default=None, alias="abroadType", description="海外站点类型")


class TrendsReportResponse(BaseModel):
    """趋势报告响应"""
    class Config:
        populate_by_name = True

    report_id: Optional[str] = Field(default=None, alias="reportId", description="报告ID")
    report_url: Optional[str] = Field(default=None, alias="reportUrl", description="报告URL")
    summary: Optional[str] = Field(default=None, description="报告摘要")
    data: Optional[Dict[str, Any]] = Field(default=None, description="报告数据")


class DeepResearchAPI:
    """趋势报告 API"""

    def __init__(self, base_url: str, timeout: float = 120.0):
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

        logger.debug(f"[DeepResearchAPI] {method} {url}")
        if "json" in kwargs:
            logger.debug(f"[DeepResearchAPI] Request Body: {json.dumps(kwargs['json'], ensure_ascii=False)}")

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
            logger.error(f"[DeepResearchAPI] 请求超时: {url}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, "趋势报告服务请求超时")
        except Exception as e:
            logger.error(f"[DeepResearchAPI] 请求异常: {e}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, str(e))

    def generate_report(
        self,
        user_id: int,
        team_id: int,
        params: TrendsReportRequest,
    ) -> TrendsReportResponse:
        """生成趋势报告"""
        data = self._request(
            method="POST",
            path="/api/deepresearch/report",
            user_id=user_id,
            team_id=team_id,
            json=params.model_dump(by_alias=True, exclude_none=True),
        )

        response = CommonResponse[TrendsReportResponse].model_validate(data)
        if not response.success:
            logger.warning(f"[DeepResearchAPI] 业务错误: {response.error_code} - {response.error_desc}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, response.error_desc or "趋势报告服务返回错误")

        return response.result or TrendsReportResponse()


# 全局实例
deepresearch_api_client = DeepResearchAPI(
    base_url=settings.deepresearch_api_url,
    timeout=settings.deepresearch_api_timeout,
)
