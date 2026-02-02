# -*- coding: utf-8 -*-
"""
店铺排行 API 封装
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import requests
from loguru import logger
from pydantic import BaseModel, Field

from app.config import settings
from app.core.errors import AppException, ErrorCode
from app.schemas.response.common import CommonResponse


class ShopRankRequest(BaseModel):
    """店铺排行请求参数"""
    class Config:
        populate_by_name = True

    industry: str = Field(description="行业")
    time_range_start: Optional[str] = Field(default=None, alias="timeRangeStart", description="时间范围开始")
    time_range_end: Optional[str] = Field(default=None, alias="timeRangeEnd", description="时间范围结束")
    rank_metric: str = Field(default="sales", alias="rankMetric", description="排行指标")
    limit: int = Field(default=50, alias="limit", description="返回数量")


class ShopRankItem(BaseModel):
    """店铺排行项"""
    class Config:
        populate_by_name = True

    shop_id: Optional[str] = Field(default=None, alias="shopId", description="店铺ID")
    shop_name: Optional[str] = Field(default=None, alias="shopName", description="店铺名称")
    rank: Optional[int] = Field(default=None, description="排名")
    sales: Optional[int] = Field(default=None, description="销量")
    growth_rate: Optional[float] = Field(default=None, alias="growthRate", description="增长率")


class ShopRankResponse(BaseModel):
    """店铺排行响应"""
    class Config:
        populate_by_name = True

    result_count: Optional[int] = Field(default=0, alias="resultCount", description="结果数量")
    result_list: Optional[List[ShopRankItem]] = Field(default_factory=list, alias="resultList", description="排行列表")


class ShopRankAPI:
    """店铺排行 API"""

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

        logger.debug(f"[ShopRankAPI] {method} {url}")
        if "json" in kwargs:
            logger.debug(f"[ShopRankAPI] Request Body: {json.dumps(kwargs['json'], ensure_ascii=False)}")

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
            logger.error(f"[ShopRankAPI] 请求超时: {url}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, "店铺排行服务请求超时")
        except Exception as e:
            logger.error(f"[ShopRankAPI] 请求异常: {e}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, str(e))

    def get_rankings(
        self,
        user_id: int,
        team_id: int,
        params: ShopRankRequest,
    ) -> ShopRankResponse:
        """获取店铺排行"""
        data = self._request(
            method="POST",
            path="/api/shop/rank",
            user_id=user_id,
            team_id=team_id,
            json=params.model_dump(by_alias=True, exclude_none=True),
        )

        response = CommonResponse[ShopRankResponse].model_validate(data)
        if not response.success:
            logger.warning(f"[ShopRankAPI] 业务错误: {response.error_code} - {response.error_desc}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, response.error_desc or "店铺排行服务返回错误")

        return response.result or ShopRankResponse()


# 全局实例
shop_rank_api_client = ShopRankAPI(
    base_url=settings.shop_rank_api_url,
    timeout=settings.shop_rank_api_timeout,
)
