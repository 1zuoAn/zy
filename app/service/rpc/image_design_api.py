# -*- coding: utf-8 -*-
"""
图片生成/编辑 API 封装
"""
from __future__ import annotations

import json
from typing import List, Optional

import requests
from loguru import logger
from pydantic import BaseModel, Field

from app.config import settings
from app.core.errors import AppException, ErrorCode
from app.schemas.response.common import CommonResponse


class ImageCreateRequest(BaseModel):
    """图片生成请求"""
    class Config:
        populate_by_name = True

    prompt: str = Field(description="生成提示词")
    style: Optional[str] = Field(default=None, description="风格")
    size: Optional[str] = Field(default="1024x1024", description="图片尺寸")


class ImageEditRequest(BaseModel):
    """图片编辑请求"""
    class Config:
        populate_by_name = True

    prompt: str = Field(description="编辑提示词")
    image_urls: List[str] = Field(alias="imageUrls", description="待编辑图片URL列表")


class ImageDesignResponse(BaseModel):
    """图片设计响应"""
    class Config:
        populate_by_name = True

    image_url: Optional[str] = Field(default=None, alias="imageUrl", description="生成/编辑后的图片URL")
    image_urls: Optional[List[str]] = Field(default=None, alias="imageUrls", description="多张图片URL")


class ImageDesignAPI:
    """图片生成/编辑 API"""

    def __init__(self, base_url: str, timeout: float = 60.0):
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

        logger.debug(f"[ImageDesignAPI] {method} {url}")
        if "json" in kwargs:
            logger.debug(f"[ImageDesignAPI] Request Body: {json.dumps(kwargs['json'], ensure_ascii=False)}")

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
            logger.error(f"[ImageDesignAPI] 请求超时: {url}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, "图片设计服务请求超时")
        except Exception as e:
            logger.error(f"[ImageDesignAPI] 请求异常: {e}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, str(e))

    def create_image(
        self,
        user_id: int,
        team_id: int,
        params: ImageCreateRequest,
    ) -> ImageDesignResponse:
        """生成新图片"""
        data = self._request(
            method="POST",
            path="/api/image/create",
            user_id=user_id,
            team_id=team_id,
            json=params.model_dump(by_alias=True, exclude_none=True),
        )

        response = CommonResponse[ImageDesignResponse].model_validate(data)
        if not response.success:
            logger.warning(f"[ImageDesignAPI] 业务错误: {response.error_code} - {response.error_desc}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, response.error_desc or "图片生成服务返回错误")

        return response.result or ImageDesignResponse()

    def edit_image(
        self,
        user_id: int,
        team_id: int,
        params: ImageEditRequest,
    ) -> ImageDesignResponse:
        """编辑现有图片"""
        data = self._request(
            method="POST",
            path="/api/image/edit",
            user_id=user_id,
            team_id=team_id,
            json=params.model_dump(by_alias=True, exclude_none=True),
        )

        response = CommonResponse[ImageDesignResponse].model_validate(data)
        if not response.success:
            logger.warning(f"[ImageDesignAPI] 业务错误: {response.error_code} - {response.error_desc}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, response.error_desc or "图片编辑服务返回错误")

        return response.result or ImageDesignResponse()


# 全局实例
image_design_api_client = ImageDesignAPI(
    base_url=settings.image_design_api_url,
    timeout=settings.image_design_api_timeout,
)
