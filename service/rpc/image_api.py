# -*- coding: utf-8 -*-
"""
图片生成 API 封装

包含:
- Gemini 图片生成
- OSS 图片上传
"""
from __future__ import annotations

import base64
import time
from typing import Any, Dict, List, Optional

import requests
from loguru import logger
from pydantic import BaseModel, Field

from app.config import settings
from app.core.clients.coze_loop_client import coze_loop_client_provider


class ImageGenerateResponse(BaseModel):
    """图片生成响应"""

    base64_image: Optional[str] = Field(default=None, description="Base64编码的图片")
    error_message: Optional[str] = Field(default=None, description="错误信息")
    success: bool = Field(default=False, description="是否成功")


class OSSUploadResponse(BaseModel):
    """OSS上传响应"""

    oss_url: Optional[str] = Field(default=None, description="OSS图片URL")
    success: bool = Field(default=False, description="是否成功")
    error_message: Optional[str] = Field(default=None, description="错误信息")


class ImageAPI:
    """图片生成 API - 封装 Gemini 图片生成"""

    def __init__(
        self,
        openrouter_api_key: str,
        openrouter_base_url: str = "https://openrouter.ai/api/v1",
        oss_upload_url: str = "",
        timeout: float = 300.0,
        image_api_url: str = "",
        image_api_key: str = "",
        image_api_timeout: float = 300.0,
        image_api_aspect_ratio: str = "3:4",
        image_api_image_size: str = "1K",
    ):
        self._openrouter_api_key = openrouter_api_key
        self._openrouter_base_url = openrouter_base_url.rstrip("/")
        self._oss_upload_url = oss_upload_url
        self._timeout = timeout
        self._image_api_url = image_api_url.rstrip("/") if image_api_url else ""
        self._image_api_key = image_api_key
        self._image_api_timeout = image_api_timeout
        self._image_api_aspect_ratio = image_api_aspect_ratio
        self._image_api_image_size = image_api_image_size

    def _use_gemini_api(self) -> bool:
        return bool(self._image_api_url and self._image_api_key)

    def _build_image_api_url(self, model: str) -> str:
        if not self._image_api_url:
            return ""
        if "{model}" in self._image_api_url:
            model_name = model.split("/")[-1]
            return self._image_api_url.format(model=model_name)
        return self._image_api_url

    def _build_gemini_payload(
        self,
        prompt: str,
        images: List[Dict[str, str]] | None = None,
    ) -> Dict[str, Any]:
        parts: List[Dict[str, Any]] = [{"text": prompt}]
        for image in images or []:
            parts.append({
                "inline_data": {
                    "mime_type": image.get("mime_type", "image/jpeg"),
                    "data": image.get("data", ""),
                }
            })
        return {
            "contents": [
                {
                    "parts": parts,
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {
                    "aspectRatio": self._image_api_aspect_ratio,
                    "imageSize": self._image_api_image_size,
                },
            },
        }

    def _extract_inline_image_data(self, result: dict) -> Optional[str]:
        candidates = result.get("candidates", [])
        for candidate in candidates:
            parts = candidate.get("content", {}).get("parts", [])
            for part in parts:
                inline_data = part.get("inlineData") or part.get("inline_data")
                if inline_data:
                    data = inline_data.get("data")
                    if data:
                        return data
        return None

    def _request_gemini(self, model: str, prompt: str, images: List[Dict[str, str]] | None = None) -> ImageGenerateResponse:
        url = self._build_image_api_url(model)
        if not url:
            return ImageGenerateResponse(
                success=False,
                error_message="图片生成接口未配置",
            )
        if not self._image_api_key:
            return ImageGenerateResponse(
                success=False,
                error_message="图片生成密钥未配置",
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._image_api_key}",
        }
        payload = self._build_gemini_payload(prompt, images)

        logger.debug(f"[ImageAPI] Gemini 请求: {prompt[:100]}..., images={len(images or [])}")

        try:
            resp = requests.post(
                url=url,
                headers=headers,
                json=payload,
                timeout=self._image_api_timeout,
            )
            resp.raise_for_status()
            result = resp.json()

            data = self._extract_inline_image_data(result)
            if data:
                return ImageGenerateResponse(success=True, base64_image=data)

            error_message = result.get("error", {}).get("message") or "响应中没有生成图片"
            return ImageGenerateResponse(success=False, error_message=error_message)
        except requests.exceptions.Timeout:
            logger.error("[ImageAPI] Gemini 请求超时")
            return ImageGenerateResponse(
                success=False,
                error_message="图片生成请求超时",
            )
        except Exception as e:
            logger.error(f"[ImageAPI] Gemini 请求异常: {e}")
            return ImageGenerateResponse(
                success=False,
                error_message=str(e),
            )

    def _fetch_image_base64(self, image_url: str) -> Optional[Dict[str, str]]:
        cleaned_url = image_url.replace("-internal", "")
        try:
            resp = requests.get(cleaned_url, timeout=20.0)
            resp.raise_for_status()
            data = base64.b64encode(resp.content).decode("utf-8")
            return {"data": data, "mime_type": "image/jpeg"}
        except Exception as e:
            logger.error(f"[ImageAPI] 下载图片失败: {cleaned_url}, error={e}")
            return None

    def generate_image_from_text(
        self,
        prompt: str,
        model: str = "google/gemini-3-pro-image-preview",
    ) -> ImageGenerateResponse:
        """
        文生图 - 根据文本提示词生成图片

        Args:
            prompt: 图片生成提示词
            model: 使用的模型

        Returns:
            ImageGenerateResponse: 包含 base64 图片或错误信息
        """
        client = coze_loop_client_provider.get_client()
        with client.start_span("文生图API调用", "tool") as span:
            start_time = time.time()
            span.set_input({"prompt": prompt[:200], "model": model})

            response = self._request_gemini(model=model, prompt=prompt)
            elapsed = time.time() - start_time
            span.set_output({
                "success": response.success,
                "elapsed_seconds": round(elapsed, 2),
                "error_message": response.error_message,
            })
            return response

    def generate_image_from_images(
        self,
        prompt: str,
        image_urls: List[str],
        model: str = "google/gemini-3-pro-image-preview",
    ) -> ImageGenerateResponse:
        """
        图生图 - 根据输入图片和提示词生成新图片

        Args:
            prompt: 编辑提示词
            image_urls: 输入图片URL列表（1-2张）
            model: 使用的模型

        Returns:
            ImageGenerateResponse: 包含 base64 图片或错误信息
        """
        client = coze_loop_client_provider.get_client()
        with client.start_span("图生图API调用", "tool") as span:
            start_time = time.time()
            span.set_input({
                "prompt": prompt[:200],
                "image_count": len(image_urls),
                "model": model,
            })

            images: List[Dict[str, str]] = []
            for img_url in image_urls:
                image_data = self._fetch_image_base64(img_url)
                if not image_data:
                    response = ImageGenerateResponse(
                        success=False,
                        error_message="图片下载失败",
                    )
                    elapsed = time.time() - start_time
                    span.set_output({
                        "success": response.success,
                        "elapsed_seconds": round(elapsed, 2),
                        "error_message": response.error_message,
                    })
                    return response
                images.append(image_data)

            response = self._request_gemini(model=model, prompt=prompt, images=images)
            elapsed = time.time() - start_time
            span.set_output({
                "success": response.success,
                "elapsed_seconds": round(elapsed, 2),
                "error_message": response.error_message,
            })
            return response

    def _extract_image_from_response(self, result: dict) -> ImageGenerateResponse:
        """从响应中提取图片（兼容 Gemini/OpenRouter）"""
        try:
            inline_data = self._extract_inline_image_data(result)
            if inline_data:
                return ImageGenerateResponse(
                    success=True,
                    base64_image=inline_data,
                )

            choices = result.get("choices", [])
            if not choices:
                return ImageGenerateResponse(
                    success=False,
                    error_message="响应中没有 choices",
                )

            message = choices[0].get("message", {})
            images = message.get("images", [])

            if not images:
                return ImageGenerateResponse(
                    success=False,
                    error_message="响应中没有生成图片，可能违反安全限制",
                )

            # 提取 base64 图片
            image_url = images[0].get("image_url", {}).get("url", "")
            if not image_url:
                return ImageGenerateResponse(
                    success=False,
                    error_message="图片URL为空",
                )

            # 格式: data:image/png;base64,xxxxx
            if "," in image_url:
                base64_data = image_url.split(",", 1)[1]
            else:
                base64_data = image_url

            return ImageGenerateResponse(
                success=True,
                base64_image=base64_data,
            )

        except Exception as e:
            logger.error(f"[ImageAPI] 解析响应失败: {e}")
            return ImageGenerateResponse(
                success=False,
                error_message=f"解析响应失败: {e}",
            )

    def upload_to_oss(self, base64_data: str) -> OSSUploadResponse:
        """
        上传 Base64 图片到 OSS

        Args:
            base64_data: Base64 编码的图片数据

        Returns:
            OSSUploadResponse: 包含 OSS URL 或错误信息
        """
        if not self._oss_upload_url:
            logger.error("[ImageAPI] OSS上传URL未配置")
            return OSSUploadResponse(
                success=False,
                error_message="OSS上传URL未配置",
            )

        # 添加 data URI 前缀
        if not base64_data.startswith("data:"):
            base64_with_prefix = f"data:image/png;base64,{base64_data}"
        else:
            base64_with_prefix = base64_data

        payload = {
            "base64Data": base64_with_prefix,
            "fileName": None,
            "ossConfig": {
                "endpoint": settings.oss_endpoint or "oss-cn-hangzhou-internal.aliyuncs.com",
                "accessKeyId": settings.oss_access_key_id or "",
                "accessKeySecret": settings.oss_access_key_secret or "",
                "bucketName": settings.oss_bucket_name or "zhiyi-image",
                "rootPathName": settings.oss_root_path or "zhixiaoyi_image_of_agent",
            },
        }

        logger.debug(f"[ImageAPI] OSS上传请求: {self._oss_upload_url}")

        try:
            resp = requests.post(
                url=self._oss_upload_url,
                json=payload,
                timeout=60.0,
            )
            resp.raise_for_status()
            result = resp.json()

            # 提取 OSS URL
            oss_url = result.get("result", {}).get("ossUrl", "")
            if not oss_url:
                return OSSUploadResponse(
                    success=False,
                    error_message="OSS上传成功但未返回URL",
                )

            # 处理 URL: 移除 -internal, 改为 https
            oss_url = oss_url.replace("-internal", "").replace("http:", "https:")

            return OSSUploadResponse(
                success=True,
                oss_url=oss_url,
            )

        except requests.exceptions.Timeout:
            logger.error(f"[ImageAPI] OSS上传超时")
            return OSSUploadResponse(
                success=False,
                error_message="OSS上传超时",
            )
        except Exception as e:
            logger.error(f"[ImageAPI] OSS上传异常: {e}")
            return OSSUploadResponse(
                success=False,
                error_message=str(e),
            )


# 延迟初始化的全局实例
_image_api_client: Optional[ImageAPI] = None


def _resolve_oss_upload_url() -> str:
    if settings.oss_upload_url:
        return settings.oss_upload_url
    if settings.fashion_parent_infra_api_url:
        return f"{settings.fashion_parent_infra_api_url.rstrip('/')}/upload/base64"
    return ""


def get_image_api_client() -> ImageAPI:
    """获取图片 API 客户端（延迟初始化）"""
    global _image_api_client
    if _image_api_client is None:
        _image_api_client = ImageAPI(
            openrouter_api_key=settings.openrouter_api_key or "",
            openrouter_base_url="https://openrouter.ai/api/v1",
            oss_upload_url=_resolve_oss_upload_url(),
            timeout=300.0,
            image_api_url=settings.image_api_url or "",
            image_api_key=settings.image_api_key or "",
            image_api_timeout=settings.image_api_timeout,
            image_api_aspect_ratio=settings.image_api_aspect_ratio,
            image_api_image_size=settings.image_api_image_size,
        )
    return _image_api_client


__all__ = [
    "ImageAPI",
    "ImageGenerateResponse",
    "OSSUploadResponse",
    "get_image_api_client",
]
