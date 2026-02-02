"""
VLM (Vision Language Model) 服务
用于图片描述生成
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from loguru import logger

from app.core.config.constants import LlmModelName, LlmProvider
from app.core.tools import llm_factory


class VLMService:
    """VLM 图片描述生成服务"""

    def __init__(self):
        pass

    def describe(self, image_url: str) -> str:
        """
        调用 VLM 生成图片描述

        Args:
            image_url: 图片 OSS URL

        Returns:
            str: 图片的文字描述
        """
        try:
            llm = llm_factory.get_llm(
                LlmProvider.OPENROUTER.name,
                LlmModelName.OPENROUTER_GEMINI_2_5_FLASH.value,
            )

            # 构建 VLM 请求
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "请用中文详细描述这张图片的内容、风格和特点。"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
            )

            response = llm.invoke([message])
            caption = response.content.strip()

            logger.info(f"[VLM] 图片描述生成成功: {caption[:100]}")
            return caption

        except Exception as e:
            logger.exception(f"[VLM] 图片描述生成失败: {e}")
            # 降级: 返回默认描述
            return "用户上传的图片"


# 单例
_vlm_service_instance: VLMService | None = None


def get_vlm_service() -> VLMService:
    """获取 VLMService 单例"""
    global _vlm_service_instance
    if _vlm_service_instance is None:
        _vlm_service_instance = VLMService()
    return _vlm_service_instance


__all__ = ["VLMService", "get_vlm_service"]
