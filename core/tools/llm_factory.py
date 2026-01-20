"""
LLM 工厂模块
统一管理多个 LLM 供应商的客户端创建和配置
"""
from functools import lru_cache
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from app.core.config.constants import LlmProvider
from loguru import logger

from app.config import settings


class LLMFactory:
    """LLM 工厂类"""

    @staticmethod
    def create_openai_llm(
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatOpenAI:
        """
        创建 OpenAI LLM 实例

        Args:
            model: 模型名称，默认使用配置中的值
            temperature: 温度参数，默认使用配置中的值
            max_tokens: 最大 token 数，默认使用配置中的值

        Returns:
            ChatOpenAI 实例
        """
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY 未配置")

        logger.info(f"创建 OpenAI LLM: {model or settings.llm_model}")

        return ChatOpenAI(
            model=model or settings.llm_model,
            temperature=temperature or settings.llm_temperature,
            max_tokens=max_tokens or settings.llm_max_tokens,
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_api_base,
            openai_organization=settings.openai_organization,
        )

    @staticmethod
    def create_openrouter_llm(
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatOpenAI:
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY 未配置")
        base = settings.openrouter_api_base or "https://openrouter.ai/api/v1"
        key = settings.openrouter_api_key
        logger.info(f"创建 OpenRouter LLM: {model or settings.llm_model}")
        return ChatOpenAI(
            model=model or settings.llm_model,
            temperature=temperature or settings.llm_temperature,
            max_tokens=max_tokens or settings.llm_max_tokens,
            openai_api_key=key,
            openai_api_base=base,
        )

    @staticmethod
    def create_huanxin_llm(
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatOpenAI:
        if not (settings.huanxin_api_key):
            raise ValueError("HUANXIN API KEY 未配置")
        base = settings.huanxin_api_base or "https://api.aigcark.com/v1"
        key = settings.huanxin_api_key
        logger.info(f"创建 Huanxin LLM: {model or settings.llm_model}")
        return ChatOpenAI(
            model=model or settings.llm_model,
            temperature=temperature or settings.llm_temperature,
            max_tokens=max_tokens or settings.llm_max_tokens,
            openai_api_key=key,
            openai_api_base=base,
        )

    @classmethod
    def create_llm(
        cls,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> BaseChatModel:
        """
        根据配置创建 LLM 实例

        Args:
            provider: LLM 供应商，默认使用配置中的值
            model: 模型名称，默认使用配置中的值
            temperature: 温度参数，默认使用配置中的值
            max_tokens: 最大 token 数，默认使用配置中的值

        Returns:
            对应供应商的 LLM 实例

        Raises:
            ValueError: 如果供应商不支持
        """
        provider = provider or settings.llm_provider
        provider = provider.value if isinstance(provider, LlmProvider) else provider

        logger.info(f"创建 LLM - Provider: {provider}, Model: {model or settings.llm_model}")

        if provider == "OPENAI":
            return cls.create_openai_llm(model, temperature, max_tokens)
        elif provider == "OPENROUTER":
            return cls.create_openrouter_llm(model, temperature, max_tokens)
        elif provider == "HUANXIN":
            return cls.create_huanxin_llm(model, temperature, max_tokens)
        else:
            raise ValueError(f"不支持的 LLM 供应商: {provider}")


@lru_cache()
def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> BaseChatModel:
    """
    获取默认的 LLM 实例（使用缓存）

    Args:
        provider: LLM 供应商
        model: 模型名称

    Returns:
        LLM 实例
    """
    return LLMFactory.create_llm(provider=provider, model=model)


# 便捷函数
def create_llm_with_callbacks(
    callbacks: Optional[list] = None,
    **kwargs,
) -> BaseChatModel:
    """
    创建带回调的 LLM 实例（用于执行追溯）

    Args:
        callbacks: 回调列表
        **kwargs: 其他参数

    Returns:
        LLM 实例
    """
    llm = LLMFactory.create_llm(**kwargs)
    if callbacks:
        llm.callbacks = callbacks
    return llm
