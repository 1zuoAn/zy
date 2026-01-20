"""
配置中心客户端模块

提供统一的配置中心访问接口，支持：
- Apollo 配置中心（推荐，已完整实现）
- Nacos 配置中心（可选，需要额外配置）

注意：主要的配置管理通过 app.config 模块的 SettingsManager 实现，
本模块提供额外的配置中心访问能力，用于动态获取非 Settings 类管理的配置。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from loguru import logger

if TYPE_CHECKING:
    from app.core.clients.apollo_client import ApolloConfigProvider


class ConfigClient:
    """
    统一配置客户端

    提供对配置中心的统一访问接口。
    对于 Apollo，优先使用 app.config 中的 SettingsManager。
    """

    def __init__(self):
        """初始化配置客户端"""
        self._apollo_provider: Optional["ApolloConfigProvider"] = None

    def set_apollo_provider(self, provider: "ApolloConfigProvider") -> None:
        """
        设置 Apollo 配置提供者

        Args:
            provider: ApolloConfigProvider 实例
        """
        self._apollo_provider = provider
        logger.info("[ConfigClient] Apollo 配置提供者已设置")

    def get_config(
        self,
        key: str,
        default: Optional[str] = None,
        source: str = "apollo",
    ) -> Optional[str]:
        """
        获取配置

        Args:
            key: 配置键
            default: 默认值
            source: 配置来源，目前支持 apollo

        Returns:
            配置值
        """
        if source == "apollo":
            return self._get_apollo_config(key, default)
        else:
            logger.warning(f"[ConfigClient] 不支持的配置来源: {source}")
            return default

    def _get_apollo_config(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """从 Apollo 获取配置"""
        if not self._apollo_provider:
            logger.warning("[ConfigClient] Apollo 配置提供者未初始化")
            return default

        return self._apollo_provider.get(key, default)

    def get_int(self, key: str, default: int = 0, source: str = "apollo") -> int:
        """获取整数配置"""
        if source == "apollo" and self._apollo_provider:
            return self._apollo_provider.get_int(key, default)
        return default

    def get_float(self, key: str, default: float = 0.0, source: str = "apollo") -> float:
        """获取浮点数配置"""
        if source == "apollo" and self._apollo_provider:
            return self._apollo_provider.get_float(key, default)
        return default

    def get_bool(self, key: str, default: bool = False, source: str = "apollo") -> bool:
        """获取布尔配置"""
        if source == "apollo" and self._apollo_provider:
            return self._apollo_provider.get_bool(key, default)
        return default


# 全局配置客户端实例
config_client = ConfigClient()


def get_config_client() -> ConfigClient:
    """
    获取配置客户端实例

    Returns:
        配置客户端实例
    """
    return config_client


def init_config_client_from_settings() -> None:
    """
    从 settings 初始化配置客户端

    在应用启动时调用，将 Apollo 配置提供者注入到 ConfigClient
    """
    from app.config import settings

    apollo_provider = settings.get_apollo_provider()
    if apollo_provider:
        config_client.set_apollo_provider(apollo_provider)
        logger.info("[ConfigClient] Apollo 配置提供者已初始化")
