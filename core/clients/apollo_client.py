"""
Apollo 配置中心客户端

开发参考文档：https://www.apolloconfig.com/#/zh/client/other-language-client-user-guide

基于 Apollo Open API 实现的轻量级 Python 客户端，支持：
- Meta Server 服务发现
- 配置获取与本地缓存
- 长轮询配置变更监听（热更新）

注意：所有配置都支持热更新，Settings 类每次访问属性时都会从缓存读取最新值。
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import httpx
from loguru import logger


@dataclass
class ApolloConfig:
    """Apollo 连接配置"""

    meta_server_url: str  # Meta Server 地址
    app_id: str  # 应用 ID
    cluster: str = "default"  # 集群名称
    namespace: str = "application"  # namespace 名称
    timeout: int = 90  # 长轮询超时（秒）
    poll_interval: int = 5  # 轮询失败后重试间隔（秒）


class ApolloClient:
    """
    轻量级 Apollo 客户端

    基于 Apollo Open API 实现，支持：
    - Meta Server 服务发现
    - 配置获取与本地缓存
    - 长轮询配置变更监听
    """

    def __init__(self, config: ApolloConfig):
        self._config = config
        self._config_service_url: Optional[str] = None
        self._cache: Dict[str, str] = {}
        self._release_key: Optional[str] = None
        self._notification_id: int = -1
        self._lock = threading.RLock()
        self._running = False
        self._poll_thread: Optional[threading.Thread] = None
        self._http_client = httpx.Client(timeout=config.timeout + 10)
        self._callbacks: List[Callable[[Dict[str, str], Dict[str, str]], None]] = []

    def _discover_config_service(self) -> str:
        """通过 Meta Server 发现 Config Service 地址"""
        url = f"{self._config.meta_server_url.rstrip('/')}/services/config"
        try:
            resp = self._http_client.get(url, timeout=10)
            resp.raise_for_status()
            services = resp.json()
            if services:
                # 返回第一个可用的 Config Service
                service = services[0]
                home_url = service.get("homepageUrl", "")
                logger.info(f"[Apollo] 发现 Config Service: {home_url}")
                return home_url.rstrip("/")
        except Exception as e:
            logger.error(f"[Apollo] Meta Server 服务发现失败: {e}")
            raise RuntimeError(f"Apollo Meta Server 服务发现失败: {e}") from e
        raise RuntimeError("无可用的 Apollo Config Service")

    def _get_config_service_url(self) -> str:
        """获取 Config Service URL（带缓存）"""
        if not self._config_service_url:
            self._config_service_url = self._discover_config_service()
        return self._config_service_url

    def get_config(self) -> Dict[str, str]:
        """
        获取配置

        Returns:
            配置字典 {key: value}
        """
        base_url = self._get_config_service_url()
        url = (
            f"{base_url}/configs/{self._config.app_id}/"
            f"{self._config.cluster}/{self._config.namespace}"
        )

        try:
            resp = self._http_client.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            configurations = data.get("configurations", {})
            self._release_key = data.get("releaseKey")
            logger.debug(f"[Apollo] 获取配置成功，共 {len(configurations)} 项")
            return configurations
        except Exception as e:
            logger.error(f"[Apollo] 获取配置失败: {e}")
            raise RuntimeError(f"Apollo 获取配置失败: {e}") from e

    def get_cached_config(self) -> Dict[str, str]:
        """获取本地缓存的配置"""
        with self._lock:
            return self._cache.copy()

    def _long_poll(self) -> bool:
        """
        长轮询检查配置变更

        Returns:
            True 表示有变更，False 表示无变更或超时
        """
        base_url = self._get_config_service_url()
        url = f"{base_url}/notifications/v2"

        notifications = [
            {
                "namespaceName": self._config.namespace,
                "notificationId": self._notification_id,
            }
        ]

        params = {
            "appId": self._config.app_id,
            "cluster": self._config.cluster,
            "notifications": json.dumps(notifications),
        }

        try:
            resp = self._http_client.get(
                url,
                params=params,
                timeout=self._config.timeout,
            )

            if resp.status_code == 304:
                # 无变更
                return False

            if resp.status_code == 200:
                data = resp.json()
                for item in data:
                    if item.get("namespaceName") == self._config.namespace:
                        new_notification_id = item.get("notificationId", -1)
                        if new_notification_id != self._notification_id:
                            self._notification_id = new_notification_id
                            logger.info(
                                f"[Apollo] 检测到配置变更，notificationId: {new_notification_id}"
                            )
                            return True
            return False
        except httpx.TimeoutException:
            # 长轮询超时是正常的
            return False
        except Exception as e:
            logger.warning(f"[Apollo] 长轮询异常: {e}")
            return False

    def _poll_loop(self):
        """长轮询循环"""
        logger.info("[Apollo] 启动配置变更监听...")
        while self._running:
            try:
                if self._long_poll():
                    # 有变更，重新获取配置
                    old_config = self._cache.copy()
                    new_config = self.get_config()

                    with self._lock:
                        self._cache = new_config

                    # 触发回调
                    self._notify_callbacks(old_config, new_config)

            except Exception as e:
                logger.error(f"[Apollo] 轮询循环异常: {e}")
                time.sleep(self._config.poll_interval)

    def _notify_callbacks(
        self, old_config: Dict[str, str], new_config: Dict[str, str]
    ):
        """通知所有注册的回调"""
        for callback in self._callbacks:
            try:
                callback(old_config, new_config)
            except Exception as e:
                logger.error(f"[Apollo] 回调执行异常: {e}")

    def register_callback(
        self, callback: Callable[[Dict[str, str], Dict[str, str]], None]
    ):
        """注册配置变更回调"""
        self._callbacks.append(callback)

    def start(self) -> Dict[str, str]:
        """
        启动客户端

        Returns:
            初始配置字典
        """
        # 首次获取配置
        config = self.get_config()
        with self._lock:
            self._cache = config

        # 启动长轮询线程
        self._running = True
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="apollo-poll-thread",
            daemon=True,
        )
        self._poll_thread.start()

        logger.info(f"[Apollo] 客户端启动完成，加载 {len(config)} 项配置")
        return config

    def stop(self):
        """停止客户端"""
        self._running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=5)
        self._http_client.close()
        logger.info("[Apollo] 客户端已停止")


class ApolloConfigProvider:
    """
    Apollo 配置提供者

    封装 ApolloClient，提供类型安全的配置访问。
    所有配置都支持热更新，每次访问都从缓存读取最新值。
    """

    def __init__(self, client: ApolloClient):
        self._client = client
        # 注册配置变更日志回调
        self._client.register_callback(self._on_config_change)

    def get(self, key: str, default: Any = None) -> Optional[str]:
        """获取配置值"""
        config = self._client.get_cached_config()
        return config.get(key, default)

    def get_int(self, key: str, default: int = 0) -> int:
        """获取整数配置"""
        value = self.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """获取浮点数配置"""
        value = self.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """获取布尔配置"""
        value = self.get(key)
        if value is None:
            return default
        return value.lower() in ("true", "1", "yes", "on")

    def get_list(
        self, key: str, separator: str = ",", default: Optional[List[str]] = None
    ) -> List[str]:
        """获取列表配置"""
        value = self.get(key)
        if value is None:
            return default or []
        return [item.strip() for item in value.split(separator) if item.strip()]

    def get_all(self) -> Dict[str, str]:
        """获取所有配置"""
        return self._client.get_cached_config()

    def _on_config_change(
        self, old_config: Dict[str, str], new_config: Dict[str, str]
    ):
        """配置变更日志记录"""
        changed_keys = []

        for key in set(old_config.keys()) | set(new_config.keys()):
            old_val = old_config.get(key)
            new_val = new_config.get(key)
            if old_val != new_val:
                changed_keys.append(key)
                logger.info(f"[Apollo] 配置变更: {key} = {old_val} -> {new_val}")

        if changed_keys:
            logger.info(f"[Apollo] 配置变更完成，共 {len(changed_keys)} 项")
