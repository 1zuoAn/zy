"""
Redis 客户端模块
提供 Redis 连接和常用操作封装
"""
from typing import Any, cast

from loguru import logger
from redis import Redis

from app.config import settings


class RedisClient:
    """Redis 客户端类"""

    def __init__(self) -> None:
        self.sync_client: Redis | None = None

    def init_sync_client(self) -> Redis:
        """
        初始化同步 Redis 客户端

        Returns:
            Redis 客户端实例
        """
        if self.sync_client:
            return self.sync_client

        logger.info(f"初始化同步 Redis 客户端: {settings.redis_host}:{settings.redis_port}")

        self.sync_client = Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password,
            db=settings.redis_db,
            decode_responses=True,
            max_connections=settings.redis_max_connections,
        )

        try:
            self.sync_client.ping()
            logger.info("同步 Redis 客户端初始化完成")
        except Exception as e:
            logger.error(f"Redis 连接失败: {e}")
            raise

        return self.sync_client


    def get_sync_client(self) -> Redis:
        """
        获取同步 Redis 客户端

        Returns:
            Redis 客户端实例
        """
        if not self.sync_client:
            return self.init_sync_client()
        return self.sync_client

    # ============================================
    # 常用操作封装
    # ============================================

    def set_value(self, key: str, value: Any, ttl: int | None = None) -> bool:
        client = self.get_sync_client()
        return bool(cast(Any, client.set(key, value, ex=ttl)))

    def get_value(self, key: str) -> str | None:
        client = self.get_sync_client()
        return cast(str | None, client.get(key))

    def delete_key(self, key: str) -> int:
        client = self.get_sync_client()
        return cast(int, client.delete(key))

    def exists(self, key: str) -> bool:
        client = self.get_sync_client()
        count = cast(int, client.exists(key))
        return count > 0

    def set_ttl(self, key: str, ttl: int) -> bool:
        client = self.get_sync_client()
        return bool(cast(Any, client.expire(key, ttl)))

    def increment(self, key: str, amount: int = 1) -> int:
        client = self.get_sync_client()
        return cast(int, client.incrby(key, amount))

    def list_left_push(self, key: str, *values: Any) -> int:
        client = self.get_sync_client()
        return cast(int, client.lpush(key, *values))

    def list_right_push(self, key: str, *values: Any) -> int:
        client = self.get_sync_client()
        return cast(int, client.rpush(key, *values))

    def list_left_pop(self, key: str) -> str | None:
        client = self.get_sync_client()
        return cast(str | None, client.lpop(key))

    def list_right_pop(self, key: str) -> str | None:
        client = self.get_sync_client()
        return cast(str | None, client.rpop(key))

    def list_range(self, key: str, start: int = 0, end: int = -1) -> list[str]:
        client = self.get_sync_client()
        return cast(list[str], client.lrange(key, start, end))

    def list_length(self, key: str) -> int:
        client = self.get_sync_client()
        return cast(int, client.llen(key))

    def list_remove(self, key: str, count: int, value: Any) -> int:
        client = self.get_sync_client()
        return cast(int, client.lrem(key, count, value))

    def list_index(self, key: str, index: int) -> str | None:
        client = self.get_sync_client()
        return cast(str | None, client.lindex(key, index))

    def list_set(self, key: str, index: int, value: Any) -> bool:
        client = self.get_sync_client()
        return bool(cast(Any, client.lset(key, index, value)))

    def list_trim(self, key: str, start: int, end: int) -> bool:
        client = self.get_sync_client()
        return bool(cast(Any, client.ltrim(key, start, end)))

    def list_blocking_left_pop(self, keys: list[str], timeout: int = 0) -> list[str] | None:
        client = self.get_sync_client()
        return cast(list[str] | None, client.blpop(keys, timeout=timeout))

    def list_blocking_right_pop(self, keys: list[str], timeout: int = 0) -> list[str] | None:
        client = self.get_sync_client()
        return cast(list[str] | None, client.brpop(keys, timeout=timeout))

    def close(self) -> None:
        logger.info("关闭 Redis 连接...")
        if self.sync_client:
            try:
                self.sync_client.close()
            except Exception:
                pass
            logger.info("同步 Redis 客户端已关闭")


# 全局 Redis 客户端实例
redis_client = RedisClient()


# 依赖注入函数（同步）
def get_redis() -> Redis:
    return redis_client.get_sync_client()
