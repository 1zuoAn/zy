"""
Artifact 数据存储抽象层

设计原则：
- State 只存 ID + 元信息，实际数据通过此层查询
- 底层使用 MySQL 进行存储（默认 TTL 7 天，应用层过滤过期数据）

使用方式：
    store = get_artifact_store()
    store.save_payload(session_id, "list_001", [...])
    data = store.get_payload(session_id, "list_001")

MySQL 表结构（需要在目标数据库中创建）：

    CREATE TABLE IF NOT EXISTS zxy_artifact_store (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        session_id VARCHAR(128) NOT NULL,
        artifact_id VARCHAR(128) NOT NULL,
        artifact_type VARCHAR(64),
        description TEXT,
        content_cache MEDIUMTEXT,
        payload MEDIUMTEXT,
        meta JSON,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        expired_at DATETIME NOT NULL,
        UNIQUE KEY uk_session_artifact (session_id, artifact_id),
        INDEX idx_session_id (session_id),
        INDEX idx_expired_at (expired_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel
from sqlalchemy import text

from app.core.clients.db_client import mysql_session
from app.core.config.constants import DBAlias

# ArtifactStore 使用的数据库（olap_zxy_agent）
_DB_ALIAS = DBAlias.OLAP_ZXY_AGENT


def _normalize_payload(payload: Any) -> Any:
    """将各种类型的 payload 标准化为可 JSON 序列化的格式"""
    if isinstance(payload, BaseModel):
        return payload.model_dump()
    if is_dataclass(payload):
        return asdict(payload)
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return payload
    if isinstance(payload, tuple):
        return list(payload)
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="ignore")
    return payload


def _safe_json_dumps(payload: Any) -> str:
    """安全地将 payload 序列化为 JSON 字符串"""
    normalized = _normalize_payload(payload)
    if isinstance(normalized, str):
        try:
            return json.dumps(json.loads(normalized), ensure_ascii=False)
        except Exception:
            return json.dumps(normalized, ensure_ascii=False)
    return json.dumps(normalized, ensure_ascii=False, default=str)


class MySQLArtifactStore:
    """MySQL 实现"""

    _default_ttl = 60 * 60 * 24 * 7  # 7 天

    def __init__(self, ttl: int | None = None) -> None:
        self._ttl = ttl or self._default_ttl

    def save_payload(
        self,
        session_id: str,
        artifact_id: str,
        payload: Any,
        ttl: int | None = None,
    ) -> None:
        """保存 Artifact 数据到 MySQL"""
        ttl_seconds = ttl or self._ttl
        expired_at = datetime.now() + timedelta(seconds=ttl_seconds)

        # 从 payload 中提取结构化字段
        normalized = _normalize_payload(payload)
        if isinstance(normalized, dict):
            artifact_type = normalized.get("type")
            description = normalized.get("description")
            content_cache = normalized.get("content_cache")
            inner_payload = normalized.get("payload")
            meta = normalized.get("meta")
        else:
            artifact_type = None
            description = None
            content_cache = None
            inner_payload = normalized
            meta = None

        payload_json = _safe_json_dumps(inner_payload) if inner_payload else None
        meta_json = _safe_json_dumps(meta) if meta else None

        sql = text("""
            INSERT INTO zxy_artifact_store 
                (session_id, artifact_id, artifact_type, description, content_cache, payload, meta, expired_at)
            VALUES 
                (:session_id, :artifact_id, :artifact_type, :description, :content_cache, :payload, :meta, :expired_at)
            ON DUPLICATE KEY UPDATE
                artifact_type = VALUES(artifact_type),
                description = VALUES(description),
                content_cache = VALUES(content_cache),
                payload = VALUES(payload),
                meta = VALUES(meta),
                expired_at = VALUES(expired_at)
        """)

        try:
            with mysql_session(_DB_ALIAS) as session:
                session.execute(
                    sql,
                    {
                        "session_id": session_id,
                        "artifact_id": artifact_id,
                        "artifact_type": artifact_type,
                        "description": description,
                        "content_cache": content_cache,
                        "payload": payload_json,
                        "meta": meta_json,
                        "expired_at": expired_at,
                    },
                )
        except Exception as e:
            logger.error(f"[ArtifactStore] 保存失败: session={session_id}, artifact={artifact_id}, error={e}")
            raise

    def get_payload(self, session_id: str, artifact_id: str) -> Any | None:
        """获取 Artifact 数据（自动过滤过期数据）"""
        sql = text("""
            SELECT artifact_type, description, content_cache, payload, meta, created_at
            FROM zxy_artifact_store
            WHERE session_id = :session_id 
              AND artifact_id = :artifact_id
              AND expired_at > NOW()
        """)

        try:
            with mysql_session(_DB_ALIAS) as session:
                result = session.execute(
                    sql,
                    {"session_id": session_id, "artifact_id": artifact_id},
                )
                row = result.mappings().first()
                if row is None:
                    return None

                # 重建与 Redis 版本兼容的 record 结构
                record: dict[str, Any] = {
                    "id": artifact_id,
                    "type": row["artifact_type"],
                    "description": row["description"],
                    "content_cache": row["content_cache"],
                    "meta": json.loads(row["meta"]) if row["meta"] else None,
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                }

                # 解析 payload
                if row["payload"]:
                    try:
                        record["payload"] = json.loads(row["payload"])
                    except Exception:
                        record["payload"] = row["payload"]

                return record
        except Exception as e:
            logger.error(f"[ArtifactStore] 查询失败: session={session_id}, artifact={artifact_id}, error={e}")
            return None

    def delete_payload(self, session_id: str, artifact_id: str) -> None:
        """删除 Artifact 数据"""
        sql = text("""
            DELETE FROM zxy_artifact_store
            WHERE session_id = :session_id AND artifact_id = :artifact_id
        """)

        try:
            with mysql_session(_DB_ALIAS) as session:
                session.execute(
                    sql,
                    {"session_id": session_id, "artifact_id": artifact_id},
                )
        except Exception as e:
            logger.error(f"[ArtifactStore] 删除失败: session={session_id}, artifact={artifact_id}, error={e}")

    def get_list_data(
        self,
        session_id: str,
        artifact_id: str,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[Any], int, bool]:
        """
        获取列表数据（支持分页）

        Returns:
            (items, total, has_more)
        """
        payload = self.get_payload(session_id, artifact_id)
        if payload is None:
            return [], 0, False

        # 从 record 中提取实际的列表数据
        items = payload.get("payload", []) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            items = []

        total = len(items)
        start = (page - 1) * page_size
        end = start + page_size
        page_items = items[start:end]
        has_more = end < total

        return page_items, total, has_more

    def list_artifacts(self, session_id: str) -> list[dict[str, Any]]:
        """列出当前 session 的所有 artifacts（用于调试/错误提示）"""
        sql = text("""
            SELECT artifact_id, artifact_type, description
            FROM zxy_artifact_store
            WHERE session_id = :session_id AND expired_at > NOW()
        """)

        try:
            with mysql_session(_DB_ALIAS) as session:
                result = session.execute(sql, {"session_id": session_id})
                rows = result.mappings().all()

                return [
                    {
                        "id": row["artifact_id"],
                        "type": row["artifact_type"],
                        "description": row["description"],
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"[ArtifactStore] 列出 artifacts 失败: session={session_id}, error={e}")
            return []

    def create_artifact(
        self,
        session_id: str,
        artifact_type: str,
        description: str,
        meta: dict[str, Any],
        list_data: list | None = None,
        content_cache: str | None = None,
        storage_key: str | None = None,
    ) -> dict[str, Any]:
        """
        创建 Artifact（简化接口）

        Args:
            list_data: 列表数据（会存入 payload）
            content_cache: 图片描述或任务内容
            storage_key: 物理存储地址（图片 URL）

        Returns:
            Artifact 记录
        """
        artifact_id = (
            f"{artifact_type.replace('_', '')}_{int(time.time() * 1000)}_{uuid4().hex[:6]}"
        )

        record = {
            "id": artifact_id,
            "type": artifact_type,
            "description": description,
            "content_cache": content_cache,
            "meta": meta,
            "storage_key": storage_key,
            "payload": list_data,
            "created_at": datetime.now().isoformat(),
        }

        self.save_payload(session_id, artifact_id, record)
        return record


_store_lock = threading.Lock()
_store_instance: MySQLArtifactStore | None = None


def get_artifact_store() -> MySQLArtifactStore:
    """获取 MySQLArtifactStore 单例"""
    global _store_instance
    if _store_instance is None:
        with _store_lock:
            if _store_instance is None:
                _store_instance = MySQLArtifactStore()
    return _store_instance


__all__ = ["MySQLArtifactStore", "get_artifact_store"]
