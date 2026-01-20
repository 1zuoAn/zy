"""
数据库客户端模块
提供 MySQL 数据库连接和会话管理
"""
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Union

from loguru import logger
from sqlalchemy import TextClause, create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.core.config.constants import DBAlias

# 创建 Base 类用于 ORM 模型继承
Base = declarative_base()


class DatabaseClient:
    """数据库客户端类"""

    def __init__(self, url: str, alias):
        self.url = url
        self.alias = alias
        self.engine = None
        self.SessionLocal = None

    def init_sync_engine(self) -> None:
        logger.info(f"初始化同步数据库引擎: {self.alias}")

        self.engine = create_engine(
            self.url,
            pool_size=settings.mysql_pool_size,
            max_overflow=settings.mysql_max_overflow,
            pool_pre_ping=True,
            echo=settings.debug,
        )

        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

        logger.info("同步数据库引擎初始化完成")

    # 移除异步引擎与会话

    def get_session(self) -> Session:
        """
        获取同步数据库会话

        Returns:
            数据库会话实例
        """
        if not self.SessionLocal:
            self.init_sync_engine()

        return self.SessionLocal()

    def create_tables(self) -> None:
        """创建所有表（用于开发环境）"""
        if not self.engine:
            self.init_sync_engine()

        logger.info("创建数据库表...")
        Base.metadata.create_all(bind=self.engine)
        logger.info("数据库表创建完成")

    def close(self) -> None:
        logger.info("关闭数据库连接...")
        if self.engine:
            self.engine.dispose()
            logger.info("同步数据库引擎已关闭")



named_mysql_db_clients: Dict[str, DatabaseClient] = {}
named_pg_db_clients: Dict[str, DatabaseClient] = {}


def init_database_clients_from_settings() -> None:
    if settings.mysql_olap_zxy_agent_url and DBAlias.OLAP_ZXY_AGENT.value not in named_mysql_db_clients:
        client = DatabaseClient(url=settings.mysql_olap_zxy_agent_url, alias=DBAlias.OLAP_ZXY_AGENT.value)
        client.init_sync_engine()
        named_mysql_db_clients[DBAlias.OLAP_ZXY_AGENT.value] = client
        logger.info(f"注册 MySQL 客户端: {DBAlias.OLAP_ZXY_AGENT.value.lower()}")

    if settings.mysql_b_url and DBAlias.B.value not in named_mysql_db_clients:
        client = DatabaseClient(url=settings.mysql_b_url, alias=DBAlias.B.value)
        client.init_sync_engine()
        named_mysql_db_clients[DBAlias.B.value] = client
        logger.info(f"注册 MySQL 客户端: {DBAlias.B.value}")

    if settings.hologres_url and DBAlias.DB_ABROAD_AI.value not in named_pg_db_clients:
        client = DatabaseClient(url=settings.hologres_url, alias=DBAlias.DB_ABROAD_AI.value)
        client.init_sync_engine()
        named_pg_db_clients[DBAlias.DB_ABROAD_AI.value] = client
        logger.info(f"注册 Hologres 客户端: {DBAlias.DB_ABROAD_AI.value}")


def get_mysql_db_for(alias: str) -> Session:
    alias = alias.value if isinstance(alias, DBAlias) else alias
    client = named_mysql_db_clients.get(alias)
    if not client:
        raise KeyError(f"未注册数据库实例: {alias}")
    db = client.get_session()
    try:
        yield db
    finally:
        db.close()


def get_pg_db_for(alias: str) -> Session:
    alias = alias.value if isinstance(alias, DBAlias) else alias
    client = named_pg_db_clients.get(alias)
    if not client:
        raise KeyError(f"未注册数据库实例: {alias}")
    db = client.get_session()
    try:
        yield db
    finally:
        db.close()


def close_all_named_db_clients() -> None:
    for client in named_mysql_db_clients.values():
        try:
            client.close()
        except Exception:
            pass
    for client in named_pg_db_clients.values():
        try:
            client.close()
        except Exception:
            pass


# ============================================================
# 上下文管理器：推荐在工作流节点中使用
# ============================================================


@contextmanager
def mysql_session(alias: Union[DBAlias, str]) -> Generator[Session, None, None]:
    """
    MySQL 会话上下文管理器（自动提交/回滚/关闭）

    用于需要写入数据库的场景，退出 with 块时自动提交事务。
    如果发生异常，自动回滚事务。

    示例::

        with mysql_session(DBAlias.OLAP_ZXY_AGENT) as db:
            db.execute(text("INSERT INTO ..."), params)
            # 退出 with 块后自动提交

    Args:
        alias: 数据库别名（DBAlias 枚举或字符串）

    Yields:
        SQLAlchemy Session 对象
    """
    alias_str = alias.value if isinstance(alias, DBAlias) else alias
    client = named_mysql_db_clients.get(alias_str)
    if not client:
        raise KeyError(f"未注册数据库实例: {alias_str}")

    session = client.get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def mysql_session_readonly(alias: Union[DBAlias, str]) -> Generator[Session, None, None]:
    """
    MySQL 只读会话上下文管理器（不自动提交）

    用于只读查询场景，跳过 commit 提高性能。

    示例::

        with mysql_session_readonly(DBAlias.OLAP_ZXY_AGENT) as db:
            result = db.execute(text("SELECT ..."), params)
            data = result.fetchall()

    Args:
        alias: 数据库别名（DBAlias 枚举或字符串）

    Yields:
        SQLAlchemy Session 对象
    """
    alias_str = alias.value if isinstance(alias, DBAlias) else alias
    client = named_mysql_db_clients.get(alias_str)
    if not client:
        raise KeyError(f"未注册数据库实例: {alias_str}")

    session = client.get_session()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def pg_session(alias: Union[DBAlias, str]) -> Generator[Session, None, None]:
    """
    PostgreSQL 会话上下文管理器（自动提交/回滚/关闭）

    用于需要写入数据库的场景，退出 with 块时自动提交事务。

    示例::

        with pg_session(DBAlias.DB_ABROAD_AI) as db:
            db.execute(text("INSERT INTO ..."), params)

    Args:
        alias: 数据库别名（DBAlias 枚举或字符串）

    Yields:
        SQLAlchemy Session 对象
    """
    alias_str = alias.value if isinstance(alias, DBAlias) else alias
    client = named_pg_db_clients.get(alias_str)
    if not client:
        raise KeyError(f"未注册数据库实例: {alias_str}")

    session = client.get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def pg_session_readonly(alias: Union[DBAlias, str]) -> Generator[Session, None, None]:
    """
    PostgreSQL 只读会话上下文管理器（不自动提交）

    用于只读查询场景。

    示例::

        with pg_session_readonly(DBAlias.DB_ABROAD_AI) as db:
            result = db.execute(text("SELECT ..."), params)
            data = result.fetchall()

    Args:
        alias: 数据库别名（DBAlias 枚举或字符串）

    Yields:
        SQLAlchemy Session 对象
    """
    alias_str = alias.value if isinstance(alias, DBAlias) else alias
    client = named_pg_db_clients.get(alias_str)
    if not client:
        raise KeyError(f"未注册数据库实例: {alias_str}")

    session = client.get_session()
    try:
        yield session
    finally:
        session.close()
