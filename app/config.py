"""
配置管理模块 - 基于 Apollo 配置中心

使用 __getattr__ 动态代理，新增配置只需在 _DEFAULTS 中添加一行。
"""
from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any, Dict, Optional

from loguru import logger

if TYPE_CHECKING:
    from app.core.clients.apollo_client import ApolloClient, ApolloConfigProvider

    class Settings:
        """类型存根 - 仅用于 IDE 静态分析"""

        # 应用配置
        app_name: str
        app_version: str
        debug: bool
        environment: str
        # API 配置
        api_prefix: str
        host: str
        port: int
        # LLM 配置
        llm_provider: str
        llm_model: str
        llm_temperature: float
        llm_max_tokens: int
        # OpenAI / OpenRouter / HuanXin / Dashscope
        openai_api_key: Optional[str]
        openai_api_base: Optional[str]
        openai_organization: Optional[str]
        openrouter_api_key: Optional[str]
        openrouter_api_base: str
        huanxin_api_key: Optional[str]
        huanxin_api_base: Optional[str]
        dashscope_api_key: Optional[str]
        dashscope_api_base: Optional[str]
        # CozeLoop 配置
        cozeloop_workspace_id: Optional[str]
        cozeloop_api_token: Optional[str]
        cozeloop_prompt_label: str
        # 图片生成 API
        image_api_url: Optional[str]
        image_api_key: Optional[str]
        image_api_timeout: float
        image_api_aspect_ratio: str
        image_api_image_size: str
        # MySQL 通用配置
        mysql_pool_size: int
        mysql_max_overflow: int
        # MySQL olap_zxy_agent
        mysql_olap_zxy_agent_host: Optional[str]
        mysql_olap_zxy_agent_port: Optional[int]
        mysql_olap_zxy_agent_user: Optional[str]
        mysql_olap_zxy_agent_password: Optional[str]
        mysql_olap_zxy_agent_database: Optional[str]
        # MySQL B
        mysql_b_host: Optional[str]
        mysql_b_port: Optional[int]
        mysql_b_user: Optional[str]
        mysql_b_password: Optional[str]
        mysql_b_database: Optional[str]
        # Redis 配置
        redis_host: str
        redis_port: int
        redis_password: Optional[str]
        redis_db: int
        redis_max_connections: int
        # Hologres 配置
        hologres_host: Optional[str]
        hologres_port: Optional[int]
        hologres_user: Optional[str]
        hologres_password: Optional[str]
        hologres_database: Optional[str]
        # Es 配置
        es_host: Optional[str]
        es_user: Optional[str]
        es_password: Optional[str]
        # 日志配置
        log_level: str
        log_format: str
        log_file: str
        log_rotation: str
        log_retention: str
        # 重试配置
        retry_max_attempts: int
        retry_wait_exponential_multiplier: int
        retry_wait_exponential_max: int
        # CORS 配置
        cors_origins: list[str]
        cors_allow_credentials: bool
        cors_allow_methods: list[str]
        cors_allow_headers: list[str]
        # 线程池配置
        thread_pool_max_workers: int
        # 外部系统依赖
        abroad_api_url: str
        abroad_api_timeout: float
        zxh_api_url: str
        zxh_api_timeout: float
        zhikuan_api_url: str
        zhikuan_api_timeout: float
        # douyi_api_url 已废弃，抖衣和知衣共用 zhiyi_api_url
        zhiyi_api_url: str
        zhiyi_api_timeout: float
        fashion_parent_infra_api_url: str
        fashion_parent_infra_api_timeout: float
        # Dify API
        dify_api_url: str
        dify_api_key: Optional[str]
        dify_property_workflow_id: Optional[str]
        dify_zhiyi_property_workflow_id: Optional[str]  # 知衣属性解析工作流ID
        dify_zhiyi_shop_workflow_id: Optional[str]  # 知衣店铺检索工作流ID（品牌->shopId）
        # 火山引擎知识库配置
        volcengine_kb_api_url: str
        volcengine_kb_api_key: str
        volcengine_kb_api_timeout: float
        # 主工作流配置
        main_workflow_image_retry_attempts: int
        main_workflow_image_max_workers: int
        main_workflow_summary_max_length: int
        # OSS 配置
        oss_upload_url: str
        oss_endpoint: str
        oss_access_key_id: str
        oss_access_key_secret: str
        oss_bucket_name: str
        oss_root_path: str
        # 计算属性
        mysql_olap_zxy_agent_url: Optional[str]
        mysql_b_url: Optional[str]
        hologres_url: Optional[str]

        def get_apollo_provider(self) -> Optional["ApolloConfigProvider"]: ...
        def stop(self) -> None: ...


# Apollo 连接配置（支持环境变量覆盖）
# 环境通过不同的 Meta Server 地址区分：
#   - GRAY: http://192.168.200.37:8022
#   - PROD: http://192.168.200.37:8021
APOLLO_META_SERVER_URL = os.getenv("APOLLO_META_SERVER_URL", "http://192.168.200.37:8022")
APOLLO_APP_ID = os.getenv("APOLLO_APP_ID", "zxy-workflow-py")
APOLLO_CLUSTER = os.getenv("APOLLO_CLUSTER", "default")
APOLLO_NAMESPACE = os.getenv("APOLLO_NAMESPACE", "application")


class Settings:
    """
    配置类 - 从 Apollo 动态读取

    使用 __getattr__ 动态代理，新增配置只需在 _DEFAULTS 中添加一行。
    类型转换基于默认值类型自动推断。
    """

    # 配置项的默认值（唯一的配置定义位置）
    _DEFAULTS: Dict[str, Any] = {
        # 应用配置
        "app_name": "ZXY Workflow",
        "app_version": "0.1.0",
        "debug": True,
        "environment": "gray",
        # API 配置
        "api_prefix": "/api/v1",
        "host": "0.0.0.0",
        "port": 8080,
        # LLM 配置
        "llm_provider": "openai",
        "llm_model": "gpt-3.5-turbo",
        "llm_temperature": 0.7,
        "llm_max_tokens": 65000,
        # OpenAI / OpenRouter
        "openai_api_key": None,
        "openai_api_base": None,
        "openai_organization": None,
        "openrouter_api_key": None,
        "openrouter_api_base": "https://openrouter.ai/api/v1",
        "huanxin_api_key": None,
        "huanxin_api_base": None,
        "dashscope_api_key": None,
        "dashscope_api_base": None,
        # CozeLoop 配置
        "cozeloop_workspace_id": None,
        "cozeloop_api_token": None,
        "cozeloop_prompt_label": "gray",
        # 图片生成 API
        "image_api_url": None,
        "image_api_key": None,
        "image_api_timeout": 300.0,
        "image_api_aspect_ratio": "3:4",
        "image_api_image_size": "1K",
        # MySQL 通用配置
        "mysql_pool_size": 20,
        "mysql_max_overflow": 20,
        # MySQL olap_zxy_agent
        "mysql_olap_zxy_agent_host": None,
        "mysql_olap_zxy_agent_port": None,
        "mysql_olap_zxy_agent_user": None,
        "mysql_olap_zxy_agent_password": None,
        "mysql_olap_zxy_agent_database": None,
        # MySQL B
        "mysql_b_host": None,
        "mysql_b_port": None,
        "mysql_b_user": None,
        "mysql_b_password": None,
        "mysql_b_database": None,
        # Redis 配置
        "redis_host": "localhost",
        "redis_port": 6379,
        "redis_password": None,
        "redis_db": 0,
        "redis_max_connections": 50,
        # Hologres 配置
        "hologres_host": None,
        "hologres_port": None,
        "hologres_user": None,
        "hologres_password": None,
        "hologres_database": None,
        # Es 配置
        "es_host": "zhixiaoyi-agent-server-yd5.private.cn-hangzhou.es-serverless.aliyuncs.com:9200",
        "es_user": "",
        "es_password": "",
        # 日志配置
        "log_level": "INFO",
        "log_format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        "log_file": "logs/app.log",
        "log_rotation": "500 MB",
        "log_retention": "10 days",
        # 重试配置
        "retry_max_attempts": 3,
        "retry_wait_exponential_multiplier": 1,
        "retry_wait_exponential_max": 10,
        # CORS 配置
        "cors_origins": ["*"],
        "cors_allow_credentials": True,
        "cors_allow_methods": ["*"],
        "cors_allow_headers": ["*"],
        # 线程池配置
        "thread_pool_max_workers": 40,
        # 外部依赖系统 API
        "abroad_api_url": None,
        "abroad_api_timeout": 20.0,
        "zxh_api_url": None,
        "zxh_api_timeout": 20.0,
        "zhikuan_api_url": None,
        "zhikuan_api_timeout": 20.0,
        "zhiyi_api_url": None,
        "zhiyi_api_timeout": 20.0,
        "fashion_parent_infra_api_url": "",
        "fashion_parent_infra_api_timeout": 20.0,
        # Dify API
        "dify_api_url": "https://dify-internal.zhiyitech.cn",
        "dify_api_key": None,
        "dify_property_workflow_id": None,
        "dify_zhiyi_property_workflow_id": "app-I64LIkTdjzKUriYkT5ZKUXkW",  # 知衣属性解析工作流
        "dify_zhiyi_shop_workflow_id": "app-UYP2IVg3pYJS0AczmDuTkUEp",  # 知衣店铺检索工作流（品牌->shopId）
        # 火山引擎知识库配置
        "volcengine_kb_api_url": "http://api-knowledgebase.mlp.cn-beijing.volces.com",
        "volcengine_kb_api_key": None,  # 从 Apollo 或环境变量获取，禁止硬编码
        "volcengine_kb_api_timeout": 60.0,
        # 主工作流配置
        "main_workflow_image_retry_attempts": 3,
        "main_workflow_image_max_workers": 4,
        "main_workflow_summary_max_length": 500,
        # OSS 配置
        "oss_upload_url": None,
        "oss_endpoint": "oss-cn-hangzhou-internal.aliyuncs.com",
        "oss_access_key_id": None,  # 从 Apollo 获取
        "oss_access_key_secret": None,  # 从 Apollo 获取
        "oss_bucket_name": "zhiyi-image",
        "oss_root_path": "zhixiaoyi_image_of_agent",
    }

    _instance: Optional["Settings"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._apollo_client: Optional["ApolloClient"] = None
        self._provider: Optional["ApolloConfigProvider"] = None
        self._init_apollo()

    def _init_apollo(self):
        """初始化 Apollo 客户端"""
        from app.core.clients.apollo_client import (
            ApolloClient,
            ApolloConfig,
            ApolloConfigProvider,
        )

        config = ApolloConfig(
            meta_server_url=APOLLO_META_SERVER_URL,
            app_id=APOLLO_APP_ID,
            cluster=APOLLO_CLUSTER,
            namespace=APOLLO_NAMESPACE,
        )
        self._apollo_client = ApolloClient(config)
        self._apollo_client.start()
        self._provider = ApolloConfigProvider(self._apollo_client)
        logger.info("[Config] Apollo 配置加载完成")

    def __getattr__(self, name: str) -> Any:
        """动态获取配置值"""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        default = self._DEFAULTS.get(name)

        # 根据默认值类型决定如何获取
        if default is None:
            return self._provider.get(name)
        elif isinstance(default, bool):
            return self._provider.get_bool(name, default)
        elif isinstance(default, int):
            return self._provider.get_int(name, default)
        elif isinstance(default, float):
            return self._provider.get_float(name, default)
        elif isinstance(default, list):
            return self._provider.get_list(name, default=default)
        else:
            return self._provider.get(name, default)

    # ==================== 计算属性 ====================

    @property
    def mysql_olap_zxy_agent_url(self) -> Optional[str]:
        host = self.mysql_olap_zxy_agent_host
        user = self.mysql_olap_zxy_agent_user
        database = self.mysql_olap_zxy_agent_database
        if host and user and database:
            port = self.mysql_olap_zxy_agent_port
            pwd = self.mysql_olap_zxy_agent_password
            return f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{database}?charset=utf8mb4"
        return None

    @property
    def mysql_b_url(self) -> Optional[str]:
        host = self.mysql_b_host
        user = self.mysql_b_user
        database = self.mysql_b_database
        if host and user and database:
            port = self.mysql_b_port or 3306
            pwd = self.mysql_b_password or ""
            return f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{database}?charset=utf8mb4"
        return None

    @property
    def hologres_url(self) -> Optional[str]:
        host = self.hologres_host
        user = self.hologres_user
        database = self.hologres_database
        if host and user and database:
            port = self.hologres_port or 5432
            pwd = self.hologres_password or ""
            return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{database}"
        return None

    # ==================== 公共方法 ====================

    def get_apollo_provider(self) -> Optional["ApolloConfigProvider"]:
        """获取 Apollo 配置提供者（用于注册热更新回调）"""
        return self._provider

    def stop(self):
        """停止 Apollo 客户端"""
        if self._apollo_client:
            self._apollo_client.stop()
            logger.info("[Config] Apollo 客户端已停止")


# 全局配置实例
settings = Settings()
