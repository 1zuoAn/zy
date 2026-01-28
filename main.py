"""
FastAPI 应用入口
"""
import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.api.router import api_router
from app.config import settings
from app.core.errors import AppException, ErrorCode, error_payload
from app.utils.thread_pool import shutdown_thread_pool
from app.core.clients.db_client import init_database_clients_from_settings, close_all_named_db_clients
from app.core.clients.redis_client import redis_client
from app.core.clients.coze_loop_client import coze_loop_client_provider
from app.core.clients.config_client import init_config_client_from_settings
from app.service.intent import intent_service

def setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format=settings.log_format,
        level=settings.log_level,
        colorize=True,
    )
    if settings.log_file:
        log_path = Path(settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            settings.log_file,
            format=settings.log_format,
            level=settings.log_level,
            rotation=settings.log_rotation,
            retention=settings.log_retention,
            compression="zip",
            encoding="utf-8",
        )
    logger.info(f"日志系统初始化完成 - Level: {settings.log_level}")



async def on_startup() -> None:
    logger.info(f"{settings.app_name} v{settings.app_version} 正在启动...")
    logger.info(f"环境: {settings.environment}")
    logger.info(f"Debug 模式: {settings.debug}")

    # 初始化客户端
    init_config_client_from_settings()
    init_database_clients_from_settings()
    redis_client.init_sync_client()
    coze_loop_client_provider.get_client()
    
    await intent_service.startup()


async def on_shutdown() -> None:
    logger.info(f"{settings.app_name} 正在关闭...")
    
    await intent_service.shutdown()
    
    close_all_named_db_clients()
    redis_client.close()
    coze_loop_client_provider.close()
    shutdown_thread_pool()

    settings.stop()


def create_application() -> FastAPI:
    """创建 FastAPI 应用实例"""
    # 配置日志
    setup_logging()

    # 创建应用
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="基于 FastAPI 和 LangChain 的 AI 工作流引擎",
        docs_url=f"{settings.api_prefix}/docs" if settings.debug else None,
        redoc_url=f"{settings.api_prefix}/redoc" if settings.debug else None,
        openapi_url=f"{settings.api_prefix}/openapi.json" if settings.debug else None,
    )

    # 配置 CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )

    app.include_router(api_router, prefix=settings.api_prefix)

    app.add_event_handler("startup", on_startup)
    app.add_event_handler("shutdown", on_shutdown)

    # 根路径
    @app.get("/")
    def root():
        return {
            "app": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "docs": f"{settings.api_prefix}/docs" if settings.debug else "disabled",
        }

    # 健康检查
    @app.get("/health")
    def health_check():
        return {
            "status": "healthy",
            "app": settings.app_name,
            "version": settings.app_version,
        }

    @app.exception_handler(AppException)
    def app_exception_handler(request: Request, exc: AppException):
        # 记录业务异常
        logger.error(f"业务异常 [code={exc.code}] [trace_id={exc.trace_id}]: {exc.message}")
        payload = error_payload(exc.code, exc.message, trace_id=exc.trace_id, details=exc.details)
        return JSONResponse(payload, status_code=exc.http_status)

    @app.exception_handler(Exception)
    def generic_exception_handler(request: Request, exc: Exception):
        # 记录未预期的异常，包含完整堆栈
        logger.exception(f"未捕获的异常: {str(exc)}")
        payload = error_payload(ErrorCode.INTERNAL_ERROR, str(exc))
        return JSONResponse(payload, status_code=500)

    logger.info("FastAPI 应用创建完成")
    return app


# 创建应用实例
app = create_application()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
