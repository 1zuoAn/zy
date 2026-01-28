"""
健康检查 API
"""
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import settings
from app.schemas.response.common import CommonResponse

router = APIRouter()


class HealthResponse(BaseModel):
    """健康检查响应模型"""

    status: str
    app_name: str
    version: str
    environment: str
    timestamp: datetime


@router.get("", response_model=CommonResponse[HealthResponse])
def health_check() -> CommonResponse[HealthResponse]:
    """
    健康检查端点

    返回应用的健康状态和基本信息
    """
    return CommonResponse[HealthResponse](
        result=HealthResponse(
            status="healthy",
            app_name=settings.app_name,
            version=settings.app_version,
            environment=settings.environment,
            timestamp=datetime.now(),
        )
    )


@router.get("/ping")
def ping() -> dict:
    """
    简单的 ping 端点

    用于快速检查服务是否可用
    """
    return {"message": "pong"}
