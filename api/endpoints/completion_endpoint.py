# -*- coding: utf-8 -*-
"""
输入补全 API 端点
"""
from fastapi import APIRouter
from loguru import logger

from app.schemas.request.completion_request import CompletionRequest
from app.schemas.response.common import CommonResponse
from app.service.completion_service import completion_service

router = APIRouter()


@router.post("/suggest")
async def suggest_completion(request: CompletionRequest):
    """
    获取输入补全建议
    """
    logger.info(f"[补全API] 收到请求: {request.model_dump_json()}")
    try:
        result = await completion_service.get_completion(request)
        logger.info(f"[补全API] 返回结果: result=[{result}], length={len(result)}")
        return CommonResponse[str](result=result)
    except Exception as e:
        logger.error(f"[补全API] 处理异常: {e}", exc_info=True)
        raise
