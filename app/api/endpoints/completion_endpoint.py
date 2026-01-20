# -*- coding: utf-8 -*-
"""
输入补全 API 端点
"""
from fastapi import APIRouter

from app.schemas.request.completion_request import CompletionRequest
from app.schemas.response.common import CommonResponse
from app.service.completion_service import completion_service

router = APIRouter()


@router.post("/suggest")
async def suggest_completion(request: CompletionRequest):
    """
    获取输入补全建议
    """
    result = await completion_service.get_completion(request)
    return CommonResponse[str](result=result)
