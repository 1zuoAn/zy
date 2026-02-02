# -*- coding: utf-8 -*-
"""
建议生成 API 端点
"""
from fastapi import APIRouter

from app.schemas.request.suggestion_request import SuggestionGenerateRequest
from app.schemas.response.common import CommonResponse
from app.schemas.response.suggestion_response import SuggestionGenerateResponse
from app.service.suggestion_service import suggestion_service

router = APIRouter()


@router.post("/generate")
def generate_suggestions(request: SuggestionGenerateRequest):
    """
    生成智能追问建议（成功/失败两套）
    """
    result = suggestion_service.generate(request)
    return CommonResponse[SuggestionGenerateResponse](result=result)
