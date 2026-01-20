# -*- coding: utf-8 -*-
"""
建议生成响应
"""
from typing import List

from pydantic import BaseModel, Field, ConfigDict


class SuggestionItem(BaseModel):
    """单条建议"""
    model_config = ConfigDict(
        populate_by_name=True,
    )

    text: str = Field(..., description="小方框展示文案")
    filled_query: str = Field(..., description="点击后填入输入框的完整请求句")


class SuggestionGenerateResponse(BaseModel):
    """建议生成响应"""
    model_config = ConfigDict(
        populate_by_name=True,
    )

    success_suggestions: List[SuggestionItem] = Field(default_factory=list, description="成功场景建议")
    failed_suggestions: List[SuggestionItem] = Field(default_factory=list, description="失败场景建议")
