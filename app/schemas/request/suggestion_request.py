# -*- coding: utf-8 -*-
"""
建议生成请求
"""
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict
from pydantic.alias_generators import to_camel


class SuggestionGenerateRequest(BaseModel):
    """建议生成请求"""
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="ignore",  # 忽略额外字段（兼容 n8n 发送的额外字段）
    )

    user_query: str = Field(description="用户原始输入")
    preferred_entity: Optional[str] = Field(default=None, description="平台偏好：知衣/抖衣/海外探款")
    industry: Optional[str] = Field(default=None, description="行业")
