# -*- coding: utf-8 -*-
"""
输入补全请求 Schema
"""
from typing import Optional

from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    """补全请求"""
    input: str = Field(description="用户当前输入")
    user_id: Optional[str] = Field(default=None, description="用户ID，用于个性化记忆")
    preferred_entity: Optional[str] = Field(default=None, description="用户偏好实体/模式")
    industry: Optional[str] = Field(default=None, description="用户选择行业")
    abroad_type: Optional[str] = Field(default=None, description="海外站点类型")
