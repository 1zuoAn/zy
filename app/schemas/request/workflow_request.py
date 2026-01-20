# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2025/11/26 20:09
# @File     : workflow_request.py
from __future__ import annotations

from typing import List, Optional, Union
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field


class WorkflowRequest(BaseModel):
    """
    选品工作流通用请求
    """
    model_config = ConfigDict(populate_by_name=True)
    team_id: int = Field(description='团队id')
    user_id: int = Field(description='用户id')
    session_id: str = Field(description='会话id')
    message_id: str = Field(description='消息id')
    user_query: str = Field(description='用户问题')
    industry: str = Field(description='用户选择行业')
    preferred_entity: str = Field(description='用户偏好实体')
    user_preferences: str = Field(description='用户选款偏好')
    query_references: Optional[list[WorkflowQueryReferenceItem]] = Field(default=None, description='用户问题中的关联实体')
    abroad_type: Optional[str] = Field(default=None, description='用户选择的海外探款数据类型')

    # 图片设计工作流字段
    image_prompt: Optional[str] = Field(default=None, alias="imagePrompt", description='文生图提示词')
    input_images: Optional[Union[str, List[str]]] = Field(
        default=None,
        alias="image",
        description='图生图输入图片(URL列表或#分隔字符串)',
    )
    edit_prompt: Optional[str] = Field(default=None, alias="prompt", description='图生图编辑提示词')


class MainWorkflowRequest(WorkflowRequest):
    """
    主工作流请求，支持图片输入
    """
    images: Optional[List[str]] = Field(default=None, description='图片URL列表')


class WorkflowQueryReferenceItem(BaseModel):
    """
    选品工作流的关联实体
    """
    class Config:
        populate_by_name = True

    reference_name: str = Field(default=None, description='文本嵌入名称', alias='referenceName')
    entity_type: str = Field(default=None, description='实体类型', alias='entityType')
    entity_id: Optional[str] = Field(default=None, description='实体id', alias='entityId')
    display_name: Optional[str] = Field(default=None, description='展示名称', alias='displayName')
    platform_type: Optional[Union[int, str]] = Field(
        default=None, description='平台类型ID', alias='platformType'
    )
    platform_name: Optional[str] = Field(
        default=None, description='平台名称', alias='platformName'
    )

class QueryReferenceEntityType(str, Enum):
    """查询引用实体类型"""
    # 店铺
    TAOBAO_SHOP = "TAOBAO_SHOP"
    DOUYIN_SHOP = "DOUYIN_SHOP"
    # 商品
    TAOBAO_GOODS = "TAOBAO_GOODS"
    DOUYIN_GOODS = "DOUYIN_GOODS"
    # 海外品牌和站点
    ABROAD_BRAND = "ABROAD_BRAND"
    ABROAD_SITE = "ABROAD_SITE"
    # 海外探款店铺
    ABROAD_GOODS_SHOP = "ABROAD_GOODS_SHOP"

