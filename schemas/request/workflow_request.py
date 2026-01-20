# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2025/11/26 20:09
# @File     : workflow_request.py
from __future__ import annotations

from typing import List, Optional, Union
from enum import Enum
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class WorkflowRequest(BaseModel):
    """
    选品工作流通用请求
    """
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _flatten_n8n_payload(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data

        payload = dict(data)
        body = payload.get("body")
        container = body if isinstance(body, dict) else payload

        def set_if_missing(key: str, value: object) -> None:
            if key not in payload and value is not None:
                payload[key] = value

        set_if_missing("team_id", container.get("team_id") or container.get("teamId"))
        set_if_missing("user_id", container.get("user_id") or container.get("userId"))
        set_if_missing("session_id", container.get("session_id") or container.get("sessionId"))
        set_if_missing("message_id", container.get("message_id") or container.get("messageId"))

        content = container.get("content")
        if isinstance(content, dict):
            set_if_missing("user_query", content.get("text"))
            if "images" not in payload and content.get("images") is not None:
                payload["images"] = content.get("images")

        ext = container.get("extension_info") or container.get("extensionInfo")
        if isinstance(ext, dict):
            set_if_missing("preferred_entity", ext.get("preferredEntity") or ext.get("preferred_entity"))
            set_if_missing("industry", ext.get("industry"))
            set_if_missing("abroad_type", ext.get("abroadType") or ext.get("abroad_type"))
            set_if_missing("is_monitored", ext.get("isMonitored") or ext.get("is_monitored"))
            set_if_missing("is_user_preferences", ext.get("isUserPreferences") or ext.get("is_user_preferences"))
            set_if_missing("perm_entity_type_list", ext.get("permEntityTypeList") or ext.get("perm_entity_type_list"))
            set_if_missing("user_preferences", ext.get("userPreferences") or ext.get("user_preferences"))

        set_if_missing(
            "query_references",
            container.get("query_references") or container.get("queryReferences"),
        )

        return payload
    team_id: int = Field(
        description='团队id',
        validation_alias=AliasChoices("team_id", "teamId"),
    )
    user_id: int = Field(
        description='用户id',
        validation_alias=AliasChoices("user_id", "userId"),
    )
    session_id: str = Field(
        description='会话id',
        validation_alias=AliasChoices("session_id", "sessionId"),
    )
    message_id: str = Field(
        description='消息id',
        validation_alias=AliasChoices("message_id", "messageId"),
    )
    user_query: str = Field(
        description='用户问题',
        validation_alias=AliasChoices("user_query", "userQuery", "query"),
    )
    industry: str = Field(
        description='用户选择行业',
        validation_alias=AliasChoices("industry", "industry"),
    )
    preferred_entity: str = Field(
        description='用户偏好实体',
        validation_alias=AliasChoices("preferred_entity", "preferredEntity"),
    )
    user_preferences: str = Field(
        default="",
        description='用户选款偏好',
        validation_alias=AliasChoices("user_preferences", "userPreferences"),
    )
    is_monitored: Optional[bool] = Field(
        default=None,
        description="是否参考监控数据",
        validation_alias=AliasChoices("is_monitored", "isMonitored"),
    )
    is_user_preferences: Optional[bool] = Field(
        default=None,
        description="是否参考用户画像",
        validation_alias=AliasChoices("is_user_preferences", "isUserPreferences"),
    )
    perm_entity_type_list: Optional[Union[List[int], str]] = Field(
        default=None,
        description="权限列表",
        validation_alias=AliasChoices("perm_entity_type_list", "permEntityTypeList"),
    )
    query_references: Optional[list[WorkflowQueryReferenceItem]] = Field(
        default=None,
        description='用户问题中的关联实体',
        validation_alias=AliasChoices("query_references", "queryReferences"),
    )
    abroad_type: Optional[str] = Field(
        default=None,
        description='用户选择的海外探款数据类型',
        validation_alias=AliasChoices("abroad_type", "abroadType"),
    )
    suppress_messages: bool = Field(
        default=False,
        description="是否抑制消息推送(内部控制)",
    )
    image_content: Optional[str] = Field(
        default=None,
        description="图片内容描述(主流程提取)",
    )

    # 图片设计工作流字段
    image_prompt: Optional[str] = Field(default=None, alias="imagePrompt", description='文生图提示词')
    input_images: Optional[Union[str, List[str]]] = Field(
        default=None,
        alias="image",
        description='图生图输入图片(URL列表或#分隔字符串)',
    )
    edit_prompt: Optional[str] = Field(default=None, alias="prompt", description='图生图编辑提示词')

    # 数据洞察字段
    thinking: Optional[bool] = Field(default=None, description="是否使用深度思考（数据洞察使用）")


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
