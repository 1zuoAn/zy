# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2025/11/26 21:24
# @File     : redis_message.py
from __future__ import annotations
from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field


class TextMessageContent(BaseModel):
    text: Optional[str] = Field(default=None, description="文本内容")


class WithActionContent(TextMessageContent):
    actions: Optional[list[str]] = Field(default_factory=list)
    agent: Optional[str] = Field(default=None, description="执行的agent类型")
    data: Optional[ParameterData] = Field(default=None)
    # 在退还扣点时使用
    cost_id: Optional[str] = Field(default=None, description="扣费id，为None则使用message_id进行退费")


class CustomDataContent(BaseModel):
    data: dict = Field(default_factory=dict)


class UserTagFilterItem(BaseModel):
    """输出的用户画像筛选项结构"""
    name: Optional[str] = Field(default=None, description="筛选项名称")
    filter_type: Optional[str] = Field(default=None, description="筛选项类型")
    value: Optional[str] = Field(default=None, description="筛选项值")

class ParameterData(BaseModel):
    """参数输出的内部数据结构"""
    request_path: Optional[str] = Field(default=None, description="请求路径（知衣不需要此字段）")
    request_body: Optional[str] = Field(default=None, description="请求体 JSON 字符串")
    actions: list[str] = Field(default_factory=list, description="可用操作")
    title: Optional[str] = Field(default="", description="标题")
    entity_type: Optional[int] = Field(default=None, description="实体类型")
    filters: Optional[list[UserTagFilterItem]] = Field(default_factory=list, description="筛选条件")

    task_status: Optional[int] = Field(default=None, description="查询任务是否成功：1-成功/0-失败")


class ParameterDataContent(BaseModel):
    """content_type=5 的内容结构"""
    data: ParameterData


class TaskProgressItem(BaseModel):
    """任务进度项"""
    step_name: str = Field(description="步骤名称")
    status: str = Field(default="pending", description="状态: pending/running/completed/failed")
    detail: Optional[str] = Field(default=None, description="详情文本，支持动态数值")


class TaskProgressContent(BaseModel):
    """content_type=9 任务进度内容"""
    current_step: int = Field(description="当前步骤索引(0-based)")
    total_steps: int = Field(description="总步骤数")
    steps: List[TaskProgressItem] = Field(default_factory=list, description="步骤列表")


class SuggestionItem(BaseModel):
    """建议项"""
    label: str = Field(description="建议标签，如'限制价位200-300'")
    query: str = Field(description="点击后自动组装的完整 query")


class SuggestionContent(BaseModel):
    """content_type=10 对话建议内容"""
    suggestions: List[SuggestionItem] = Field(default_factory=list, description="建议列表，1-3个")
    context_type: str = Field(default="selection", description="上下文类型: selection/chat")


class PhaseProgressContent(BaseModel):
    """content_type=1 阶段化进度内容（仅当前阶段）"""
    task_name: str = Field(description="任务名称，如'选品任务'")
    phase_name: str = Field(description="阶段名称，如'选品任务规划'")
    status: str = Field(description="状态: completed/failed")
    text: str = Field(description="阶段内容，多行用换行符分隔")


class BaseRedisMessage(BaseModel):
    env: str = 'gray'
    session_id: str = Field(description="传入的会话id")
    reply_message_id: str = Field(description="传入的消息id")
    reply_id: str = Field(description="format: reply_{message_id}")
    operate_id: str
    reply_seq: int
    status: str
    content_type: int
    content: Union[None, TextMessageContent, WithActionContent, CustomDataContent, ParameterDataContent, TaskProgressContent, SuggestionContent, PhaseProgressContent]
    create_ts: int = Field(description="13位时间戳")
