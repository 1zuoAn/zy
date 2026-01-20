# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2025/11/26 16:31
# @File     : context.py
from dataclasses import dataclass
from typing import List, Optional
from pydantic import BaseModel


@dataclass()
class SelectWorkflowContext:
    def __init__(self):
        self.team_id: int = None
        self.user_id: int = None
        self.session_id: str = None
        self.message_id: str = None
        self.user_query: str = None
        self.industry: str = None
        self.preferred_entity: str = None
        self.user_preferences: str = None


@dataclass
class ConversationMessage:
    """会话消息"""
    id: int
    session_id: str
    user_query: str


class SubWorkflowResult(BaseModel):
    """子工作流执行结果"""
    workflow_name: str
    output: str
    relate_data: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None


def format_conversation_history(
    messages: List[ConversationMessage], 
    include_ai_messages: bool = False
) -> str:
    """
    格式化会话历史
    
    Args:
        messages: 会话消息列表
        include_ai_messages: 是否包含AI消息
        
    Returns:
        str: 格式化后的会话历史
    """
    if not messages:
        return ""
    
    formatted_messages = []
    for msg in messages:
        if include_ai_messages:
            formatted_messages.append(msg.user_query)
        else:
            # 只包含人类消息
            if msg.user_query.startswith("human:"):
                formatted_messages.append(msg.user_query[6:])  # 去掉 "human:" 前缀
    
    if not formatted_messages:
        return ""
    
    if len(formatted_messages) == 1:
        return f"最新消息：{formatted_messages[0]}"
    else:
        history = "，".join(formatted_messages[:-1])
        latest = formatted_messages[-1]
        return f"历史消息：{history}，最新消息：{latest}"
