# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2025/11/28 11:26
# @File     : workflow_track.py
from pydantic import BaseModel


class UserQueryLlmParameter(BaseModel):
    """
    工作流llm参数解析输出埋点
    """
    query: str
    session_id: str
    message_id: str
    user_id: str
    team_id: str
    llm_parameter: str