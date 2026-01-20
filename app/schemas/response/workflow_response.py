# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2025/11/27 17:02
# @File     : workflow_response.py
from typing import Optional

from pydantic import BaseModel, Field


class WorkflowResponse(BaseModel):
    select_result: Optional[str] = Field(description="选品结果")
    relate_data: Optional[str] = Field(description="相关数据")
