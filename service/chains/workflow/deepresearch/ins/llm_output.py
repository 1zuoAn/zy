# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2026/1/22 21:16
# @File     : llm_output.py
"""
INS数据洞察工作流 - LLM结构化输出类定义
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.service.chains.workflow.deepresearch.deepresearch_graph_state import BaseDeepresearchGraphState


class InsApiParseParam(BaseModel):
    """
    API参数解析结果 - LLM结构化输出
    用于从用户问题中解析日期参数
    """
    start_date: str = Field(description="开始日期，格式 yyyy-MM-dd")
    end_date: str = Field(description="结束日期，格式 yyyy-MM-dd")
    date_type: str = Field(description="日期类型：weekly（时间跨度小于两个月）/ monthly（时间跨度大于两个月）")


class InsDeepresearchState(BaseDeepresearchGraphState):
    """
    INS数据洞察deepresearch
    """
    is_thinking: bool = Field(default=False, description="是否为深度思考模式")

    api_parse_param: InsApiParseParam | None = Field(default=None, description="API参数解析结果")

    report_text: str | None = Field(default=None, description="生成的报告文本")


__all__ = [
    "InsApiParseParam",
]
