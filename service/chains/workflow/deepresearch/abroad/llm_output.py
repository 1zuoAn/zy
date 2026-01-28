# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2026/1/15 22:48
# @File     : llm_output.py
from __future__ import annotations

from pydantic import BaseModel, Field


class AbroadExtractedProperty(BaseModel):
    """
    属性提取结果
    用于LLM从用户问题中提取需要查询的属性名称
    """
    property_name: str = Field(
        description="属性名称，如：袖长、裙长、面料、适用场景等。从已有属性表中匹配到的属性名"
    )


class AbroadDimensionAnalysis(BaseModel):
    """
    维度分析结果
    用于LLM判断用户查询的维度类型
    """
    dimension_type: int = Field(
        default=0,
        description="维度类型：0-其他/默认属性，1-颜色，2-面料，3-价格带"
    )
