# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2026/1/15 15:05
# @File     : llm_output.py
"""
抖衣深度思考工作流 - LLM结构化输出类定义
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class DouyiExtractedProperty(BaseModel):
    """
    属性提取结果 - LLM结构化输出
    用于从用户问题中提取要分析的产品属性
    """
    front_property: list[str] = Field(
        default_factory=list,
        description="前端属性名列表，如 ['袖型', '面料']"
    )
    properties: list[str] = Field(
        default_factory=list,
        description="属性值列表，格式为 '属性名:属性值'，如 ['袖型:常规', '袖型:泡泡袖', '面料:聚酯纤维']"
    )


class DouyiPriceRangeItem(BaseModel):
    """单个价格范围"""
    min_price: int = Field(description="最低价格（单位：分）")
    max_price: int | None = Field(default=None, description="最高价格（单位：分），可为空表示无上限")


class DouyiExtractedPriceRange(BaseModel):
    """
    价格带提取结果 - LLM结构化输出
    用于从用户问题中提取要分析的价格范围
    """
    price_ranges: list[DouyiPriceRangeItem] = Field(
        default_factory=list,
        description="价格范围列表"
    )

    def to_api_format(self) -> list[dict[str, int]]:
        """转换为API请求格式"""
        return [
            {"minPrice": item.min_price, "maxPrice": item.max_price or 99999999}
            for item in self.price_ranges
        ]


__all__ = [
    "DouyiExtractedProperty",
    "DouyiPriceRangeItem",
    "DouyiExtractedPriceRange",
]
