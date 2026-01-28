# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2026/1/14 10:25
# @File     : llm_output.py
from pydantic import BaseModel, Field


class ShopCleanResult(BaseModel):
    content_list: list[str] = Field(default_factory=list, description="清洗后保留的店铺列表")

class PropertyExtractResult(BaseModel):
    """属性提取结果 - 用于从用户问题中提取属性名"""
    property_name: str = Field(description="提取的属性名称，如'裙长'、'袖长'等")

class PriceBandExtractResult(BaseModel):
    """价格带提取结果"""
    price_band_list: list[str] = Field(default_factory=list, description="从用户问题中提取的价格带")


# ========== 大盘分析 LLM 输出模型 ==========

class HydcDimensionAnalyzeResult(BaseModel):
    """大盘维度分析结果 - 用于判断用户查询的维度类型"""
    dimension_type: int = Field(
        description="维度类型：0=其他属性(如袖型、裙长等), 1=颜色, 2=品牌",
        ge=0, le=2
    )
    reason: str = Field(description="判断理由")


class HydcPropertyExtractResult(BaseModel):
    """大盘属性提取结果 - 用于从用户问题中提取具体属性名"""
    property_name: str = Field(description="提取的属性名称，如'袖型'、'裙长'、'面料'等")