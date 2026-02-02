# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2026/1/14 17:23
# @File     : schema.py
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict

from app.service.rpc.zhiyi.schemas import SaleTrendCleanResponse, CategoryTrendCleanResponse, PriceRangeCleanResponse, \
    ColorCleanResponse, BrandCleanResponse, Top10ItemsCleanResponse, ShopHotItemRawResponse


# 存储一些工作流产出的中间结果类
class ZhiyiThinkingApiParseParam(BaseModel):
    """
    API参数解析结果模型
    用于LLM结构化输出，解析用户问题中的查询参数
    """
    root_category_id: int | None = Field(default=None, description="根类目ID")
    root_category_id_name: str | None = Field(default=None, description="根类目名称")
    category_id: int | None = Field(
        default=None,
        description="类目ID，品类维表中没有对应时为null，不要把rootCategoryId当作categoryId填入，也不要给出相似的categoryId"
    )
    category_id_name: str | None = Field(default=None, description="类目名称，没有对应categoryId时为null")
    start_date: str | None = Field(default=None, description="开始日期，ISO 8601格式，例如 '2025-11-01'")
    end_date: str | None = Field(default=None, description="结束日期，ISO 8601格式，例如 '2025-11-30'")
    date_type: str | None = Field(default=None, description="日期类型，例如 'week' 或 'month'，不明确时返回null")
    table_type: Literal["1", "2", "3"] = Field(default="1",
        description="图表类型：'1'-品类分析（默认）、'2'-属性分析（长袖/长裙/季节/场景/尺码/面料/品牌/颜色等）、'3'-价格带分析（价格带相关/价格区间等）。注意：若categoryId为null，则不可为'2'"
    )
    is_shop: bool = Field(default=False, description="是否查询店铺数据，True表示查询店铺，False表示查询大盘")
    shop_name: str | None = Field(default=None, description="店铺名称，当is_shop为True时必填")



class ZhiyiCategoryFormatItem(BaseModel):
    root_category_id: str = Field(default=None)
    root_category_id_name: str = Field(default=None)
    category_id: str = Field(default=None)
    category_name: str = Field(default=None)

class ZhiyiParsedCategory(BaseModel):
    category_list: list[ZhiyiCategoryFormatItem] = Field(default_factory=list, description="处理后的类目列表")


# ========== 聚合数据模型（用于报告生成和 Excel 导出） ==========

class ZhiyiAggregatedData(BaseModel):
    """知衣数据洞察聚合分析数据 - 用于报告生成和 Excel 导出"""
    model_config = ConfigDict(populate_by_name=True)

    # ========== 分析类型标识 ==========
    is_shop: bool = Field(description="是否为店铺分析")
    table_type: Literal["1", "2", "3"] = Field(
        description="图表类型：1-品类分析, 2-属性分析, 3-价格带分析"
    )
    hydc_dimension_type: int | None = Field(
        default=None,
        description="大盘维度类型：0-属性, 1-颜色, 2-品牌 (仅table_type=2时有效)"
    )

    # ========== 基础参数信息 ==========
    shop_name: str | None = Field(default=None, description="店铺名称 (仅店铺分析)")
    category_path: str | None = Field(default=None, description="品类路径，如 '女装-连衣裙'")
    start_date: str | None = Field(default=None, description="开始日期")
    end_date: str | None = Field(default=None, description="结束日期")
    property_name: str | None = Field(default=None, description="属性名称 (仅属性分析时有效)")

    # ========== 概览趋势数据 ==========
    overview_volume: SaleTrendCleanResponse | None = Field(
        default=None, description="销量趋势数据"
    )
    overview_amount: SaleTrendCleanResponse | None = Field(
        default=None, description="销售额趋势数据"
    )

    # ========== 品类分析数据 (table_type=1) ==========
    category_data: CategoryTrendCleanResponse | None = Field(
        default=None, description="品类分布数据"
    )

    # ========== 价格带数据 ==========
    price_data: PriceRangeCleanResponse | None = Field(
        default=None, description="价格带分布数据"
    )

    # ========== 颜色数据 ==========
    color_data: ColorCleanResponse | None = Field(
        default=None, description="颜色分布数据"
    )

    # ========== 属性数据 (table_type=2, dimension=0) ==========
    property_data: ColorCleanResponse | None = Field(
        default=None, description="属性分布数据"
    )

    # ========== 品牌数据 (大盘品牌分析, dimension=2) ==========
    brand_data: BrandCleanResponse | None = Field(
        default=None, description="品牌分布数据"
    )

    # ========== Top商品数据 ==========
    top_goods: Top10ItemsCleanResponse | None = Field(
        default=None, description="Top10商品数据"
    )
    top_goods_raw: ShopHotItemRawResponse | None = Field(
        default=None, description="Top商品原始数据"
    )

