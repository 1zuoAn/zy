# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2026/1/15 22:48
# @File     : schema.py
from __future__ import annotations

from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict

from app.schemas.response.common import PageResult
from app.service.rpc.abroad.schemas import AbroadGoodsEntity


# ========== API参数解析结果 ==========

class AbroadMainParseParam(BaseModel):
    """
    海外探款API参数解析结果模型
    用于LLM结构化输出，解析用户问题中的查询参数
    """
    root_category_id: str | None = Field(default=None, description="根类目ID")
    root_category_id_name: str | None = Field(default=None, description="根类目名称")
    category_id: str | None = Field(
        default=None,
        description="类目ID，品类维表中没有对应时为null，不要给出相似的categoryId"
    )
    category_id_name: str | None = Field(default=None, description="类目名称")
    start_date: str | None = Field(default=None, description="开始日期，ISO 8601格式，例如 '2025-11-01'")
    end_date: str | None = Field(default=None, description="结束日期，ISO 8601格式，例如 '2025-11-30'")
    date_type: str | None = Field(default=None, description="日期类型：WEEK / MONTH")
    trend_date_type: str | None = Field(default=None, description="日期类型 DAY / WEEK / MONTH")
    platform: str | None = Field(default=None, description="站点平台：SHEIN/Temu/独立站/亚马逊。海外探款这个词除外")
    extract_platform_name: str | None = Field(default=None, description="从用户问题中分析出的站点名称")
    table_type: int = Field(default=1, description="图表类型：1-品类分析（默认，查询维度多）、2-非品类分析（属性/颜色/面料/价格带等，查询维度少）")


class AbroadPlatformType(BaseModel):
    """站点判断结果"""
    target_platform_type_list: list[int] = Field(default_factory=list, description="站点类型ID，如4表示亚马逊美国站")


# ========== 维度分析类型（用于分支2动态查询） ==========

class AbroadDimensionType(IntEnum):
    """维度分析类型枚举"""
    OTHER = 0       # 其他/默认属性
    COLOR = 1       # 颜色
    FABRIC = 2      # 面料
    PRICE = 3       # 价格带


# ========== 趋势数据清洗模型（trend-summary接口） ==========

class AbroadTrendSlimItem(BaseModel):
    """海外探款趋势数据精简项"""
    model_config = ConfigDict(populate_by_name=True)

    record_date: str | None = Field(default=None, alias="recordDate", description="日期/周次范围")
    sale_volume: int = Field(default=0, alias="saleVolume", description="销量")
    sale_amount: int = Field(default=0, alias="saleAmount", description="销售额")


class AbroadTrendSlimResult(BaseModel):
    """海外探款趋势数据精简结果"""
    model_config = ConfigDict(populate_by_name=True)

    sale_volume_trend: list[AbroadTrendSlimItem] = Field(
        default_factory=list, alias="saleVolumeTrend", description="销量趋势"
    )
    sale_amount_trend: list[AbroadTrendSlimItem] = Field(
        default_factory=list, alias="saleAmountTrend", description="销售额趋势"
    )
    currency: str = Field(default="USD", description="货币单位")


class AbroadTrendCleanResponse(BaseModel):
    """海外探款趋势数据清洗后响应"""
    success: bool = True
    result: AbroadTrendSlimResult | None = None


# ========== 维度分析数据清洗模型（info接口，品类/颜色/价格带通用） ==========

class AbroadDimensionInfoItem(BaseModel):
    """海外探款维度分析数据精简项（品类/颜色/价格带等通用）"""
    model_config = ConfigDict(populate_by_name=True)

    dimension_info: str | None = Field(default=None, alias="dimensionInfo", description="维度值，如品类名/颜色/价格区间")
    sale_volume: int | None = Field(default=0, alias="saleVolume", description="销量")
    sale_amount: int | None = Field(default=0, alias="saleAmount", description="销售额")
    sale_volume_ratio: float | None = Field(default=0.0, alias="saleVolumeRatio", description="销量占比")
    sale_amount_ratio: float | None = Field(default=0.0, alias="saleAmountRatio", description="销售额占比")
    sale_volume_mom_ratio: float | None = Field(default=None, alias="saleVolumeMomRatio", description="销量环比")
    sale_amount_mom_ratio: float | None = Field(default=None, alias="saleAmountMomRatio", description="销售额环比")
    sale_volume_yoy_ratio: float | None = Field(default=None, alias="saleVolumeYoyRatio", description="销量同比")
    sale_amount_yoy_ratio: float | None = Field(default=None, alias="saleAmountYoyRatio", description="销售额同比")


class AbroadDimensionInfoResult(BaseModel):
    """海外探款维度分析数据精简结果"""
    model_config = ConfigDict(populate_by_name=True)

    result_count: int = Field(default=0, alias="resultCount", description="结果数量")
    items: list[AbroadDimensionInfoItem] = Field(default_factory=list, description="维度数据列表")


class AbroadDimensionInfoCleanResponse(BaseModel):
    """海外探款维度分析数据清洗后响应"""
    success: bool = True
    result: AbroadDimensionInfoResult | None = None


# ========== 颜色数据清洗模型（特殊结构，可能包含trends） ==========

class AbroadColorSlimItem(BaseModel):
    """海外探款颜色数据精简项"""
    model_config = ConfigDict(populate_by_name=True)

    property_value: str | None = Field(default=None, alias="propertyValue", description="颜色值")
    sales_volume: int | None = Field(default=0, alias="salesVolume", description="销量")
    sales_amount: int | None = Field(default=0, alias="salesAmount", description="销售额")
    mom: float | None = Field(default=0.0, alias="salesVolumeMomRatio", description="销量环比")
    rate: float | None = Field(default=0.0, description="占比")
    other_flag: bool | None = Field(default=False, alias="otherFlag", description="是否为'其他'合并项")


class AbroadColorCleanResponse(BaseModel):
    """海外探款颜色数据清洗后响应"""
    model_config = ConfigDict(populate_by_name=True)

    success: bool = True
    result: list[AbroadColorSlimItem] = Field(default_factory=list, alias="color_slim")


# ========== 价格带数据清洗模型 ==========

class AbroadPriceSlimItem(BaseModel):
    """海外探款价格带精简项"""
    model_config = ConfigDict(populate_by_name=True)

    left_price: int | None = Field(default=0, alias="leftPrice", description="价格下限（分）")
    right_price: int | None = Field(default=None, alias="rightPrice", description="价格上限（分）")
    sales_volume: int | None = Field(default=0, alias="salesVolume", description="销量")
    rate: float | None = Field(default=0.0, description="占比")


class AbroadPriceCleanResponse(BaseModel):
    """海外探款价格带清洗后响应"""
    model_config = ConfigDict(populate_by_name=True)

    success: bool = True
    result: list[AbroadPriceSlimItem] = Field(default_factory=list, alias="price_slim")


# ========== 属性趋势数据清洗模型（v2/trend接口） ==========

class AbroadPropertyTrendItem(BaseModel):
    """海外探款属性趋势数据项"""
    model_config = ConfigDict(populate_by_name=True)

    date_range: str | None = Field(default=None, alias="dateRange", description="日期范围")
    sales_volume: int = Field(default=0, alias="salesVolume", description="销量")


class AbroadPropertySlimItem(BaseModel):
    """海外探款属性数据精简项"""
    model_config = ConfigDict(populate_by_name=True)

    property_value: str | None = Field(default=None, alias="propertyValue", description="属性值")
    sales_volume: int = Field(default=0, alias="salesVolume", description="销量")
    sales_amount: int = Field(default=0, alias="salesAmount", description="销售额")
    rate: float = Field(default=0.0, description="占比")
    trends: list[AbroadPropertyTrendItem] = Field(default_factory=list, description="趋势数据")


class AbroadPropertyCleanResponse(BaseModel):
    """海外探款属性数据清洗后响应"""
    model_config = ConfigDict(populate_by_name=True)

    success: bool = True
    result: list[AbroadPropertySlimItem] = Field(default_factory=list, alias="color_slim")


# ========== Top商品数据清洗模型（goods-zone-list接口） ==========

class AbroadTopGoodsSlimItem(BaseModel):
    """海外探款Top商品精简项"""
    model_config = ConfigDict(populate_by_name=True)

    item_id: str | None = Field(default=None, alias="itemId", description="商品ID")
    title: str | None = Field(default=None, description="商品标题")
    pic_url: str | None = Field(default=None, alias="picUrl", description="商品图片URL")
    category_name: str | None = Field(default=None, alias="categoryName", description="品类名称")
    category_detail: str | None = Field(default=None, alias="categoryDetail", description="品类路径")
    sales_volume: int = Field(default=0, alias="salesVolume", description="销量（30天）")
    sales_amount: int = Field(default=0, alias="salesAmount", description="销售额（30天）")
    min_price: str | None = Field(default=None, alias="minPrice", description="最低价")
    max_s_price: int | None = Field(default=None, alias="maxSPrice", description="最高价")


class AbroadTopGoodsSlimResult(BaseModel):
    """海外探款Top商品精简结果"""
    model_config = ConfigDict(populate_by_name=True)

    start: int = Field(default=0)
    page_size: int = Field(default=10, alias="pageSize")
    result_count: int = Field(default=0, alias="resultCount")
    top10_slim: list[AbroadTopGoodsSlimItem] = Field(default_factory=list)


class AbroadTopGoodsCleanResponse(BaseModel):
    """海外探款Top商品清洗后响应"""
    success: bool = True
    result: AbroadTopGoodsSlimResult | None = None


# ========== 聚合数据模型（用于报告生成） ==========

class AbroadAggregatedData(BaseModel):
    """海外探款聚合分析数据 - 用于报告生成"""
    model_config = ConfigDict(populate_by_name=True)

    # 分析维度标识
    table_type: int = Field(description="分析类型：1-品类分析，2-非品类分析")
    # 趋势数据
    trend_data: AbroadTrendCleanResponse | None = Field(default=None, alias="trendData", description="趋势数据")
    # 品类数据
    category_data: AbroadDimensionInfoCleanResponse | None = Field(default=None, alias="categoryData", description="品类分布数据")
    # 颜色数据
    color_data: AbroadColorCleanResponse | None = Field(default=None, alias="colorData", description="颜色分布数据")
    # 价格带数据
    price_data: AbroadPriceCleanResponse | None = Field(default=None, alias="priceData", description="价格带分布数据")
    # 属性数据（面料、袖长等）
    property_data: AbroadPropertyCleanResponse | None = Field(default=None, alias="propertyData", description="属性分布数据")
    # Top商品数据
    top_goods: AbroadTopGoodsCleanResponse | None = Field(default=None, alias="topGoods", description="Top10商品数据")
    top_goods_raw: PageResult[AbroadGoodsEntity] | None = Field(default=None, alias="topGoodsRaw", description="top商品数据原结果")


# ========== API 原始响应模型（未清洗） ==========
# 以下模型用于接收后端 API 的原始响应，后续由清洗方法转换为上述精简模型
# TODO: 字段结构根据清洗后模型反推，待实际 API 响应校验后补充完善

class AbroadTrendSummaryRawItem(BaseModel):
    """trend-summary 接口原始数据项（单个趋势点）"""
    model_config = ConfigDict(populate_by_name=True)

    record_date: str | None = Field(default=None, alias="recordDate", description="日期/周次")
    key: str | None = Field(default=None, description="日期键")
    value: int | None = Field(default=None, description="数值")


class AbroadTrendSummaryBucketItem(BaseModel):
    """trend-summary 接口原始数据 bucket 项"""
    model_config = ConfigDict(populate_by_name=True)

    bucket: str = Field(default="", description="bucket标识")
    currency: str = Field(default="USD", description="货币单位")
    sale_volume_trend: list[AbroadTrendSummaryRawItem] = Field(
        default_factory=list, alias="saleVolumeTrend", description="销量趋势列表"
    )
    sale_amount_trend: list[AbroadTrendSummaryRawItem] = Field(
        default_factory=list, alias="saleAmountTrend", description="销售额趋势列表"
    )
    goods_num_trend: list[AbroadTrendSummaryRawItem] = Field(
        default_factory=list, alias="goodsNumTrend", description="商品数趋势列表"
    )
    shop_num_trend: list[AbroadTrendSummaryRawItem] = Field(
        default_factory=list, alias="shopNumTrend", description="店铺数趋势列表"
    )


class AbroadTrendSummaryRawData(BaseModel):
    """trend-summary 接口原始数据结构（兼容旧结构）"""
    model_config = ConfigDict(populate_by_name=True)

    sale_volume_trend: list[AbroadTrendSummaryRawItem] = Field(
        default_factory=list, alias="saleVolumeTrend", description="销量趋势列表"
    )
    sale_amount_trend: list[AbroadTrendSummaryRawItem] = Field(
        default_factory=list, alias="saleAmountTrend", description="销售额趋势列表"
    )
    currency: str = Field(default="USD", description="货币单位")


class AbroadTrendSummaryRawResponse(BaseModel):
    """trend-summary 接口原始响应"""
    model_config = ConfigDict(populate_by_name=True)

    success: bool = Field(default=True, description="请求是否成功")
    error_code: str | None = Field(default=None, alias="errorCode", description="错误码")
    error_desc: str | None = Field(default=None, alias="errorDesc", description="错误描述")
    result: list[AbroadTrendSummaryBucketItem] = Field(default_factory=list, description="数据结果（数组格式）")


class AbroadDimensionInfoRawItem(BaseModel):
    """info 接口原始数据项（品类/颜色/价格带通用）"""
    model_config = ConfigDict(populate_by_name=True)

    dimension_info: str | None = Field(default=None, alias="dimensionInfo", description="维度值")
    sale_volume: int | None = Field(default=0, alias="saleVolume", description="销量")
    sale_amount: int | None = Field(default=0, alias="saleAmount", description="销售额")
    sale_volume_ratio: float | None = Field(default=0.0, alias="saleVolumeRatio", description="销量占比")
    sale_amount_ratio: float | None = Field(default=0.0, alias="saleAmountRatio", description="销售额占比")
    sale_volume_mom_ratio: float | None = Field(default=None, alias="saleVolumeMomRatio", description="销量环比")
    sale_amount_mom_ratio: float | None = Field(default=None, alias="saleAmountMomRatio", description="销售额环比")
    sale_volume_yoy_ratio: float | None = Field(default=None, alias="saleVolumeYoyRatio", description="销量同比")
    sale_amount_yoy_ratio: float | None = Field(default=None, alias="saleAmountYoyRatio", description="销售额同比")


class AbroadDimensionInfoSummary(BaseModel):
    """info 接口原始数据汇总"""
    model_config = ConfigDict(populate_by_name=True)

    sale_volume: int | None = Field(default=0, alias="saleVolume", description="总销量")
    sale_amount: int | None = Field(default=0, alias="saleAmount", description="总销售额")
    sale_volume_mom_ratio: float | None = Field(default=None, alias="saleVolumeMomRatio", description="销量环比")
    sale_amount_mom_ratio: float | None = Field(default=None, alias="saleAmountMomRatio", description="销售额环比")
    sale_volume_yoy_ratio: float | None = Field(default=None, alias="saleVolumeYoyRatio", description="销量同比")
    sale_amount_yoy_ratio: float | None = Field(default=None, alias="saleAmountYoyRatio", description="销售额同比")
    goods_num: int | None = Field(default=0, alias="goodsNum", description="商品数")
    shop_num: int | None = Field(default=0, alias="shopNum", description="店铺数")


class AbroadDimensionInfoRawData(BaseModel):
    """info 接口原始数据结构"""
    model_config = ConfigDict(populate_by_name=True)

    summary: AbroadDimensionInfoSummary | None = Field(default=None, description="汇总数据")
    data_list: list[AbroadDimensionInfoRawItem] = Field(default_factory=list, alias="list", description="维度数据列表")
    result_count: int | None = Field(default=0, alias="resultCount", description="结果数量")


class AbroadDimensionInfoRawResponse(BaseModel):
    """info 接口原始响应"""
    model_config = ConfigDict(populate_by_name=True)

    success: bool = Field(default=True, description="请求是否成功")
    error_code: str | None = Field(default=None, alias="errorCode", description="错误码")
    error_desc: str | None = Field(default=None, alias="errorDesc", description="错误描述")
    result: AbroadDimensionInfoRawData | None = Field(default=None, description="数据结果")


class AbroadColorRawItem(BaseModel):
    """颜色数据原始项（可能与info结构不同）"""
    model_config = ConfigDict(populate_by_name=True)

    property_value: str | None = Field(default=None, alias="propertyValue", description="颜色值")
    sales_volume: int = Field(default=0, alias="salesVolume", description="销量")
    sales_amount: int = Field(default=0, alias="salesAmount", description="销售额")
    rate: float = Field(default=0.0, description="占比")
    other_flag: bool = Field(default=False, alias="otherFlag", description="是否为'其他'合并项")
    # TODO: 根据实际API响应补充字段


class AbroadColorRawResponse(BaseModel):
    """颜色数据原始响应"""
    model_config = ConfigDict(populate_by_name=True)

    success: bool = Field(default=True, description="请求是否成功")
    error_code: str | None = Field(default=None, alias="errorCode", description="错误码")
    error_desc: str | None = Field(default=None, alias="errorDesc", description="错误描述")
    result: list[AbroadColorRawItem] = Field(default_factory=list, description="颜色数据列表")


class AbroadPriceRawItem(BaseModel):
    """价格带原始项"""
    model_config = ConfigDict(populate_by_name=True)

    left_price: int = Field(default=0, alias="leftPrice", description="价格下限（分）")
    right_price: int | None = Field(default=None, alias="rightPrice", description="价格上限（分）")
    sales_volume: int = Field(default=0, alias="salesVolume", description="销量")
    rate: float = Field(default=0.0, description="占比")
    # TODO: 根据实际API响应补充字段


class AbroadPriceRawResponse(BaseModel):
    """价格带原始响应"""
    model_config = ConfigDict(populate_by_name=True)

    success: bool = Field(default=True, description="请求是否成功")
    error_code: str | None = Field(default=None, alias="errorCode", description="错误码")
    error_desc: str | None = Field(default=None, alias="errorDesc", description="错误描述")
    result: list[AbroadPriceRawItem] = Field(default_factory=list, description="价格带数据列表")


class AbroadPropertyTrendRawItem(BaseModel):
    """属性趋势原始数据项"""
    model_config = ConfigDict(populate_by_name=True)

    date_range: str | None = Field(default=None, alias="dateRange", description="日期范围")
    sales_volume: int = Field(default=0, alias="salesVolume", description="销量")


class AbroadPropertyRawItem(BaseModel):
    """属性数据原始项（v2/trend接口）"""
    model_config = ConfigDict(populate_by_name=True)

    property_value: str | None = Field(default=None, alias="propertyValue", description="属性值")
    sales_volume: int = Field(default=0, alias="salesVolume", description="销量")
    sales_amount: int = Field(default=0, alias="salesAmount", description="销售额")
    rate: float = Field(default=0.0, description="占比")
    other_flag: bool = Field(default=False, alias="otherFlag", description="是否为'其他'合并项")
    trends: list[AbroadPropertyTrendRawItem] = Field(default_factory=list, description="趋势数据")


class AbroadPropertyRawResponse(BaseModel):
    """属性数据原始响应"""
    model_config = ConfigDict(populate_by_name=True)

    success: bool = Field(default=True, description="请求是否成功")
    error_code: str | None = Field(default=None, alias="errorCode", description="错误码")
    error_desc: str | None = Field(default=None, alias="errorDesc", description="错误描述")
    result: list[AbroadPropertyRawItem] | None = Field(default_factory=list, description="属性数据列表")


class AbroadTopGoodsRawItem(BaseModel):
    """goods-zone-list 接口原始商品数据项"""
    model_config = ConfigDict(populate_by_name=True)

    product_id: str | None = Field(default=None, alias="productId", description="商品ID")
    product_name: str | None = Field(default=None, alias="productName", description="商品标题")
    pic_url: str | None = Field(default=None, alias="picUrl", description="商品图片URL")
    category_id: str | None = Field(default=None, alias="categoryId", description="品类ID")
    category_name: str | None = Field(default=None, alias="categoryName", description="品类名称")
    category_detail: str | None = Field(default=None, alias="categoryDetail", description="品类路径")
    sale_volume_30day: int = Field(default=0, alias="saleVolume30Day", description="近30天销量")
    sale_amount_30day: int = Field(default=0, alias="saleAmount30Day", description="近30天销售额")
    min_price: float | None = Field(default=None, alias="minPrice", description="最低价")
    max_s_price: float | None = Field(default=None, alias="maxSPrice", description="最高价")
    sprice: float | None = Field(default=None, alias="sprice", description="售价")
    # TODO: 根据实际API响应补充字段


class AbroadTopGoodsRawData(BaseModel):
    """goods-zone-list 接口原始数据结构"""
    model_config = ConfigDict(populate_by_name=True)

    start: int = Field(default=0, description="起始位置")
    page_size: int = Field(default=10, alias="pageSize", description="页大小")
    result_count: int = Field(default=0, alias="resultCount", description="结果数量")
    result_list: list[AbroadTopGoodsRawItem] = Field(default_factory=list, alias="resultList", description="商品列表")


class AbroadTopGoodsRawResponse(BaseModel):
    """goods-zone-list 接口原始响应"""
    model_config = ConfigDict(populate_by_name=True)

    success: bool = Field(default=True, description="请求是否成功")
    error_code: str | None = Field(default=None, alias="errorCode", description="错误码")
    error_desc: str | None = Field(default=None, alias="errorDesc", description="错误描述")
    result: AbroadTopGoodsRawData | None = Field(default=None, description="数据结果")


class AbroadPropertyListRawItem(BaseModel):
    """属性列表原始项（property-list接口）"""
    model_config = ConfigDict(populate_by_name=True)

    property_name: str | None = Field(default=None, alias="propertyName", description="属性名称")
    property_values: list[str] = Field(default_factory=list, alias="propertyValues", description="属性值列表")


class AbroadPropertyListRawResponse(BaseModel):
    """属性列表原始响应"""
    model_config = ConfigDict(populate_by_name=True)

    success: bool = Field(default=True, description="请求是否成功")
    error_code: str | None = Field(default=None, alias="errorCode", description="错误码")
    error_desc: str | None = Field(default=None, alias="errorDesc", description="错误描述")
    result: list[AbroadPropertyListRawItem] = Field(default_factory=list, description="属性列表")


# ========== 类目解析模型 ==========

class AbroadCategoryFormatItem(BaseModel):
    """
    海外探款-类目格式化项

    用于表示单个类目解析结果

    示例输入：
        key: "女装,裤装,长裤"
        value: "1,343,372"

    示例输出：
        category_path: "女装,裤装,长裤"
        category_id_path: "1,343,372"
        level: 3
    """
    category_path: str | None = Field(default=None, description="类目名称路径，如 '女装,裤装,长裤'")
    category_id_path: str | None = Field(default=None, description="类目ID路径，如 '1,343,372'")
    level: int = Field(default=1, description="类目层级：1/2/3")


class AbroadParsedCategory(BaseModel):
    """
    海外探款-解析后的类目列表

    用于包装向量检索后的类目解析结果
    """
    category_list: list[AbroadCategoryFormatItem] = Field(
        default_factory=list,
        description="处理后的类目列表"
    )


# ============== 类目解析方法 ==============

def parse_abroad_category_list(raw_data: list[dict]) -> AbroadParsedCategory:
    """
    解析海外探款类目向量检索结果

    Args:
        raw_data: 向量检索返回的原始数据
            格式: [{"key": "女装,裤装,长裤", "value": "1,343,372"}, ...]

    Returns:
        AbroadParsedCategory: 解析后的类目列表

    Examples:
        >>> raw = [
        ...     {"key": "女装", "value": "1"},
        ...     {"key": "女装,裤装", "value": "1,343"},
        ...     {"key": "女装,裤装,长裤", "value": "1,343,372"},
        ... ]
        >>> result = parse_abroad_category_list(raw)
        >>> len(result.category_list)
        3
        >>> result.category_list[2].level
        3
    """
    parsed_list = []

    for item in raw_data:
        category_path = item.get("key", "")
        category_id_path = item.get("value", "")

        if not category_path or not category_id_path:
            continue

        # 分割路径计算层级
        category_name_arr = [s.strip() for s in category_path.split(",") if s.strip()]
        category_id_arr = [s.strip() for s in category_id_path.split(",") if s.strip()]

        if not category_name_arr or not category_id_arr:
            continue

        # 取较小的层级数
        level = min(len(category_name_arr), len(category_id_arr))

        formatted_item = AbroadCategoryFormatItem(
            category_path=category_path,
            category_id_path=category_id_path,
            level=level,
        )
        parsed_list.append(formatted_item)

    return AbroadParsedCategory(category_list=parsed_list)
