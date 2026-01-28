# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2026/1/14 10:23
# @File     : __init__.py.py

from app.service.chains.workflow.deepresearch.abroad.schema import (
    # 类目解析相关
    AbroadCategoryFormatItem,
    AbroadParsedCategory,
    parse_abroad_category_list,
    # API参数解析
    AbroadMainParseParam,
    AbroadPlatformType,
    # 维度类型
    AbroadDimensionType,
    # 趋势数据
    AbroadTrendSlimItem,
    AbroadTrendSlimResult,
    AbroadTrendCleanResponse,
    # 维度分析数据
    AbroadDimensionInfoItem,
    AbroadDimensionInfoResult,
    AbroadDimensionInfoCleanResponse,
    # 颜色数据
    AbroadColorSlimItem,
    AbroadColorCleanResponse,
    # 价格带数据
    AbroadPriceSlimItem,
    AbroadPriceCleanResponse,
    # 属性趋势数据
    AbroadPropertyTrendItem,
    AbroadPropertySlimItem,
    AbroadPropertyCleanResponse,
    # Top商品数据
    AbroadTopGoodsSlimItem,
    AbroadTopGoodsSlimResult,
    AbroadTopGoodsCleanResponse,
    # 聚合数据
    AbroadAggregatedData,
    # 原始响应模型
    AbroadTrendSummaryRawItem,
    AbroadTrendSummaryRawData,
    AbroadTrendSummaryRawResponse,
    AbroadDimensionInfoRawItem,
    AbroadDimensionInfoRawData,
    AbroadDimensionInfoRawResponse,
    AbroadColorRawItem,
    AbroadColorRawResponse,
    AbroadPriceRawItem,
    AbroadPriceRawResponse,
    AbroadPropertyTrendRawItem,
    AbroadPropertyRawItem,
    AbroadPropertyRawResponse,
    AbroadTopGoodsRawItem,
    AbroadTopGoodsRawData,
    AbroadTopGoodsRawResponse,
    AbroadPropertyListRawItem,
    AbroadPropertyListRawResponse,
)

__all__ = [
    # 类目解析相关
    "AbroadCategoryFormatItem",
    "AbroadParsedCategory",
    "parse_abroad_category_list",
    # API参数解析
    "AbroadMainParseParam",
    "AbroadPlatformType",
    # 维度类型
    "AbroadDimensionType",
    # 趋势数据
    "AbroadTrendSlimItem",
    "AbroadTrendSlimResult",
    "AbroadTrendCleanResponse",
    # 维度分析数据
    "AbroadDimensionInfoItem",
    "AbroadDimensionInfoResult",
    "AbroadDimensionInfoCleanResponse",
    # 颜色数据
    "AbroadColorSlimItem",
    "AbroadColorCleanResponse",
    # 价格带数据
    "AbroadPriceSlimItem",
    "AbroadPriceCleanResponse",
    # 属性趋势数据
    "AbroadPropertyTrendItem",
    "AbroadPropertySlimItem",
    "AbroadPropertyCleanResponse",
    # Top商品数据
    "AbroadTopGoodsSlimItem",
    "AbroadTopGoodsSlimResult",
    "AbroadTopGoodsCleanResponse",
    # 聚合数据
    "AbroadAggregatedData",
    # 原始响应模型
    "AbroadTrendSummaryRawItem",
    "AbroadTrendSummaryRawData",
    "AbroadTrendSummaryRawResponse",
    "AbroadDimensionInfoRawItem",
    "AbroadDimensionInfoRawData",
    "AbroadDimensionInfoRawResponse",
    "AbroadColorRawItem",
    "AbroadColorRawResponse",
    "AbroadPriceRawItem",
    "AbroadPriceRawResponse",
    "AbroadPropertyTrendRawItem",
    "AbroadPropertyRawItem",
    "AbroadPropertyRawResponse",
    "AbroadTopGoodsRawItem",
    "AbroadTopGoodsRawData",
    "AbroadTopGoodsRawResponse",
    "AbroadPropertyListRawItem",
    "AbroadPropertyListRawResponse",
]

