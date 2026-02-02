# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2026/1/13 16:43
# @File     : deepresearch_graph_state.py
from __future__ import annotations
from typing import Any, Dict, List
from pydantic import BaseModel, Field

from app.schemas.request.workflow_request import WorkflowRequest
from app.service.chains.workflow.deepresearch.deepresearch_message_pusher import DeepresearchMessagePusher
from app.service.chains.workflow.deepresearch.douyi.schema import (
    DouyiParsedCategory,
    DouyiMainParseParam,
    DouyiTrendCleanResponse,
    DouyiPriceCleanResponse,
    DouyiPropertyCleanResponse,
    DouyiTopGoodsCleanResponse,
    DouyiAggregatedData,
)
from app.service.chains.workflow.deepresearch.douyi.llm_output import (
    DouyiExtractedProperty,
    DouyiExtractedPriceRange,
)
from app.service.chains.workflow.deepresearch.zhiyi.schema import (
    ZhiyiParsedCategory,
    ZhiyiThinkingApiParseParam,
    ZhiyiAggregatedData,
)
from app.service.chains.workflow.deepresearch.abroad.schema import (
    AbroadMainParseParam,
    AbroadParsedCategory,
    AbroadTrendCleanResponse,
    AbroadDimensionInfoCleanResponse,
    AbroadColorCleanResponse,
    AbroadPriceCleanResponse,
    AbroadPropertyCleanResponse,
    AbroadTopGoodsCleanResponse,
    AbroadAggregatedData,
)
from app.schemas.response.common import PageResult
from app.service.rpc.abroad.schemas import AbroadGoodsEntity
from app.service.rpc.zhiyi.schemas import (
    SaleTrendCleanResponse,
    PriceRangeCleanResponse,
    ColorCleanResponse,
    CategoryTrendCleanResponse,
    Top10ItemsCleanResponse,
    BrandCleanResponse, ShopHotItemRawResponse,
    DouyiGoodsEntity,
)


class BaseDeepresearchGraphState(BaseModel):
    """
    deepresearch工作流需要通用使用的状态字段
    """
    model_config = {"arbitrary_types_allowed": True}

    request: WorkflowRequest = Field(default=None)
    workflow_response: Any = Field(default=None)

    # 消息推送器实例（非序列化字段）
    message_pusher: DeepresearchMessagePusher = Field(default=None, exclude=True)


class ZhiyiDeepresearchState(BaseDeepresearchGraphState):
    """
    知衣deepresearch
    """
    # 单独的是否深度思考字段，方便追踪
    is_thinking: bool = False

    recall_category: ZhiyiParsedCategory = Field(default=None, description="解析后的向量召回的品类列表")
    api_parse_param: ZhiyiThinkingApiParseParam = Field(default=None, description="api参数解析结果")
    target_shop_id: int | None = Field(default=None, description="选定的店铺id")
    target_shop_name: str | None = Field(default=None, description="选定的店铺名")
    price_band_list: list[str] | None = Field(default=None, description="解析得到的价格带")

    # 店铺分析结果 - 使用类型化的清洗响应模型
    shop_overview_volume: SaleTrendCleanResponse = Field(default=None, description="店铺概览-销量趋势")
    shop_overview_amount: SaleTrendCleanResponse = Field(default=None, description="店铺概览-销售额趋势")
    shop_price_trend: PriceRangeCleanResponse = Field(default=None, description="店铺价格带数据")
    shop_color_data: ColorCleanResponse = Field(default=None, description="店铺颜色数据")
    shop_property_data: ColorCleanResponse = Field(default=None, description="店铺属性数据")
    shop_category_trend: CategoryTrendCleanResponse = Field(default=None, description="店铺品类趋势")
    shop_top_goods: Top10ItemsCleanResponse = Field(default=None, description="店铺Top10商品")
    shop_top_goods_raw: ShopHotItemRawResponse = Field(default=None, description="店铺top商品原结果")

    # 属性分析辅助
    shop_property_name: str | None = Field(default=None, description="提取的属性名称")
    shop_property_values: list[str] | None = Field(default=None, description="可用属性列表")
    # 价格分析辅助

    # ========== 大盘分析维度判断 ==========
    hydc_dimension_type: int | None = Field(default=None, description="大盘维度类型：0=属性/1=颜色/2=品牌")

    # ========== 大盘品类分析结果 (table_type=1) ==========
    hydc_category_overview_volume: SaleTrendCleanResponse | None = Field(default=None, description="大盘品类-销量趋势")
    hydc_category_overview_amount: SaleTrendCleanResponse | None = Field(default=None, description="大盘品类-销售额趋势")
    hydc_category_price_data: PriceRangeCleanResponse | None = Field(default=None, description="大盘品类-价格带数据")
    hydc_category_color_data: ColorCleanResponse | None = Field(default=None, description="大盘品类-颜色数据")
    hydc_category_category_data: CategoryTrendCleanResponse | None = Field(default=None, description="大盘品类-品类趋势")
    hydc_category_top_goods: Top10ItemsCleanResponse | None = Field(default=None, description="大盘品类-Top10商品")
    hydc_category_top_goods_raw: ShopHotItemRawResponse | None = Field(default=None, description="大盘品类-top商品原结果")

    # ========== 大盘属性分析结果 (table_type=2, dimension=0) ==========
    hydc_property_overview_volume: SaleTrendCleanResponse | None = Field(default=None, description="大盘属性-销量趋势")
    hydc_property_overview_amount: SaleTrendCleanResponse | None = Field(default=None, description="大盘属性-销售额趋势")
    hydc_property_data: ColorCleanResponse | None = Field(default=None, description="大盘属性-属性分布数据")
    hydc_property_top_goods: Top10ItemsCleanResponse | None = Field(default=None, description="大盘属性-Top10商品")
    hydc_property_top_goods_raw: ShopHotItemRawResponse | None = Field(default=None, description="大盘属性-top商品原结果")
    hydc_property_name: str | None = Field(default=None, description="大盘属性-提取的属性名")
    hydc_property_values: list[str] | None = Field(default=None, description="大盘属性-提取的属性列表")

    # ========== 大盘颜色分析结果 (table_type=2, dimension=1) ==========
    hydc_color_overview_volume: SaleTrendCleanResponse | None = Field(default=None, description="大盘颜色-销量趋势")
    hydc_color_overview_amount: SaleTrendCleanResponse | None = Field(default=None, description="大盘颜色-销售额趋势")
    hydc_color_data: ColorCleanResponse | None = Field(default=None, description="大盘颜色-颜色分布数据")
    hydc_color_top_goods: Top10ItemsCleanResponse | None = Field(default=None, description="大盘颜色-Top10商品")
    hydc_color_top_goods_raw: ShopHotItemRawResponse | None = Field(default=None, description="大盘颜色-top商品原结果")

    # ========== 大盘品牌分析结果 (table_type=2, dimension=2) ==========
    hydc_brand_overview_volume: SaleTrendCleanResponse | None = Field(default=None, description="大盘品牌-销量趋势")
    hydc_brand_overview_amount: SaleTrendCleanResponse | None = Field(default=None, description="大盘品牌-销售额趋势")
    hydc_brand_data: BrandCleanResponse | None = Field(default=None, description="大盘品牌-品牌分布数据")
    hydc_brand_top_goods: Top10ItemsCleanResponse | None = Field(default=None, description="大盘品牌-Top10商品")
    hydc_brand_top_goods_raw: ShopHotItemRawResponse | None = Field(default=None, description="大盘品牌-top商品原结果")

    # ========== 大盘价格带分析结果 (table_type=3) ==========
    hydc_price_overview_volume: SaleTrendCleanResponse | None = Field(default=None, description="大盘价格带-销量趋势")
    hydc_price_overview_amount: SaleTrendCleanResponse | None = Field(default=None, description="大盘价格带-销售额趋势")
    hydc_price_data: PriceRangeCleanResponse | None = Field(default=None, description="大盘价格带-价格带分布数据")
    hydc_price_top_goods: Top10ItemsCleanResponse | None = Field(default=None, description="大盘价格带-Top10商品")
    hydc_price_top_goods_raw: ShopHotItemRawResponse | None = Field(default=None, description="大盘价格带-top商品原结果")

    # ========== 聚合后的分析数据（用于报告生成和Excel导出） ==========
    aggregated_data: ZhiyiAggregatedData | None = Field(
        default=None,
        description="聚合后的分析数据"
    )

    # 分析报告内容
    report_text: str = Field(default=None, description="生成的报告内容文本")



class DouyiDeepresearchState(BaseDeepresearchGraphState):
    """
    抖衣deepresearch
    """
    # 单独的是否深度思考字段，方便追踪
    is_thinking: bool = False

    recall_category: DouyiParsedCategory | None = Field(default=None, description="向量召回的类目")
    main_parse_param: DouyiMainParseParam | None = Field(default=None, description="主要参数解析结果")

    # ========== 商品类型查询参数 ==========
    goods_relate_type: str | None = Field(default=None, description="商品关联类型(windowGoods等)")
    target_sale_volume: str | None = Field(default=None, description="目标销量类型(saleVolume/videoSaleVolume等)")
    target_sale_amount: str | None = Field(default=None, description="目标销售额类型(saleAmount/videoSaleAmount等)")

    # ========== 品类分析分支结果 ==========
    category_volume_trend: DouyiTrendCleanResponse | None = Field(default=None, description="品类分支-销量趋势")
    category_amount_trend: DouyiTrendCleanResponse | None = Field(default=None, description="品类分支-销售额趋势")
    category_price_data: DouyiPriceCleanResponse | None = Field(default=None, description="品类分支-价格带数据")
    category_top_goods: DouyiTopGoodsCleanResponse | None = Field(default=None, description="品类分支-Top商品")
    category_top_resp: PageResult[DouyiGoodsEntity] | None = Field(default=None, description="品类分支-Top商品源结果")

    # ========== 属性分析分支结果 ==========
    property_volume_trend: DouyiTrendCleanResponse | None = Field(default=None, description="属性分支-销量趋势")
    property_amount_trend: DouyiTrendCleanResponse | None = Field(default=None, description="属性分支-销售额趋势")
    property_data: DouyiPropertyCleanResponse | None = Field(default=None, description="属性分支-属性分布数据")
    property_top_goods: DouyiTopGoodsCleanResponse | None = Field(default=None, description="属性分支-Top商品")
    property_top_resp: PageResult[DouyiGoodsEntity] | None = Field(default=None, description="属性分支-Top商品源结果")
    extracted_property: DouyiExtractedProperty | None = Field(default=None, description="LLM提取的属性")
    available_property_list: str | None = Field(default=None, description="可用属性列表")

    # ========== 价格带分析分支结果 ==========
    price_volume_trend: DouyiTrendCleanResponse | None = Field(default=None, description="价格带分支-销量趋势")
    price_amount_trend: DouyiTrendCleanResponse | None = Field(default=None, description="价格带分支-销售额趋势")
    price_price_data: DouyiPriceCleanResponse | None = Field(default=None, description="价格带分支-价格带数据")
    price_top_goods: DouyiTopGoodsCleanResponse | None = Field(default=None, description="价格带分支-Top商品")
    price_top_resp: PageResult[DouyiGoodsEntity] | None = Field(default=None, description="价格带分支-Top商品源结果")
    extracted_price_range: DouyiExtractedPriceRange | None = Field(default=None, description="LLM提取的价格范围")

    # ========== 聚合后的分析数据（用于报告生成） ==========
    aggregated_data: DouyiAggregatedData | None = Field(default=None, description="聚合后的分析数据")

    # 分析报告内容
    report_text: str | None = Field(default=None, description="生成的报告内容文本")


class AbroadDeepresearchState(BaseDeepresearchGraphState):
    """
    海外探款deepresearch
    """
    # 基础信息
    is_thinking: bool = False

    # 参数解析结果
    main_parse_param: AbroadMainParseParam | None = Field(default=None, description="API参数解析结果")
    target_platform_type_list: list[int] = Field(default_factory=list, description="站点类型ID")
    recall_platforms: list[dict] = Field(default_factory=list, description="站点检索结果-可用站点列表")
    recall_category: AbroadParsedCategory | None = Field(default=None, description="解析后的向量召回的品类列表")

    # ========== 品类分析分支结果（table_type=1） ==========
    category_trend_data: AbroadTrendCleanResponse | None = Field(default=None, description="品类分支-趋势数据")
    category_category_data: AbroadDimensionInfoCleanResponse | None = Field(default=None, description="品类分支-品类分布")
    category_color_data: AbroadColorCleanResponse | None = Field(default=None, description="品类分支-颜色分布")
    category_price_data: AbroadPriceCleanResponse | None = Field(default=None, description="品类分支-价格带分布")
    category_top_goods: AbroadTopGoodsCleanResponse | None = Field(default=None, description="品类分支-Top商品")
    category_top_goods_raw: PageResult[AbroadGoodsEntity] | None = Field(default=None, description="品类分支-Top商品原结果")

    # ========== 非品类分析分支结果（table_type=2） ==========
    property_trend_data: AbroadTrendCleanResponse | None = Field(default=None, description="非品类分支-趋势数据")
    property_category_data: AbroadDimensionInfoCleanResponse | None = Field(default=None, description="非品类分支-品类分布")
    property_top_goods: AbroadTopGoodsCleanResponse | None = Field(default=None, description="非品类分支-Top商品")
    property_top_goods_raw: PageResult[AbroadGoodsEntity] | None = Field(default=None, description="非品类分支-top商品原结果")
    property_dimension_type: int | None = Field(default=None, description="用户查询的维度类型：0其他/1颜色/2面料/3价格带")
    property_dimension_data: AbroadPropertyCleanResponse | None = Field(default=None, description="动态维度数据")
    extracted_property: str | None = Field(default=None, description="LLM提取的属性名称")

    # ========== 聚合后的分析数据（用于报告生成） ==========
    aggregated_data: AbroadAggregatedData | None = Field(default=None, description="聚合后的分析数据")

    # 分析报告内容
    report_text: str | None = Field(default=None, description="生成的报告内容文本")
