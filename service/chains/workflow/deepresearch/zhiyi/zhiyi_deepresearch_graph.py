# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2026/1/13 17:14
# @File     : zhiyi_deepresearch_graph.py
import json
import re
from datetime import datetime, timedelta

from dateutil.relativedelta import relativedelta
from enum import StrEnum
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from loguru import logger
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from app.core.clients.coze_loop_client import coze_loop_client_provider
from app.core.config.constants import VolcKnowledgeServiceId, LlmProvider, LlmModelName, CozePromptHubKey, \
    WorkflowEntityType
from app.core.tools import llm_factory
from app.service.chains.workflow.base_graph import BaseWorkflowGraph
from app.service.chains.workflow.deepresearch.deepresearch_graph_state import ZhiyiDeepresearchState, \
    ZhiyiThinkingApiParseParam
from app.service.chains.workflow.deepresearch.deepresearch_message_pusher import DeepresearchMessagePusher
from app.service.chains.workflow.deepresearch.zhiyi.llm_output import (
    ShopCleanResult,
    PropertyExtractResult,
    PriceBandExtractResult,
    HydcDimensionAnalyzeResult,
    HydcPropertyExtractResult,
)
from app.service.chains.workflow.deepresearch.zhiyi.schema import (
    ZhiyiCategoryFormatItem,
    ZhiyiParsedCategory,
    ZhiyiAggregatedData,
)
from app.service.chains.workflow.deepresearch.zhiyi.excel_exporter import ZhiyiExcelExporter
from app.service.rpc.volcengine_kb_api import get_volcengine_kb_api
from app.service.rpc.zhiyi.client import get_zhiyi_api_client
from app.service.rpc.zhiyi.schemas import (
    ZhiyiSaleTrendRequest,
    ZhiyiPriceRangeTrendRequest,
    ZhiyiPropertyTrendRequest,
    ZhiyiCategoryTrendRequest,
    ZhiyiShopHotItemRequest,
    ZhiyiPropertyTopRequest,
    # 大盘请求模型
    ZhiyiHydcTrendListRequest,
    ZhiyiHydcRankItemRequest,
    # 原始响应模型
    SaleTrendRawResponse,
    PriceRangeRawResponse,
    PropertyTrendRawResponse,
    CategoryTrendRawResponse,
    ShopHotItemRawResponse,
    PropertyTopRawResponse,
    BrandRawResponse,
    # 大盘趋势响应模型
    HydcTrendListRawResponse,
    # 清洗后响应模型
    SaleTrendCleanResponse,
    PriceRangeCleanResponse,
    ColorCleanResponse,
    CategoryTrendCleanResponse,
    Top10ItemsCleanResponse,
    BrandCleanResponse,
    SaleTrendSlimResult,
    PriceRangeSlimResult,
    ColorSlimResult,
    Top10ItemsSlimResult,
    CategoryTrendSlimResult,
    BrandSlimResult,
    SaleTrendSlimItem,
    PriceRangeSlimItem,
    ColorSlimItem,
    Top10ItemSlim,
    CategoryTrendSlimItem,
    BrandSlimItem,
)


class NodeKey(StrEnum):
    """思考子图节点标识"""
    # 公共节点
    INIT = "初始化"
    CATEGORY_SEARCH = "知衣品类检索"
    API_PARAM_PARSE = "api参数解析"
    ROUTE_IS_SHOP = "是否店铺路由"
    SHOP_SEARCH = "店铺检索"

    # 通用数据查询
    MATCH_PROPERTY = "匹配属性名"

    # 店铺 - 品类分析
    CATEGORY_SHOP_START = "店铺品类分析分支-开始"
    CATEGORY_SHOP_OVERVIEW = "品类分析分支-店铺概览"
    CATEGORY_SHOP_PRICE = "品类分析分支-店铺价格带"
    CATEGORY_SHOP_COLOR = "品类分析分支-店铺颜色"
    CATEGORY_SHOP_CATEGORY = "品类分析分支-店铺品类"
    CATEGORY_SHOP_TOP_GOODS = "品类分析分支-店铺top商品"
    CATEGORY_SHOP_AGGREGATE = "品类分析分支-店铺分析结果聚合"

    # 店铺 - 属性分析
    PROPERTY_SHOP_START = "店铺属性分支-开始"
    PROPERTY_SHOP_OVERVIEW = "属性分支-店铺概览"
    PROPERTY_SHOP_PROPERTY = "属性分支-店铺属性"
    PROPERTY_SHOP_TOP_GOODS = "属性分支-店铺top商品"
    PROPERTY_SHOP_AGGREGATE = "属性分支-店铺分析结果聚合"

    # 店铺 - 价格带分析
    SHOP_EXTRACT_PRICE_BAND = "店铺价格带分支-抽取价格带"
    PRICE_SHOP_START = "店铺价格带分支-开始"
    PRICE_SHOP_OVERVIEW = "价格带分支-店铺概览"
    PRICE_SHOP_PRICE = "价格带分支-店铺价格带"
    PRICE_SHOP_TOP_GOODS = "价格带分支-店铺top商品"
    PRICE_SHOP_AGGREGATE = "价格带分支-店铺分析结果聚合"

    # 店铺-报告生成
    CATEGORY_SHOP_REPORT = "品类分析分支-生成店铺报告"
    NORMAL_SHOP_REPORT = "非品类分支-生成店铺报告"

    # 店铺-非深度思考报告生成
    CATEGORY_SHOP_NONTHINKING_REPORT = "品类分析分支-生成店铺非思考报告"
    NORMAL_SHOP_NONTHINKING_REPORT = "非品类分支-生成店铺非思考报告"

    # ========== 大盘分析通用节点 ==========
    HYDC_TABLE_TYPE_ROUTER = "大盘table_type路由"
    HYDC_DIMENSION_ANALYZE = "大盘维度分析"  # LLM判断维度类型

    # ========== 大盘品类分析 (table_type=1) ==========
    HYDC_CATEGORY_START = "大盘品类分支-开始"
    HYDC_CATEGORY_OVERVIEW = "大盘品类分支-概览趋势"
    HYDC_CATEGORY_PRICE = "大盘品类分支-价格带"
    HYDC_CATEGORY_COLOR = "大盘品类分支-颜色"
    HYDC_CATEGORY_CATEGORY = "大盘品类分支-品类趋势"
    HYDC_CATEGORY_TOP_GOODS = "大盘品类分支-Top商品"
    HYDC_CATEGORY_AGGREGATE = "大盘品类分支-聚合"
    HYDC_CATEGORY_REPORT = "大盘品类分支-报告生成"

    # ========== 大盘属性分析 (table_type=2, dimension=0) ==========
    HYDC_PROPERTY_START = "大盘属性分支-开始"
    HYDC_PROPERTY_EXTRACT = "大盘属性分支-属性提取"
    HYDC_PROPERTY_OVERVIEW = "大盘属性分支-概览趋势"
    HYDC_PROPERTY_DATA = "大盘属性分支-属性数据"
    HYDC_PROPERTY_TOP_GOODS = "大盘属性分支-Top商品"
    HYDC_PROPERTY_AGGREGATE = "大盘属性分支-聚合"

    # ========== 大盘颜色分析 (table_type=2, dimension=1) ==========
    HYDC_COLOR_START = "大盘颜色分支-开始"
    HYDC_COLOR_OVERVIEW = "大盘颜色分支-概览趋势"
    HYDC_COLOR_DATA = "大盘颜色分支-颜色数据"
    HYDC_COLOR_TOP_GOODS = "大盘颜色分支-Top商品"
    HYDC_COLOR_AGGREGATE = "大盘颜色分支-聚合"

    # ========== 大盘品牌分析 (table_type=2, dimension=2) ==========
    HYDC_BRAND_START = "大盘品牌分支-开始"
    HYDC_BRAND_OVERVIEW = "大盘品牌分支-概览趋势"
    HYDC_BRAND_DATA = "大盘品牌分支-品牌数据"
    HYDC_BRAND_TOP_GOODS = "大盘品牌分支-Top商品"
    HYDC_BRAND_AGGREGATE = "大盘品牌分支-聚合"

    # ========== 大盘价格带分析 (table_type=3) ==========
    HYDC_PRICE_START = "大盘价格带分支-开始"
    HYDC_EXTRACT_PRICE_BAND = "大盘价格带分支-抽取价格带"
    HYDC_PRICE_OVERVIEW = "大盘价格带分支-概览趋势"
    HYDC_PRICE_DATA = "大盘价格带分支-价格带数据"
    HYDC_PRICE_TOP_GOODS = "大盘价格带分支-Top商品"
    HYDC_PRICE_AGGREGATE = "大盘价格带分支-聚合"

    # ========== 大盘报告生成 ==========
    HYDC_NORMAL_REPORT = "大盘通用分支-报告生成"

    # 大盘-非深度思考报告生成
    HYDC_CATEGORY_NONTHINKING_REPORT = "大盘品类分支-非思考报告生成"
    HYDC_NORMAL_NONTHINKING_REPORT = "大盘通用分支-非思考报告生成"

    ASSEMBLE_OUTPUT = "组装输出数据"


class ZhiyiDeepresearchGraph(BaseWorkflowGraph):
    """
    数据洞察-知衣数据源-深度思考工作流
    """
    span_name = "趋势洞察知衣工作流"
    run_name = "zhiyi-deepresearch-graph"

    def __init__(self):
        super().__init__()
        self.compiled_graph = self._build_graph()
        self.product_area_match_prog = re.compile(r'<custom-productcards>(.*?)</custom-productcards>', re.S)

    def _build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(ZhiyiDeepresearchState)
        # ===定义节点===
        graph.add_node(NodeKey.INIT, self._init_state_node)
        graph.add_node(NodeKey.CATEGORY_SEARCH, self._category_search_node)
        graph.add_node(NodeKey.API_PARAM_PARSE, self._api_param_parse_node)
        graph.add_node(NodeKey.SHOP_SEARCH, self._shop_search_node)

        # 品类分析-店铺
        graph.add_node(NodeKey.CATEGORY_SHOP_START, self._query_shop_analyze_start)
        graph.add_node(NodeKey.CATEGORY_SHOP_OVERVIEW, self._query_shop_analyze_overview)
        graph.add_node(NodeKey.CATEGORY_SHOP_PRICE, self._query_shop_analyze_price)
        graph.add_node(NodeKey.CATEGORY_SHOP_COLOR, self._query_shop_analyze_color)
        graph.add_node(NodeKey.CATEGORY_SHOP_CATEGORY, self._query_shop_analyze_category)
        graph.add_node(NodeKey.CATEGORY_SHOP_TOP_GOODS, self._query_shop_analyze_top_goods)
        graph.add_node(NodeKey.CATEGORY_SHOP_AGGREGATE, self._assemble_shop_analyze_result)

        # 属性分析-店铺
        graph.add_node(NodeKey.PROPERTY_SHOP_START, self._query_shop_analyze_start)
        graph.add_node(NodeKey.MATCH_PROPERTY, self._query_top_property)
        graph.add_node(NodeKey.PROPERTY_SHOP_OVERVIEW, self._query_shop_analyze_overview)
        graph.add_node(NodeKey.PROPERTY_SHOP_PROPERTY, self._query_shop_analyze_property)
        graph.add_node(NodeKey.PROPERTY_SHOP_TOP_GOODS, self._query_shop_analyze_top_goods)
        graph.add_node(NodeKey.PROPERTY_SHOP_AGGREGATE, self._assemble_shop_analyze_result)

        # 价格带分析-店铺
        graph.add_node(NodeKey.PRICE_SHOP_START, self._query_shop_analyze_start)
        graph.add_node(NodeKey.SHOP_EXTRACT_PRICE_BAND, self._extract_price_band)
        graph.add_node(NodeKey.PRICE_SHOP_OVERVIEW, self._query_shop_analyze_overview)
        graph.add_node(NodeKey.PRICE_SHOP_PRICE, self._query_shop_analyze_price)
        graph.add_node(NodeKey.PRICE_SHOP_TOP_GOODS, self._query_shop_analyze_top_goods)
        graph.add_node(NodeKey.PRICE_SHOP_AGGREGATE, self._assemble_shop_analyze_result)

        # 店铺报告生成
        graph.add_node(NodeKey.CATEGORY_SHOP_REPORT, self._generate_shop_category_analyze_report)
        graph.add_node(NodeKey.NORMAL_SHOP_REPORT, self._generate_shop_normal_analyze_report)

        # 店铺非深度思考报告生成
        graph.add_node(NodeKey.CATEGORY_SHOP_NONTHINKING_REPORT, self._generate_shop_category_nonthinking_report)
        graph.add_node(NodeKey.NORMAL_SHOP_NONTHINKING_REPORT, self._generate_shop_normal_nonthinking_report)

        # ===== 大盘分析节点 =====
        graph.add_node(NodeKey.HYDC_TABLE_TYPE_ROUTER, self._hydc_table_type_router_node)

        # 大盘品类分析 (table_type=1)
        graph.add_node(NodeKey.HYDC_CATEGORY_START, self._query_hydc_analyze_start)
        graph.add_node(NodeKey.HYDC_CATEGORY_OVERVIEW, self._query_hydc_category_overview)
        graph.add_node(NodeKey.HYDC_CATEGORY_PRICE, self._query_hydc_category_price)
        graph.add_node(NodeKey.HYDC_CATEGORY_COLOR, self._query_hydc_category_color)
        graph.add_node(NodeKey.HYDC_CATEGORY_CATEGORY, self._query_hydc_category_category)
        graph.add_node(NodeKey.HYDC_CATEGORY_TOP_GOODS, self._query_hydc_category_top_goods)
        graph.add_node(NodeKey.HYDC_CATEGORY_AGGREGATE, self._assemble_hydc_category_result)
        graph.add_node(NodeKey.HYDC_CATEGORY_REPORT, self._generate_hydc_category_report)

        # 大盘维度分析路由 (table_type=2)
        graph.add_node(NodeKey.HYDC_DIMENSION_ANALYZE, self._analyze_hydc_dimension)

        # 大盘属性分析 (table_type=2, dimension=0)
        graph.add_node(NodeKey.HYDC_PROPERTY_START, self._query_hydc_property_start)
        graph.add_node(NodeKey.HYDC_PROPERTY_EXTRACT, self._extract_hydc_property)
        graph.add_node(NodeKey.HYDC_PROPERTY_OVERVIEW, self._query_hydc_property_overview)
        graph.add_node(NodeKey.HYDC_PROPERTY_DATA, self._query_hydc_property_data)
        graph.add_node(NodeKey.HYDC_PROPERTY_TOP_GOODS, self._query_hydc_property_top_goods)
        graph.add_node(NodeKey.HYDC_PROPERTY_AGGREGATE, self._assemble_hydc_property_result)

        # 大盘颜色分析 (table_type=2, dimension=1)
        graph.add_node(NodeKey.HYDC_COLOR_START, self._query_hydc_color_start)
        graph.add_node(NodeKey.HYDC_COLOR_OVERVIEW, self._query_hydc_color_overview)
        graph.add_node(NodeKey.HYDC_COLOR_DATA, self._query_hydc_color_data)
        graph.add_node(NodeKey.HYDC_COLOR_TOP_GOODS, self._query_hydc_color_top_goods)
        graph.add_node(NodeKey.HYDC_COLOR_AGGREGATE, self._assemble_hydc_color_result)

        # 大盘品牌分析 (table_type=2, dimension=2)
        graph.add_node(NodeKey.HYDC_BRAND_START, self._query_hydc_brand_start)
        graph.add_node(NodeKey.HYDC_BRAND_OVERVIEW, self._query_hydc_brand_overview)
        graph.add_node(NodeKey.HYDC_BRAND_DATA, self._query_hydc_brand_data)
        graph.add_node(NodeKey.HYDC_BRAND_TOP_GOODS, self._query_hydc_brand_top_goods)
        graph.add_node(NodeKey.HYDC_BRAND_AGGREGATE, self._assemble_hydc_brand_result)

        # 大盘价格带分析 (table_type=3)
        graph.add_node(NodeKey.HYDC_PRICE_START, self._query_hydc_price_start)
        graph.add_node(NodeKey.HYDC_EXTRACT_PRICE_BAND, self._extract_price_band)
        graph.add_node(NodeKey.HYDC_PRICE_OVERVIEW, self._query_hydc_price_overview)
        graph.add_node(NodeKey.HYDC_PRICE_DATA, self._query_hydc_price_data)
        graph.add_node(NodeKey.HYDC_PRICE_TOP_GOODS, self._query_hydc_price_top_goods)
        graph.add_node(NodeKey.HYDC_PRICE_AGGREGATE, self._assemble_hydc_price_result)

        # 大盘报告生成
        graph.add_node(NodeKey.HYDC_NORMAL_REPORT, self._generate_hydc_normal_report)

        # 大盘非深度思考报告生成
        graph.add_node(NodeKey.HYDC_CATEGORY_NONTHINKING_REPORT, self._generate_hydc_category_nonthinking_report)
        graph.add_node(NodeKey.HYDC_NORMAL_NONTHINKING_REPORT, self._generate_hydc_normal_nonthinking_report)

        graph.add_node(NodeKey.ASSEMBLE_OUTPUT, self._assemble_output_response)

        # ===定义入口和边===
        graph.set_entry_point(NodeKey.INIT)
        graph.add_edge(NodeKey.INIT, NodeKey.CATEGORY_SEARCH)
        graph.add_edge(NodeKey.CATEGORY_SEARCH, NodeKey.API_PARAM_PARSE)
        graph.add_conditional_edges(NodeKey.API_PARAM_PARSE, self._route_if_shop_query_node, {
            True: NodeKey.SHOP_SEARCH, False: NodeKey.HYDC_TABLE_TYPE_ROUTER
        })
        graph.add_conditional_edges(NodeKey.SHOP_SEARCH, self._route_table_type_node, {
            "1": NodeKey.CATEGORY_SHOP_START,
            "2": NodeKey.PROPERTY_SHOP_START,
            "3": NodeKey.PRICE_SHOP_START,
        })

        # 店铺 -> 品类分析
        graph.add_edge(NodeKey.CATEGORY_SHOP_START, NodeKey.CATEGORY_SHOP_OVERVIEW)
        graph.add_edge(NodeKey.CATEGORY_SHOP_START, NodeKey.CATEGORY_SHOP_CATEGORY)
        graph.add_edge(NodeKey.CATEGORY_SHOP_START, NodeKey.CATEGORY_SHOP_PRICE)
        graph.add_edge(NodeKey.CATEGORY_SHOP_START, NodeKey.CATEGORY_SHOP_COLOR)
        graph.add_edge(NodeKey.CATEGORY_SHOP_START, NodeKey.CATEGORY_SHOP_TOP_GOODS)
        graph.add_edge(
            [NodeKey.CATEGORY_SHOP_OVERVIEW,
            NodeKey.CATEGORY_SHOP_CATEGORY,
            NodeKey.CATEGORY_SHOP_PRICE,
            NodeKey.CATEGORY_SHOP_COLOR,
            NodeKey.CATEGORY_SHOP_TOP_GOODS],
            NodeKey.CATEGORY_SHOP_AGGREGATE
        )
        graph.add_conditional_edges(NodeKey.CATEGORY_SHOP_AGGREGATE, self._route_if_thinking_report, {
            True: NodeKey.CATEGORY_SHOP_REPORT,
            False: NodeKey.CATEGORY_SHOP_NONTHINKING_REPORT
        })
        graph.add_edge(NodeKey.CATEGORY_SHOP_REPORT, NodeKey.ASSEMBLE_OUTPUT)
        graph.add_edge(NodeKey.CATEGORY_SHOP_NONTHINKING_REPORT, NodeKey.ASSEMBLE_OUTPUT)

        # 店铺 -> 属性分析
        graph.add_edge(NodeKey.PROPERTY_SHOP_START, NodeKey.MATCH_PROPERTY)
        graph.add_edge(NodeKey.MATCH_PROPERTY, NodeKey.PROPERTY_SHOP_OVERVIEW)
        graph.add_edge(NodeKey.MATCH_PROPERTY, NodeKey.PROPERTY_SHOP_PROPERTY)
        graph.add_edge(NodeKey.MATCH_PROPERTY, NodeKey.PROPERTY_SHOP_TOP_GOODS)
        graph.add_edge(
            [NodeKey.PROPERTY_SHOP_OVERVIEW,
             NodeKey.PROPERTY_SHOP_PROPERTY,
             NodeKey.PROPERTY_SHOP_TOP_GOODS],
            NodeKey.PROPERTY_SHOP_AGGREGATE
        )
        graph.add_conditional_edges(NodeKey.PROPERTY_SHOP_AGGREGATE, self._route_if_thinking_report, {
            True: NodeKey.NORMAL_SHOP_REPORT,
            False: NodeKey.NORMAL_SHOP_NONTHINKING_REPORT
        })
        graph.add_edge(NodeKey.NORMAL_SHOP_REPORT, NodeKey.ASSEMBLE_OUTPUT)
        graph.add_edge(NodeKey.NORMAL_SHOP_NONTHINKING_REPORT, NodeKey.ASSEMBLE_OUTPUT)

        # 店铺 -> 价格带分析
        graph.add_edge(NodeKey.PRICE_SHOP_START, NodeKey.SHOP_EXTRACT_PRICE_BAND)
        graph.add_edge(NodeKey.SHOP_EXTRACT_PRICE_BAND, NodeKey.PRICE_SHOP_OVERVIEW)
        graph.add_edge(NodeKey.SHOP_EXTRACT_PRICE_BAND, NodeKey.PRICE_SHOP_PRICE)
        graph.add_edge(NodeKey.SHOP_EXTRACT_PRICE_BAND, NodeKey.PRICE_SHOP_TOP_GOODS)
        graph.add_edge(
            [NodeKey.PRICE_SHOP_OVERVIEW,
            NodeKey.PRICE_SHOP_PRICE,
            NodeKey.PRICE_SHOP_TOP_GOODS],
            NodeKey.PRICE_SHOP_AGGREGATE
        )
        graph.add_conditional_edges(NodeKey.PRICE_SHOP_AGGREGATE, self._route_if_thinking_report, {
            True: NodeKey.NORMAL_SHOP_REPORT,
            False: NodeKey.NORMAL_SHOP_NONTHINKING_REPORT
        })
        # 注意：NORMAL_SHOP_REPORT到ASSEMBLE_OUTPUT的边已在步骤4.2中添加
        # NORMAL_SHOP_NONTHINKING_REPORT到ASSEMBLE_OUTPUT的边也已在步骤4.2中添加

        # ===== 大盘分析边 =====
        # 大盘 table_type 路由
        graph.add_conditional_edges(NodeKey.HYDC_TABLE_TYPE_ROUTER, self._route_hydc_table_type_node, {
            "1": NodeKey.HYDC_CATEGORY_START,
            "2": NodeKey.HYDC_DIMENSION_ANALYZE,
            "3": NodeKey.HYDC_PRICE_START,
        })

        # 大盘 -> 品类分析 (table_type=1)
        graph.add_edge(NodeKey.HYDC_CATEGORY_START, NodeKey.HYDC_CATEGORY_OVERVIEW)
        graph.add_edge(NodeKey.HYDC_CATEGORY_START, NodeKey.HYDC_CATEGORY_PRICE)
        graph.add_edge(NodeKey.HYDC_CATEGORY_START, NodeKey.HYDC_CATEGORY_COLOR)
        graph.add_edge(NodeKey.HYDC_CATEGORY_START, NodeKey.HYDC_CATEGORY_CATEGORY)
        graph.add_edge(NodeKey.HYDC_CATEGORY_START, NodeKey.HYDC_CATEGORY_TOP_GOODS)
        graph.add_edge(
            [NodeKey.HYDC_CATEGORY_OVERVIEW,
             NodeKey.HYDC_CATEGORY_PRICE,
             NodeKey.HYDC_CATEGORY_COLOR,
             NodeKey.HYDC_CATEGORY_CATEGORY,
             NodeKey.HYDC_CATEGORY_TOP_GOODS],
            NodeKey.HYDC_CATEGORY_AGGREGATE
        )
        graph.add_conditional_edges(NodeKey.HYDC_CATEGORY_AGGREGATE,
                                     self._route_if_thinking_report, {
            True: NodeKey.HYDC_CATEGORY_REPORT,
            False: NodeKey.HYDC_CATEGORY_NONTHINKING_REPORT
        })
        graph.add_edge(NodeKey.HYDC_CATEGORY_REPORT, NodeKey.ASSEMBLE_OUTPUT)
        graph.add_edge(NodeKey.HYDC_CATEGORY_NONTHINKING_REPORT, NodeKey.ASSEMBLE_OUTPUT)

        # 大盘 -> 维度分析路由 (table_type=2)
        graph.add_conditional_edges(NodeKey.HYDC_DIMENSION_ANALYZE, self._route_hydc_dimension_node, {
            0: NodeKey.HYDC_PROPERTY_START,  # 属性
            1: NodeKey.HYDC_COLOR_START,     # 颜色
            2: NodeKey.HYDC_BRAND_START,     # 品牌
        })

        # 大盘 -> 属性分析 (dimension=0)
        graph.add_edge(NodeKey.HYDC_PROPERTY_START, NodeKey.HYDC_PROPERTY_EXTRACT)
        graph.add_edge(NodeKey.HYDC_PROPERTY_EXTRACT, NodeKey.HYDC_PROPERTY_OVERVIEW)
        graph.add_edge(NodeKey.HYDC_PROPERTY_EXTRACT, NodeKey.HYDC_PROPERTY_DATA)
        graph.add_edge(NodeKey.HYDC_PROPERTY_EXTRACT, NodeKey.HYDC_PROPERTY_TOP_GOODS)
        graph.add_edge(
            [NodeKey.HYDC_PROPERTY_OVERVIEW,
             NodeKey.HYDC_PROPERTY_DATA,
             NodeKey.HYDC_PROPERTY_TOP_GOODS],
            NodeKey.HYDC_PROPERTY_AGGREGATE
        )
        graph.add_conditional_edges(NodeKey.HYDC_PROPERTY_AGGREGATE, self._route_if_thinking_report, {
            True: NodeKey.HYDC_NORMAL_REPORT,
            False: NodeKey.HYDC_NORMAL_NONTHINKING_REPORT
        })

        # 大盘 -> 颜色分析 (dimension=1)
        graph.add_edge(NodeKey.HYDC_COLOR_START, NodeKey.HYDC_COLOR_OVERVIEW)
        graph.add_edge(NodeKey.HYDC_COLOR_START, NodeKey.HYDC_COLOR_DATA)
        graph.add_edge(NodeKey.HYDC_COLOR_START, NodeKey.HYDC_COLOR_TOP_GOODS)
        graph.add_edge(
            [NodeKey.HYDC_COLOR_OVERVIEW,
             NodeKey.HYDC_COLOR_DATA,
             NodeKey.HYDC_COLOR_TOP_GOODS],
            NodeKey.HYDC_COLOR_AGGREGATE
        )
        graph.add_conditional_edges(NodeKey.HYDC_COLOR_AGGREGATE, self._route_if_thinking_report, {
            True: NodeKey.HYDC_NORMAL_REPORT,
            False: NodeKey.HYDC_NORMAL_NONTHINKING_REPORT
        })

        # 大盘 -> 品牌分析 (dimension=2)
        graph.add_edge(NodeKey.HYDC_BRAND_START, NodeKey.HYDC_BRAND_OVERVIEW)
        graph.add_edge(NodeKey.HYDC_BRAND_START, NodeKey.HYDC_BRAND_DATA)
        graph.add_edge(NodeKey.HYDC_BRAND_START, NodeKey.HYDC_BRAND_TOP_GOODS)
        graph.add_edge(
            [NodeKey.HYDC_BRAND_OVERVIEW,
             NodeKey.HYDC_BRAND_DATA,
             NodeKey.HYDC_BRAND_TOP_GOODS],
            NodeKey.HYDC_BRAND_AGGREGATE
        )
        graph.add_conditional_edges(NodeKey.HYDC_BRAND_AGGREGATE, self._route_if_thinking_report, {
            True: NodeKey.HYDC_NORMAL_REPORT,
            False: NodeKey.HYDC_NORMAL_NONTHINKING_REPORT
        })

        # 大盘 -> 价格带分析 (table_type=3)
        graph.add_edge(NodeKey.HYDC_PRICE_START, NodeKey.HYDC_EXTRACT_PRICE_BAND)
        graph.add_edge(NodeKey.HYDC_EXTRACT_PRICE_BAND, NodeKey.HYDC_PRICE_OVERVIEW)
        graph.add_edge(NodeKey.HYDC_EXTRACT_PRICE_BAND, NodeKey.HYDC_PRICE_DATA)
        graph.add_edge(NodeKey.HYDC_EXTRACT_PRICE_BAND, NodeKey.HYDC_PRICE_TOP_GOODS)
        graph.add_edge(
            [NodeKey.HYDC_PRICE_OVERVIEW,
             NodeKey.HYDC_PRICE_DATA,
             NodeKey.HYDC_PRICE_TOP_GOODS],
            NodeKey.HYDC_PRICE_AGGREGATE
        )
        graph.add_conditional_edges(NodeKey.HYDC_PRICE_AGGREGATE, self._route_if_thinking_report, {
            True: NodeKey.HYDC_NORMAL_REPORT,
            False: NodeKey.HYDC_NORMAL_NONTHINKING_REPORT
        })

        # 大盘通用报告 -> 输出
        graph.add_edge(NodeKey.HYDC_NORMAL_REPORT, NodeKey.ASSEMBLE_OUTPUT)
        # 大盘非思考报告 -> 输出
        graph.add_edge(NodeKey.HYDC_NORMAL_NONTHINKING_REPORT, NodeKey.ASSEMBLE_OUTPUT)

        graph.set_finish_point(NodeKey.ASSEMBLE_OUTPUT)

        return graph.compile()

    def get_compiled_graph(self) -> CompiledStateGraph:
        return self.compiled_graph


    # ===执行节点的定义===
    def _init_state_node(self, state: ZhiyiDeepresearchState):
        req = state.request
        is_thinking = req.thinking

        # 创建消息推送器
        pusher = DeepresearchMessagePusher(
            request=req,
            is_thinking=is_thinking,
            workflow_type="zhiyi",
        )

        pusher.push_task_start_msg()

        # 非深度思考的推送
        if not is_thinking:
            pusher.push_phase("正在启用趋势分析助手")

        return {
            "is_thinking": is_thinking,
            "message_pusher": pusher
        }

    def _category_search_node(self, state: ZhiyiDeepresearchState):
        """检索用户提到的行业品类"""
        req = state.request
        pusher = state.message_pusher

        # 向量搜索
        kb_client = get_volcengine_kb_api()
        resp = kb_client.simple_chat(query=req.user_query, service_resource_id=VolcKnowledgeServiceId.ZHIYI_CATEGORY_VECTOR)
        parse_content_list: list[dict] = kb_client.parse_structure_chat_response(resp)

        parsed_category_list = []
        for content in parse_content_list:
            category_name_path: str = content['key'] # format: '女装,风衣'
            category_id_path: str = content['value'] # format: '16,50008901'
            category_id_arr = category_id_path.split(',')
            category_name_arr = category_name_path.split(',')
            root_category_id = category_id_arr[0] if len(category_id_arr) >= 1 else None
            leaf_category_id = category_id_arr[1] if len(category_id_arr) >= 2 else None
            root_category_name = category_name_arr[0] if len(category_name_arr) >= 1 else None
            leaf_category_name = category_name_arr[1] if len(category_name_arr) >= 2 else None

            item = ZhiyiCategoryFormatItem(
                root_category_id=root_category_id,
                category_id=leaf_category_id,
                root_category_id_name=root_category_name,
                category_name=leaf_category_name,
            )
            parsed_category_list.append(item)

        return {
            "recall_category": ZhiyiParsedCategory(category_list=parsed_category_list)
        }

    def _api_param_parse_node(self, state: ZhiyiDeepresearchState):
        """llm解析用户问题中的查询相关的参数"""
        req = state.request
        recall_category = state.recall_category
        is_thinking = state.is_thinking
        pusher = state.message_pusher

        # 获取llm
        llm = llm_factory.get_llm(
            LlmProvider.OPENROUTER.name,
            LlmModelName.OPENROUTER_GEMINI_3_FLASH_PREVIEW.value,
        )
        struct_llm = llm.with_structured_output(schema=ZhiyiThinkingApiParseParam).with_retry(stop_after_attempt=2)
        # 获取prompt
        # 计算日期相关参数
        today = datetime.now()
        t_minus_1 = today - timedelta(days=1)  # 当前日期的前一天
        last_month_end = today.replace(day=1) - timedelta(days=1)  # 本月的前一个月的最后一天
        six_months_ago_first = (today - relativedelta(months=6)).replace(day=1)  # 六个月前的第一天
        recent_30day_start = t_minus_1 - timedelta(days=29)  # 近30天开始日期

        invoke_params = {
            # 时间相关
            "current_date": t_minus_1.strftime("%Y-%m-%d"),  # 当前日期（T-1）
            "default_end_date": last_month_end.strftime("%Y-%m-%d"),  # 默认结束日期：本月的前一个月的最后一天
            "default_start_date": six_months_ago_first.strftime("%Y-%m-%d"),  # 默认开始日期：六个月前的第一天
            "recent_30day_start_date": recent_30day_start.strftime("%Y-%m-%d"),  # 近一个月开始日期
            # 业务参数
            "user_query": req.user_query,
            "industry": req.industry,
            "category_list": recall_category.model_dump_json(ensure_ascii=False),
        }
        messages: list[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.DR_ZHIYI_THINKING_PARAM_PARSE_PROMPT.value,
            variables=invoke_params,
        )

        # 执行llm
        parse_param: ZhiyiThinkingApiParseParam = struct_llm.with_config(run_name="解析api参数").invoke(messages)

        # 深度思考的消息因为要区分店铺/行业洞察，下放到分支中发送

        return {
            "api_parse_param": parse_param
        }

    def _route_if_shop_query_node(self, state: ZhiyiDeepresearchState):
        """判断是否是店铺查询"""
        param = state.api_parse_param
        return param.is_shop

    def _route_table_type_node(self, state: ZhiyiDeepresearchState) -> Literal["1", "2", "3"]:
        """路由聚合维度 品类/属性/价格等"""
        param = state.api_parse_param
        dimension = param.table_type
        if not dimension:
            return "1"
        else:
            return dimension

    def _route_if_thinking_report(self, state: ZhiyiDeepresearchState) -> bool:
        """判断是否深度思考模式"""
        return state.is_thinking


    def _shop_search_node(self, state: ZhiyiDeepresearchState):
        """向量召回店铺,并通过llm选出唯一目标店铺"""
        req = state.request
        is_thinking = state.is_thinking
        parse_param = state.api_parse_param
        pusher = state.message_pusher
        llm_shop_name: str = parse_param.shop_name

        # 使用火山召回相似店铺
        kb_client = get_volcengine_kb_api()
        resp = kb_client.simple_chat(query=llm_shop_name, service_resource_id=VolcKnowledgeServiceId.ZHIYI_SHOP_KNOWLEDGE.value)
        recall_list = kb_client.parse_structure_chat_response(resp)

        concat_shop_list = []
        for item in recall_list:
            shop_name = item['key']
            shop_id = item['value']
            if shop_name and shop_id:
                concat_shop_list.append(f"{shop_name},{shop_id}")

        # llm选择唯一店铺
        llm = llm_factory.get_llm(LlmProvider.HUANXIN.name, LlmModelName.HUANXIN_GEMINI_2_5_FLASH.value)
        struct_llm = llm.with_structured_output(ShopCleanResult).with_retry(stop_after_attempt=2)
        invoke_params = {
            "origin_text": llm_shop_name,
            "recall_list": json.dumps(concat_shop_list, ensure_ascii=False),
        }
        message = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.DR_ZHIYI_THINKING_SHOP_CLEAN_PROMPT.value,
            variables=invoke_params,
        )
        clean_result: ShopCleanResult = struct_llm.with_config(run_name="向量召回店铺清洗").invoke(message)

        clean_shop_list = clean_result.content_list
        if clean_shop_list:
            shop_arr = clean_shop_list[0].split(',')
            shop_name = shop_arr[0]
            shop_id = int(shop_arr[1])
        else:
            logger.error(f"[知衣数据洞察]向量召回店铺清洗后为空,session_id: {req.session_id}")
            shop_name = ""
            shop_id = ""

        # 推送店铺分支的深度思考消息
        if is_thinking:
            pusher.push_phase("数据收集", variables={
                "module_name": "店铺分析",
                "module_link": f"https://data.zhiyitech.cn/shopDetail/analysis?shopId={shop_id}",
                "industry_name": parse_param.root_category_id_name
            })

        return {
            "target_shop_name": shop_name,
            "target_shop_id": shop_id,
        }

    # === 维度专属的数据提取节点 ===
    def _parse_properties_from_query(self, state: ZhiyiDeepresearchState):
        pass

    def _parse_price_range_from_query(self, state: ZhiyiDeepresearchState):
        pass

    # === 辅助方法 ===
    def _get_granularity(self, date_type: str) -> int:
        """根据日期类型获取粒度值: week=2, month=3"""
        return 2 if date_type == "week" else 3

    def _safe_int(self, value: str | None, default: int = 0) -> int:
        """安全转换字符串为整数"""
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    # === 数据清洗方法 ===
    def _clean_overview_data(self, raw_data: SaleTrendRawResponse) -> SaleTrendCleanResponse:
        """清洗概览数据 - 对应n8n Code in JavaScript5"""
        result = raw_data.result

        slim_trend = [
            SaleTrendSlimItem(
                granularityDate=t.granularity_date,
                insertDate=t.insert_date,
                startTime=t.start_time,
                endTime=t.end_time,
                dailySumDaySalesVolume=t.daily_sum_day_sales_volume or 0,
                dailySumDaySale=t.daily_sum_day_sale or 0,
                dailyShelvesCount=t.daily_shelves_count or 0,
                dailyNewItemSalesVolume=t.daily_new_item_sales_volume or 0,
            )
            for t in result.trend_dtos
        ]

        return SaleTrendCleanResponse(
            success=raw_data.success,
            result=SaleTrendSlimResult(
                sumSalesVolume=result.sum_sales_volume,
                sumSale=result.sum_sale,
                allSalesVolume=result.all_sales_volume,
                sumShelvesCount=result.sum_shelves_count,
                sumNewItemSalesVolume=result.sum_new_item_sales_volume,
                onsaleSkuNum=result.onsale_sku_num,
                onsaleStock=result.onsale_stock,
                newSkuNum=result.new_sku_num,
                top3CategoryList=result.top3_category_list,
                hasPromotion=result.has_promotion,
                hasShuang11Presale=result.has_shuang11_presale,
                trendDTOS=slim_trend,
            )
        )

    def _clean_price_range_data(self, raw_data: PriceRangeRawResponse) -> PriceRangeCleanResponse:
        """清洗价格带数据 - 对应n8n Code in JavaScript4"""
        price_slim = [
            PriceRangeSlimItem(
                leftPrice=it.left_price,
                rightPrice=it.right_price,
                salesVolume=it.sum_day_sales_volume_by_range or 0,
                rate=it.sum_day_sales_volume_by_range_rate,
            )
            for it in raw_data.result
            if it.left_price is not None and it.sum_day_sales_volume_by_range is not None
        ]
        price_slim.sort(key=lambda x: x.left_price)

        return PriceRangeCleanResponse(
            success=raw_data.success,
            result=PriceRangeSlimResult(price_slim=price_slim)
        )

    def _clean_color_data(self, raw_data: PropertyTrendRawResponse) -> ColorCleanResponse:
        """清洗颜色/属性数据 - 对应n8n Code in JavaScript3"""
        color_slim = [
            ColorSlimItem(
                propertyValue=it.property_value,
                salesVolume=it.sum_day_sales_volume_by_property or 0,
                salesAmount=it.sum_day_sale_by_property,
                rate=it.sum_day_sales_volume_by_property_rate,
                otherFlag=it.other_flag,
            )
            for it in raw_data.result
        ]

        return ColorCleanResponse(
            success=raw_data.success,
            result=ColorSlimResult(color_slim=color_slim)
        )

    def _clean_top10_items_data(self, raw_data: ShopHotItemRawResponse) -> Top10ItemsCleanResponse:
        """清洗Top10商品数据 - 对应n8n Code in JavaScript2"""
        result = raw_data.result if raw_data.result else []

        top10_slim = [
            Top10ItemSlim(
                itemId=str(it.item_id) if it.item_id else None,
                title=it.title,
                picUrl=it.pic_url,
                categoryName=it.category_name,
                categoryDetail=it.category_detail,
                salesVolume=it.total_sale_volume or it.agg_sale_volume or 0,
                salesAmount=it.total_sale_amount or it.agg_sale_amount or 0,
                minPrice=int(it.min_price) if it.min_price and it.min_price.isdigit() else None,
                maxSPrice=it.max_s_price,
            )
            for idx, it in enumerate(result.result_list[:10])
        ]

        return Top10ItemsCleanResponse(
            success=raw_data.success,
            result=Top10ItemsSlimResult(
                start=result.start,
                pageSize=result.page_size,
                resultCount=result.result_count,
                top10_slim=top10_slim,
            )
        )

    def _clean_category_trend_data(self, raw_data: CategoryTrendRawResponse) -> CategoryTrendCleanResponse:
        """清洗品类趋势数据 - 对应n8n Code in JavaScript6"""
        result_list = raw_data.result if raw_data.result else []

        def pick_trend(t) -> dict:
            return {
                "week": t.granularity_date,
                "range": t.insert_date or t.insert_time,
                "startTime": t.start_time,
                "endTime": t.end_time,
                "salesVolume": t.daily_sum_day_sales_volume or 0,
                "salesCount": t.daily_all_sales_volume,
                "salesAmount": t.daily_sum_day_sale,
                "newItemSalesVolume": t.daily_new_item_sales_volume or 0,
                "shelvesCount": t.daily_shelves_count,
                "percentage": t.percentage,
                "onsaleSkuNum": t.onsale_sku_num,
                "onsaleStock": t.onsale_stock,
                "newSkuNum": t.new_sku_num,
            }

        top10_slim = []
        for idx, it in enumerate(result_list[:10]):
            first_name = it.first_category_name or ""
            second_name = it.second_category_name or ""
            category_path = None
            if first_name or second_name:
                category_path = f"{first_name}{' > ' if first_name and second_name else ''}{second_name}"

            top10_slim.append(
                CategoryTrendSlimItem(
                    rank=idx + 1,
                    categoryId=str(it.category_id) if it.category_id else None,
                    firstCategoryName=first_name,
                    secondCategoryName=second_name,
                    categoryPath=category_path,
                    allSalesVolume=it.all_sales_volume_by_category or 0,
                    sumDaySalesVolume=it.sum_day_sales_volume_by_category or 0,
                    sumDaySalesAmount=it.sum_day_sale_by_category or 0,
                    newItemSalesVolume=it.new_item_sales_volume_by_category or 0,
                    shelvesCount=it.shelves_count_by_category,
                    allSalesVolumeRate=it.all_sales_volume_by_category_rate,
                    sumDaySalesVolumeRate=it.sum_day_sales_volume_by_category_rate,
                    sumDaySalesAmountRate=it.sum_day_sale_by_category_rate,
                    onsaleSkuNum=it.onsale_sku_num,
                    onsaleStock=it.onsale_stock,
                    newSkuNum=it.new_sku_num,
                    trend=[pick_trend(t) for t in it.trend_dtos],
                )
            )

        return CategoryTrendCleanResponse(
            success=raw_data.success,
            result=CategoryTrendSlimResult(
                resultCount=len(result_list),
                top10_slim=top10_slim,
            )
        )

    # === 大盘分析(HYDC)数据清洗方法 - 处理 /hydc/trend-list 返回的数据 ===

    def _clean_hydc_overview_data(self, raw_data: HydcTrendListRawResponse) -> SaleTrendCleanResponse:
        """清洗大盘概览数据 - groupType=overview 返回的数据

        /hydc/trend-list 接口返回的 overview 数据结构与旧接口不同，
        这里做适配处理，返回兼容的 SaleTrendCleanResponse
        """
        # 适配 trend_dtos 数据：从第一个结果项的 trendList 获取时序数据
        slim_trend = []
        total_sale_volume = 0
        total_sale_amount = 0

        for item in raw_data.result:
            for trend in item.trend_list:
                sale_volume = self._safe_int(trend.sale_volume)
                sale_amount = self._safe_int(trend.sale_amount)
                total_sale_volume += sale_volume
                total_sale_amount += sale_amount
                slim_trend.append(SaleTrendSlimItem(
                    granularityDate=trend.agg_date,
                    dailySumDaySalesVolume=sale_volume,
                    dailySumDaySale=sale_amount,
                    dailyShelvesCount=0,
                    dailyNewItemSalesVolume=self._safe_int(trend.new_sale_num),
                ))

        return SaleTrendCleanResponse(
            success=raw_data.success,
            result=SaleTrendSlimResult(
                sumSalesVolume=total_sale_volume,
                sumSale=total_sale_amount,
                allSalesVolume=total_sale_volume,
                sumShelvesCount=None,
                sumNewItemSalesVolume=None,
                onsaleSkuNum=None,
                onsaleStock=None,
                newSkuNum=None,
                top3CategoryList=None,
                hasPromotion=None,
                hasShuang11Presale=None,
                trendDTOS=slim_trend,
            )
        )

    def _clean_hydc_price_data(self, raw_data: HydcTrendListRawResponse) -> PriceRangeCleanResponse:
        """清洗大盘价格带数据 - groupType=cprice 返回的数据"""
        price_slim = []
        for item in raw_data.result:
            # aggField 格式为 "50以下" 或 "50-100" 或 "500以上" 等
            agg_field = item.agg_field or "0-0"
            if "以下" in agg_field:
                left_price = 0
                num_part = agg_field.replace("以下", "")
                right_price = int(num_part) if num_part.isdigit() else 0
            elif "以上" in agg_field:
                num_part = agg_field.replace("以上", "")
                left_price = int(num_part) if num_part.isdigit() else 0
                right_price = 999999
            elif "-" in agg_field:
                parts = agg_field.split("-")
                left_price = int(parts[0]) if parts[0].isdigit() else 0
                right_price = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            else:
                left_price = 0
                right_price = 0

            # 从 trendList 累计销量
            total_sale_volume = sum(
                self._safe_int(t.sale_volume) for t in item.trend_list
            )

            price_slim.append(PriceRangeSlimItem(
                leftPrice=left_price,
                rightPrice=right_price,
                salesVolume=total_sale_volume,
                rate=item.metric_percent,
            ))
        price_slim.sort(key=lambda x: x.left_price)

        return PriceRangeCleanResponse(
            success=raw_data.success,
            result=PriceRangeSlimResult(price_slim=price_slim)
        )

    def _clean_hydc_color_data(self, raw_data: HydcTrendListRawResponse) -> ColorCleanResponse:
        """清洗大盘颜色数据 - groupType=color 返回的数据"""
        color_slim = []
        for item in raw_data.result:
            # 从 trendList 累计销量和销售额
            total_sale_volume = sum(self._safe_int(t.sale_volume) for t in item.trend_list)
            total_sale_amount = sum(self._safe_int(t.sale_amount) for t in item.trend_list)

            color_slim.append(ColorSlimItem(
                propertyValue=item.agg_name or item.agg_field,
                salesVolume=total_sale_volume,
                salesAmount=total_sale_amount,
                rate=item.metric_percent,
                otherFlag=False,
            ))

        return ColorCleanResponse(
            success=raw_data.success,
            result=ColorSlimResult(color_slim=color_slim)
        )

    def _clean_hydc_property_data(self, raw_data: HydcTrendListRawResponse) -> ColorCleanResponse:
        """清洗大盘属性数据 - groupType=property 返回的数据

        属性数据结构与颜色类似，复用 ColorCleanResponse
        """
        property_slim = []
        for item in raw_data.result:
            # 从 trendList 累计销量和销售额
            total_sale_volume = sum(self._safe_int(t.sale_volume) for t in item.trend_list)
            total_sale_amount = sum(self._safe_int(t.sale_amount) for t in item.trend_list)

            property_slim.append(ColorSlimItem(
                propertyValue=item.agg_name or item.agg_field,
                salesVolume=total_sale_volume,
                salesAmount=total_sale_amount,
                rate=item.metric_percent,
                otherFlag=False,
            ))

        return ColorCleanResponse(
            success=raw_data.success,
            result=ColorSlimResult(color_slim=property_slim)
        )

    def _clean_hydc_brand_data(self, raw_data: HydcTrendListRawResponse) -> BrandCleanResponse:
        """清洗大盘品牌数据 - groupType=brand 返回的数据"""
        brand_slim = []
        for item in raw_data.result:
            # 从 trendList 累计销量和销售额
            total_sale_volume = sum(self._safe_int(t.sale_volume) for t in item.trend_list)
            total_sale_amount = sum(self._safe_int(t.sale_amount) for t in item.trend_list)

            brand_slim.append(BrandSlimItem(
                brandName=item.agg_name or item.agg_field,
                salesVolume=total_sale_volume,
                salesAmount=total_sale_amount,
                rate=item.metric_percent,
                otherFlag=False,
            ))

        return BrandCleanResponse(
            success=raw_data.success,
            result=BrandSlimResult(brand_slim=brand_slim)
        )

    def _clean_hydc_category_data(self, raw_data: HydcTrendListRawResponse) -> CategoryTrendCleanResponse:
        """清洗大盘品类数据 - groupType=overview 返回的数据用于品类趋势

        由于 /hydc/trend-list 的 overview 数据与品类趋势数据结构差异较大，
        这里简化处理，返回基础的品类信息
        """
        top10_slim = []
        for idx, item in enumerate(raw_data.result[:10]):
            # 从 trendList 累计销量和销售额
            total_sale_volume = sum(self._safe_int(t.sale_volume) for t in item.trend_list)
            total_sale_amount = sum(self._safe_int(t.sale_amount) for t in item.trend_list)

            top10_slim.append(CategoryTrendSlimItem(
                rank=idx + 1,
                categoryId=item.agg_key,
                firstCategoryName=item.agg_name or item.agg_field,
                secondCategoryName=None,
                categoryPath=item.agg_name or item.agg_field,
                allSalesVolume=total_sale_volume,
                sumDaySalesVolume=total_sale_volume,
                sumDaySalesAmount=total_sale_amount,
                newItemSalesVolume=0,
                shelvesCount=None,
                allSalesVolumeRate=item.metric_percent,
                sumDaySalesVolumeRate=item.metric_percent,
                sumDaySalesAmountRate=None,
                onsaleSkuNum=None,
                onsaleStock=None,
                newSkuNum=None,
                trend=[],
            ))

        return CategoryTrendCleanResponse(
            success=raw_data.success,
            result=CategoryTrendSlimResult(
                resultCount=len(raw_data.result),
                top10_slim=top10_slim,
            )
        )

    # === 店铺分析的查询节点 ===
    def _query_shop_analyze_start(self, state: ZhiyiDeepresearchState):
        return

    def _query_top_property(self, state: ZhiyiDeepresearchState):
        """查询并获取匹配到属性名"""
        req = state.request
        param = state.api_parse_param

        # 1. 获取可用属性列表
        api_client = get_zhiyi_api_client()
        property_top_request = ZhiyiPropertyTopRequest(
            rootCategoryId=param.root_category_id,
            categoryId=param.category_id,
            rootCategoryIdList=param.root_category_id,
        )
        property_list_resp = api_client.get_property_top_list(
            user_id=req.user_id,
            team_id=req.team_id,
            params=property_top_request,
        )

        # 2. 使用LLM提取用户问题中的属性名
        property_list = property_list_resp.result
        property_names = [p.property_name for p in property_list if p.property_name]
        property_dict = {
            p.property_name: p.property_value
            for p in property_list
            if p.property_name is not None
        }

        llm = llm_factory.get_llm(LlmProvider.HUANXIN.name, LlmModelName.HUANXIN_GEMINI_3_FLASH_PREVIEW.value)
        struct_llm = llm.with_structured_output(schema=PropertyExtractResult).with_retry(stop_after_attempt=2)

        invoke_params = {
            "user_query": req.user_query,
            "property_list": json.dumps(property_names, ensure_ascii=False),
        }
        messages: list[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.DR_ZHIYI_THINKING_PROPERTY_EXTRACT_PROMPT.value,
            variables=invoke_params,
        )
        extract_result: PropertyExtractResult = struct_llm.with_config(run_name="属性提取").invoke(messages)

        # 如果提取失败，使用第一个属性作为默认值
        property_name = extract_result.property_name
        property_value_list = property_dict.get(property_name)
        if not property_name and property_names:
            property_name = property_names[0]
            property_value_list = []

        return {
            "shop_property_name": property_name,
            "shop_property_values": property_value_list,
        }

    def _extract_price_band(self, state: ZhiyiDeepresearchState):
        """提取价格带"""
        req = state.request

        # llm
        llm = llm_factory.get_llm(LlmProvider.HUANXIN.name, LlmModelName.HUANXIN_GEMINI_3_FLASH_PREVIEW.value)
        struct_llm = llm.with_structured_output(schema=PriceBandExtractResult).with_retry(stop_after_attempt=2)
        # prompt
        invoke_params = {
            "user_query": req.user_query,
        }
        messages = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.DR_ZHIYI_THINKING_PRICE_EXTRACT_PROMPT.value,
            variables=invoke_params,
        )
        # execute
        result: PriceBandExtractResult = struct_llm.with_config(run_name="提取价格带").invoke(messages)
        return {
            "price_band_list": result.price_band_list
        }

    def _query_shop_analyze_overview(self, state: ZhiyiDeepresearchState):
        """查询店铺概览数据（销量+销售额趋势）"""
        req = state.request
        param = state.api_parse_param
        shop_id = self._safe_int(state.target_shop_id)
        granularity = self._get_granularity(param.date_type)

        api_client = get_zhiyi_api_client()

        # 查询销量趋势 (distribution=1)
        volume_request = ZhiyiSaleTrendRequest(
            shopId=shop_id,
            distribution=1,
            granularity=granularity,
            startDate=param.start_date,
            endDate=param.end_date,
        )
        volume_resp = api_client.get_sale_trend(
            user_id=req.user_id,
            team_id=req.team_id,
            params=volume_request,
        )

        # 查询销售额趋势 (distribution=3)
        amount_request = ZhiyiSaleTrendRequest(
            shopId=shop_id,
            distribution=3,
            granularity=granularity,
            startDate=param.start_date,
            endDate=param.end_date,
        )
        amount_resp = api_client.get_sale_trend(
            user_id=req.user_id,
            team_id=req.team_id,
            params=amount_request,
        )

        return {
            "shop_overview_volume": self._clean_overview_data(volume_resp),
            "shop_overview_amount": self._clean_overview_data(amount_resp),
        }

    def _query_shop_analyze_price(self, state: ZhiyiDeepresearchState):
        """查询店铺价格带数据"""
        req = state.request
        param = state.api_parse_param
        shop_id = state.target_shop_id
        granularity = self._get_granularity(param.date_type)

        api_client = get_zhiyi_api_client()

        price_request = ZhiyiPriceRangeTrendRequest(
            rootCategoryId=param.root_category_id,
            categoryId=param.category_id,
            startDate=param.start_date,
            endDate=param.end_date,
            shopId=shop_id,
            granularity=granularity,
            rangeList=state.price_band_list,
        )
        price_resp = api_client.get_price_range_trend(
            user_id=req.user_id,
            team_id=req.team_id,
            params=price_request,
        )

        return {
            "shop_price_trend": self._clean_price_range_data(price_resp),
        }

    def _query_shop_analyze_property(self, state: ZhiyiDeepresearchState):
        """查询店铺属性数据（需要先提取属性名）"""
        req = state.request
        param = state.api_parse_param
        shop_id = state.target_shop_id
        granularity = self._get_granularity(param.date_type)
        property_name = state.shop_property_name

        api_client = get_zhiyi_api_client()
        # 查询属性趋势数据
        property_request = ZhiyiPropertyTrendRequest(
            rootCategoryId=param.root_category_id,
            categoryId=param.category_id,
            startDate=param.start_date,
            endDate=param.end_date,
            shopId=shop_id,
            granularity=granularity,
            propertyName=property_name,
        )
        property_resp = api_client.get_property_trend(
            user_id=req.user_id,
            team_id=req.team_id,
            params=property_request,
        )

        return {
            "shop_property_data": self._clean_color_data(property_resp),
        }

    def _query_shop_analyze_color(self, state: ZhiyiDeepresearchState):
        """查询店铺颜色数据"""
        req = state.request
        param = state.api_parse_param
        shop_id = self._safe_int(state.target_shop_id)
        granularity = self._get_granularity(param.date_type)

        api_client = get_zhiyi_api_client()

        color_request = ZhiyiPropertyTrendRequest(
            rootCategoryId=param.root_category_id,
            categoryId=param.category_id,
            startDate=param.start_date,
            endDate=param.end_date,
            shopId=shop_id,
            granularity=granularity,
            propertyType=3,
            propertyName="颜色",
        )
        color_resp = api_client.get_property_trend(
            user_id=req.user_id,
            team_id=req.team_id,
            params=color_request,
        )

        return {
            "shop_color_data": self._clean_color_data(color_resp),
        }

    def _query_shop_analyze_category(self, state: ZhiyiDeepresearchState):
        """查询店铺品类趋势数据"""
        req = state.request
        param = state.api_parse_param
        shop_id = self._safe_int(state.target_shop_id)
        granularity = self._get_granularity(param.date_type)

        api_client = get_zhiyi_api_client()

        category_request = ZhiyiCategoryTrendRequest(
            startDate=param.start_date,
            endDate=param.end_date,
            shopId=shop_id,
            rootCategoryId=param.root_category_id,
            categoryId=param.category_id,
            granularity=granularity,
        )
        category_resp = api_client.get_category_trend(
            user_id=req.user_id,
            team_id=req.team_id,
            params=category_request,
        )

        return {
            "shop_category_trend": self._clean_category_trend_data(category_resp),
        }

    def _query_shop_analyze_top_goods(self, state: ZhiyiDeepresearchState):
        """查询店铺Top10热销商品"""
        req = state.request
        param = state.api_parse_param
        shop_id = self._safe_int(state.target_shop_id)

        api_client = get_zhiyi_api_client()

        root_category_id = self._safe_int(param.root_category_id)
        category_id = param.category_id

        hot_item_request = ZhiyiShopHotItemRequest(
            shopId=shop_id,
            rootCategoryIdList=[root_category_id] if root_category_id else [],
            categoryIdList=[category_id] if category_id else None,
            startDate=param.start_date,
            endDate=param.end_date,
        )
        hot_item_resp: ShopHotItemRawResponse = api_client.get_shop_hot_items(
            user_id=req.user_id,
            team_id=req.team_id,
            params=hot_item_request,
        )

        return {
            "shop_top_goods": self._clean_top10_items_data(hot_item_resp),
            "shop_top_goods_raw": hot_item_resp
        }

    def _assemble_shop_analyze_result(self, state: ZhiyiDeepresearchState):
        """
        聚合店铺分析结果并构建 ZhiyiAggregatedData
        """
        parse_param = state.api_parse_param
        table_type = parse_param.table_type

        # 根据 table_type 选择数据
        if table_type == "1":
            # 品类分析
            aggregated = ZhiyiAggregatedData(
                is_shop=True,
                table_type=table_type,
                shop_name=state.target_shop_name,
                category_path=f"{parse_param.root_category_id_name}-{parse_param.category_id_name}",
                start_date=parse_param.start_date,
                end_date=parse_param.end_date,
                overview_volume=state.shop_overview_volume,
                overview_amount=state.shop_overview_amount,
                category_data=state.shop_category_trend,
                price_data=state.shop_price_trend,
                color_data=state.shop_color_data,
                top_goods=state.shop_top_goods,
                top_goods_raw=state.shop_top_goods_raw,
            )
        elif table_type == "2":
            # 属性分析
            aggregated = ZhiyiAggregatedData(
                is_shop=True,
                table_type=table_type,
                shop_name=state.target_shop_name,
                category_path=f"{parse_param.root_category_id_name}-{parse_param.category_id_name}",
                start_date=parse_param.start_date,
                end_date=parse_param.end_date,
                property_name=state.shop_property_name,
                overview_volume=state.shop_overview_volume,
                overview_amount=state.shop_overview_amount,
                property_data=state.shop_property_data,
                top_goods=state.shop_top_goods,
                top_goods_raw=state.shop_top_goods_raw,
            )
        else:
            # 价格带分析
            aggregated = ZhiyiAggregatedData(
                is_shop=True,
                table_type=table_type,
                shop_name=state.target_shop_name,
                category_path=f"{parse_param.root_category_id_name}-{parse_param.category_id_name}",
                start_date=parse_param.start_date,
                end_date=parse_param.end_date,
                overview_volume=state.shop_overview_volume,
                overview_amount=state.shop_overview_amount,
                price_data=state.shop_price_trend,
                top_goods=state.shop_top_goods,
                top_goods_raw=state.shop_top_goods_raw,
            )

        return {"aggregated_data": aggregated}


    def _replace_report_product_cards(self, report_text: str, origin_product_resp: ShopHotItemRawResponse) -> str:
        new_product_card_content = f"<custom-productcards>\n{origin_product_resp.model_dump_json(by_alias=True, ensure_ascii=False)}\n</custom-productcards>"
        replaced_text = self.product_area_match_prog.sub(new_product_card_content, report_text)
        return replaced_text

    def _generate_shop_category_analyze_report(self, state: ZhiyiDeepresearchState):
        """
        生成品类分析报告
        """
        req = state.request
        parse_param = state.api_parse_param
        pusher = state.message_pusher

        pusher.push_phase("洞察生成中")

        # llm
        llm = llm_factory.get_llm(LlmProvider.HUANXIN.name, LlmModelName.HUANXIN_GEMINI_3_PRO_PREVIEW.value)
        chain = llm.with_retry(stop_after_attempt=2) | StrOutputParser()

        # prompt
        invoke_params = {
            "user_query": req.user_query,
            "shop_name": state.target_shop_name,
            "date_range": f"{parse_param.start_date}至{parse_param.end_date}",
            "category_path": f"{parse_param.root_category_id_name}-{parse_param.category_id_name}",
            "sale_volume_trend_data": state.shop_overview_volume.model_dump_json(ensure_ascii=False),
            "sale_amount_trend_data": state.shop_overview_amount.model_dump_json(ensure_ascii=False),
            "category_trend_data": state.shop_category_trend.model_dump_json(ensure_ascii=False),
            "price_trend_data": state.shop_price_trend.model_dump_json(ensure_ascii=False),
            "color_trend_data": state.shop_color_data.model_dump_json(ensure_ascii=False),
            "top_goods_data": state.shop_top_goods.model_dump_json(ensure_ascii=False),
        }
        messages = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.DR_ZHIYI_THINKING_SHOP_CATEGORY_REPORT_PROMPT.value,
            variables=invoke_params,
        )

        # 执行llm
        report_text: str = chain.with_config(run_name="生成品类趋势报告").invoke(messages)

        # 替换商品卡数据
        replaced_text = self._replace_report_product_cards(report_text=report_text, origin_product_resp=state.shop_top_goods_raw)

        # 构建 Excel 导出数据
        exporter = ZhiyiExcelExporter(
            aggregated_data=state.aggregated_data,
            param=state.api_parse_param,
            is_thinking=state.is_thinking,
        )
        excel_data_list = exporter.build_excel_data_list()

        pusher.push_phase("洞察生成完成")
        pusher.push_report_and_excel_data(
            entity_type=WorkflowEntityType.TAOBAO_ITEM.code,
            report_text=replaced_text,
            excel_data_list=excel_data_list
        )

        return {
            "report_text": replaced_text
        }

    def _generate_shop_normal_analyze_report(self, state: ZhiyiDeepresearchState):
        """生成非品类分析报告（属性/价格带等）
        """
        req = state.request
        parse_param = state.api_parse_param
        pusher = state.message_pusher
        shop_name = state.target_shop_name

        pusher.push_phase("洞察生成中")

        # 判断分析维度（价格带/属性/其他属性等）
        dimension = "价格带"
        table_type = parse_param.table_type
        property_name = state.shop_property_name
        if table_type == "3":
            dimension = "价格带"
        else:
            dimension = "属性"

        if dimension == "价格带":
            dimension_data_name = "价格带分布数据"
            dimension_trend_data: BaseModel = state.shop_price_trend
        elif dimension == "属性":
            dimension_data_name = "属性分布数据"
            dimension_trend_data: BaseModel = state.shop_property_data
        else:
            logger.warning(f"[知衣数据洞察]不合法的dimension类型:{dimension}, 将使用价格带数据")
            dimension_data_name = "价格带分布数据"
            dimension_trend_data: BaseModel = state.shop_price_trend


        # llm
        llm = llm_factory.get_llm(LlmProvider.HUANXIN.name, LlmModelName.HUANXIN_GEMINI_3_PRO_PREVIEW.value)
        chain = llm.with_retry(stop_after_attempt=2) | StrOutputParser()

        # prompt
        invoke_params = {
            "user_query": req.user_query,
            "shop_name": shop_name,
            "date_range": f"{parse_param.start_date}至{parse_param.end_date}",
            "category_path": f"{parse_param.root_category_id_name}-{parse_param.category_id_name}",
            "sale_volume_trend_data": state.shop_overview_volume.model_dump_json(ensure_ascii=False),
            "sale_amount_trend_data": state.shop_overview_amount.model_dump_json(ensure_ascii=False),
            # 当前分析维度的趋势数据
            "dimension_data_name": dimension_data_name,
            "dimension_trend_data": dimension_trend_data.model_dump_json(ensure_ascii=False),
            "top_goods_data": state.shop_top_goods.model_dump_json(ensure_ascii=False),
        }
        messages = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.DR_ZHIYI_THINKING_SHOP_NORMAL_REPORT_PROMPT.value,
            variables=invoke_params,
        )

        # 执行llm
        report_text: str = chain.with_config(run_name="生产普通趋势报告").invoke(messages)

        # 替换商品卡数据
        replaced_text = self._replace_report_product_cards(report_text=report_text, origin_product_resp=state.shop_top_goods_raw)

        # 构建 Excel 导出数据
        exporter = ZhiyiExcelExporter(
            aggregated_data=state.aggregated_data,
            param=state.api_parse_param,
            is_thinking=state.is_thinking,
        )
        excel_data_list = exporter.build_excel_data_list()

        pusher.push_phase("洞察生成完成")
        pusher.push_report_and_excel_data(
            entity_type=WorkflowEntityType.TAOBAO_ITEM.code,
            report_text=replaced_text,
            excel_data_list=excel_data_list
        )

        return {
            "report_text": replaced_text
        }


    # === 大盘分析的路由和查询节点 ===

    def _hydc_table_type_router_node(self, state: ZhiyiDeepresearchState):
        """
        大盘 table_type 路由节点 - 实际路由在 _route_hydc_table_type_node 中进行
        这里主要处理消息发送
        """
        is_thinking = state.is_thinking
        parse_param = state.api_parse_param
        pusher = state.message_pusher

        # 深度思考的推送
        if is_thinking:
            pusher.push_phase("数据收集", variables={
                "module_name": "行业洞察",
                "module_link": f"https://data.zhiyitech.cn/trend?type=overview-insight",
                "industry_name": parse_param.root_category_id_name
            })
        return

    def _route_hydc_table_type_node(self, state: ZhiyiDeepresearchState) -> Literal["1", "2", "3"]:
        """路由大盘分析的 table_type"""
        param = state.api_parse_param
        dimension = param.table_type
        if not dimension:
            return "1"
        return dimension

    def _route_hydc_dimension_node(self, state: ZhiyiDeepresearchState) -> int:
        """路由维度分析类型：0=属性/1=颜色/2=品牌"""
        return state.hydc_dimension_type or 0

    def _analyze_hydc_dimension(self, state: ZhiyiDeepresearchState):
        """大盘维度分析 - LLM判断用户意图（属性/颜色/品牌）"""
        req = state.request

        llm = llm_factory.get_llm(LlmProvider.HUANXIN.name, LlmModelName.HUANXIN_GEMINI_3_FLASH_PREVIEW.value)
        struct_llm = llm.with_structured_output(schema=HydcDimensionAnalyzeResult).with_retry(stop_after_attempt=2)

        invoke_params = {
            "user_query": req.user_query,
        }
        messages = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.DR_ZHIYI_HYDC_DIMENSION_ANALYZE_PROMPT.value,
            variables=invoke_params,
        )

        result: HydcDimensionAnalyzeResult = struct_llm.with_config(run_name="查询的维度分析").invoke(messages)
        return {"hydc_dimension_type": result.dimension_type}

    def _query_hydc_analyze_start(self, state: ZhiyiDeepresearchState):
        """大盘分析开始 - 推送消息"""
        return

    # === 大盘品类分析 (table_type=1) ===

    def _query_hydc_category_overview(self, state: ZhiyiDeepresearchState):
        """查询大盘品类概览数据（销量+销售额趋势）"""
        req = state.request
        param = state.api_parse_param
        date_type = "month" if param.date_type == "month" else "week"

        api_client = get_zhiyi_api_client()

        # 使用统一的 trend-list 接口，groupType=overview
        overview_request = ZhiyiHydcTrendListRequest(
            rootCategoryId=self._safe_int(param.root_category_id),
            categoryId=self._safe_int(param.category_id) if param.category_id else None,
            startDate=param.start_date,
            endDate=param.end_date,
            groupType="overview",
            dateType=date_type,
        )
        overview_resp = api_client.get_hydc_trend_list(
            user_id=req.user_id,
            team_id=req.team_id,
            params=overview_request,
        )

        return {
            "hydc_category_overview_volume": self._clean_hydc_overview_data(overview_resp),
            "hydc_category_overview_amount": self._clean_hydc_overview_data(overview_resp),
        }

    def _query_hydc_category_price(self, state: ZhiyiDeepresearchState):
        """查询大盘品类价格带数据"""
        req = state.request
        param = state.api_parse_param
        date_type = "month" if param.date_type == "month" else "week"

        api_client = get_zhiyi_api_client()

        # groupType=cprice，添加 priceBindList
        price_request = ZhiyiHydcTrendListRequest(
            rootCategoryId=self._safe_int(param.root_category_id),
            categoryId=self._safe_int(param.category_id) if param.category_id else None,
            startDate=param.start_date,
            endDate=param.end_date,
            groupType="cprice",
            dateType=date_type,
            priceBindList=[","],
        )
        price_resp = api_client.get_hydc_trend_list(
            user_id=req.user_id,
            team_id=req.team_id,
            params=price_request,
        )

        return {
            "hydc_category_price_data": self._clean_hydc_price_data(price_resp),
        }

    def _query_hydc_category_color(self, state: ZhiyiDeepresearchState):
        """查询大盘品类颜色数据"""
        req = state.request
        param = state.api_parse_param
        date_type = "month" if param.date_type == "month" else "week"

        api_client = get_zhiyi_api_client()

        # groupType=color，添加 priceBindList
        color_request = ZhiyiHydcTrendListRequest(
            rootCategoryId=self._safe_int(param.root_category_id),
            categoryId=self._safe_int(param.category_id) if param.category_id else None,
            startDate=param.start_date,
            endDate=param.end_date,
            groupType="color",
            dateType=date_type,
            priceBindList=[","],
        )
        color_resp = api_client.get_hydc_trend_list(
            user_id=req.user_id,
            team_id=req.team_id,
            params=color_request,
        )

        return {
            "hydc_category_color_data": self._clean_hydc_color_data(color_resp),
        }

    def _query_hydc_category_category(self, state: ZhiyiDeepresearchState):
        """查询大盘品类趋势数据"""
        req = state.request
        param = state.api_parse_param
        date_type = "month" if param.date_type == "month" else "week"

        api_client = get_zhiyi_api_client()

        # 使用 groupType=overview 获取品类概览趋势
        category_request = ZhiyiHydcTrendListRequest(
            rootCategoryId=self._safe_int(param.root_category_id),
            categoryId=self._safe_int(param.category_id) if param.category_id else None,
            startDate=param.start_date,
            endDate=param.end_date,
            groupType="overview",
            dateType=date_type,
        )
        category_resp = api_client.get_hydc_trend_list(
            user_id=req.user_id,
            team_id=req.team_id,
            params=category_request,
        )

        return {
            "hydc_category_category_data": self._clean_hydc_category_data(category_resp),
        }

    def _query_hydc_category_top_goods(self, state: ZhiyiDeepresearchState):
        """查询大盘品类Top10热销商品"""
        req = state.request
        param = state.api_parse_param

        api_client = get_zhiyi_api_client()

        root_category_id = self._safe_int(param.root_category_id)
        category_id = self._safe_int(param.category_id) if param.category_id else None

        # 使用 /v2-5-6/rank/item-list-v3 接口
        rank_request = ZhiyiHydcRankItemRequest(
            rootCategoryId=root_category_id,
            categoryId=category_id,
            startDate=param.start_date,
            endDate=param.end_date,
        )
        rank_resp = api_client.get_hydc_rank_items(
            user_id=req.user_id,
            team_id=req.team_id,
            params=rank_request,
        )

        return {
            "hydc_category_top_goods": self._clean_top10_items_data(rank_resp),
            "hydc_category_top_goods_raw": rank_resp,  # 新增：保存原始响应数据
        }

    def _assemble_hydc_category_result(self, state: ZhiyiDeepresearchState):
        """聚合大盘品类分析结果并构建 ZhiyiAggregatedData"""
        parse_param = state.api_parse_param

        aggregated = ZhiyiAggregatedData(
            is_shop=False,
            table_type="1",
            category_path=f"{parse_param.root_category_id_name}-{parse_param.category_id_name}",
            start_date=parse_param.start_date,
            end_date=parse_param.end_date,
            overview_volume=state.hydc_category_overview_volume,
            overview_amount=state.hydc_category_overview_amount,
            category_data=state.hydc_category_category_data,
            price_data=state.hydc_category_price_data,
            color_data=state.hydc_category_color_data,
            top_goods=state.hydc_category_top_goods,
            top_goods_raw=state.hydc_category_top_goods_raw,
        )

        return {"aggregated_data": aggregated}

    def _generate_hydc_category_report(self, state: ZhiyiDeepresearchState):
        """生成大盘品类分析报告"""
        req = state.request
        parse_param = state.api_parse_param
        pusher = state.message_pusher
        category_name = parse_param.category_id_name or parse_param.root_category_id_name or "行业"

        pusher.push_phase("洞察生成中")

        llm = llm_factory.get_llm(LlmProvider.HUANXIN.name, LlmModelName.HUANXIN_GEMINI_3_FLASH_PREVIEW.value)
        chain = llm.with_retry(stop_after_attempt=2) | StrOutputParser()

        invoke_params = {
            "user_query": req.user_query,
            "date_range": f"{parse_param.start_date}至{parse_param.end_date}",
            "category_path": f"{parse_param.root_category_id_name}-{parse_param.category_id_name}",
            "sale_volume_trend_data": state.hydc_category_overview_volume.model_dump_json(ensure_ascii=False) if state.hydc_category_overview_volume else "{}",
            "sale_amount_trend_data": state.hydc_category_overview_amount.model_dump_json(ensure_ascii=False) if state.hydc_category_overview_amount else "{}",

            "price_trend_data": state.hydc_category_price_data.model_dump_json(ensure_ascii=False) if state.hydc_category_price_data else "{}",
            "color_trend_data": state.hydc_category_color_data.model_dump_json(ensure_ascii=False) if state.hydc_category_color_data else "{}",
            "top_products": state.hydc_category_top_goods.model_dump_json(ensure_ascii=False),
        }
        messages = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.DR_ZHIYI_HYDC_CATEGORY_REPORT_PROMPT.value,
            variables=invoke_params,
        )

        report_text: str = chain.with_config(run_name="生成大盘品类趋势报告").invoke(messages)

        # 替换商品卡数据
        replaced_text = self._replace_report_product_cards(
            report_text=report_text,
            origin_product_resp=state.hydc_category_top_goods_raw
        )

        # 构建 Excel 导出数据
        exporter = ZhiyiExcelExporter(
            aggregated_data=state.aggregated_data,
            param=state.api_parse_param,
            is_thinking=state.is_thinking,
        )
        excel_data_list = exporter.build_excel_data_list()

        pusher.push_phase("洞察生成完成")
        pusher.push_report_and_excel_data(
            entity_type=WorkflowEntityType.TAOBAO_ITEM.code,
            report_text=replaced_text,
            excel_data_list=excel_data_list
        )

        return {"report_text": replaced_text}

    # === 大盘属性分析 (table_type=2, dimension=0) ===

    def _query_hydc_property_start(self, state: ZhiyiDeepresearchState):
        """大盘属性分析开始"""
        return

    def _extract_hydc_property(self, state: ZhiyiDeepresearchState):
        """大盘属性提取 - 从用户问题中提取属性名"""
        req = state.request
        param = state.api_parse_param

        # 获取可用属性列表
        api_client = get_zhiyi_api_client()
        property_top_request = ZhiyiPropertyTopRequest(
            rootCategoryId=param.root_category_id,
            categoryId=param.category_id if param.category_id else None,
            rootCategoryIdList=param.root_category_id,
        )
        property_list_resp = api_client.get_property_top_list(
            user_id=req.user_id,
            team_id=req.team_id,
            params=property_top_request,
        )

        property_list = property_list_resp.result if property_list_resp.result else []
        property_names = [p.property_name for p in property_list if p.property_name]
        property_dict = {
            p.property_name: p.property_value
            for p in property_list
            if p.property_name is not None
        }

        # LLM提取属性名
        llm = llm_factory.get_llm(LlmProvider.HUANXIN.name, LlmModelName.HUANXIN_GEMINI_3_FLASH_PREVIEW.value)
        struct_llm = llm.with_structured_output(schema=HydcPropertyExtractResult).with_retry(stop_after_attempt=2)

        invoke_params = {
            "user_query": req.user_query,
            "property_list": json.dumps(property_names, ensure_ascii=False) if property_names else "[]",
        }
        messages = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.DR_ZHIYI_HYDC_PROPERTY_EXTRACT_PROMPT.value,
            variables=invoke_params,
        )
        extract_result: HydcPropertyExtractResult = struct_llm.with_config(run_name="大盘属性提取").invoke(messages)

        property_name = extract_result.property_name
        property_value_list = property_dict.get(property_name)
        if not property_name and property_names:
            property_name = property_names[0]
            property_value_list = []

        return {
            "hydc_property_name": property_name,
            "hydc_property_values": property_value_list,
        }

    def _query_hydc_property_overview(self, state: ZhiyiDeepresearchState):
        """查询大盘属性概览数据"""
        req = state.request
        param = state.api_parse_param
        date_type = "month" if param.date_type == "month" else "week"

        api_client = get_zhiyi_api_client()

        # 使用统一的 trend-list 接口，groupType=overview
        overview_request = ZhiyiHydcTrendListRequest(
            rootCategoryId=self._safe_int(param.root_category_id),
            categoryId=self._safe_int(param.category_id) if param.category_id else None,
            startDate=param.start_date,
            endDate=param.end_date,
            groupType="overview",
            dateType=date_type,
        )
        overview_resp = api_client.get_hydc_trend_list(
            user_id=req.user_id,
            team_id=req.team_id,
            params=overview_request,
        )

        return {
            "hydc_property_overview_volume": self._clean_hydc_overview_data(overview_resp),
            "hydc_property_overview_amount": self._clean_hydc_overview_data(overview_resp),
        }

    def _query_hydc_property_data(self, state: ZhiyiDeepresearchState):
        """查询大盘属性分布数据"""
        req = state.request
        param = state.api_parse_param
        date_type = "month" if param.date_type == "month" else "week"
        property_name = state.hydc_property_name or "颜色"

        api_client = get_zhiyi_api_client()

        # 行业洞察与店铺分析不同，需要增加属性值列表
        property_item = {
            "name": state.hydc_property_name,
            "values": ",".join(state.hydc_property_values),
        }
        property_list = [property_item]

        # groupType=property，添加 propertyKey
        property_request = ZhiyiHydcTrendListRequest(
            rootCategoryId=self._safe_int(param.root_category_id),
            categoryId=self._safe_int(param.category_id) if param.category_id else None,
            startDate=param.start_date,
            endDate=param.end_date,
            groupType="property",
            dateType=date_type,
            propertyName=property_name,
            propertyList=property_list,
        )
        property_resp = api_client.get_hydc_trend_list(
            user_id=req.user_id,
            team_id=req.team_id,
            params=property_request,
        )

        return {
            "hydc_property_data": self._clean_hydc_property_data(property_resp),
        }

    def _query_hydc_property_top_goods(self, state: ZhiyiDeepresearchState):
        """查询大盘属性Top10商品"""
        req = state.request
        param = state.api_parse_param

        api_client = get_zhiyi_api_client()

        root_category_id = self._safe_int(param.root_category_id)
        category_id = self._safe_int(param.category_id) if param.category_id else None

        # 使用 /v2-5-6/rank/item-list-v3 接口
        rank_request = ZhiyiHydcRankItemRequest(
            rootCategoryId=root_category_id,
            categoryId=category_id,
            startDate=param.start_date,
            endDate=param.end_date,
        )
        rank_resp = api_client.get_hydc_rank_items(
            user_id=req.user_id,
            team_id=req.team_id,
            params=rank_request,
        )

        return {
            "hydc_property_top_goods": self._clean_top10_items_data(rank_resp),
            "hydc_property_top_goods_raw": rank_resp,
        }

    def _assemble_hydc_property_result(self, state: ZhiyiDeepresearchState):
        """聚合大盘属性分析结果并构建 ZhiyiAggregatedData"""
        parse_param = state.api_parse_param

        aggregated = ZhiyiAggregatedData(
            is_shop=False,
            table_type="2",
            hydc_dimension_type=0,  # 属性分析
            category_path=f"{parse_param.root_category_id_name}-{parse_param.category_id_name}",
            start_date=parse_param.start_date,
            end_date=parse_param.end_date,
            property_name=state.hydc_property_name,
            overview_volume=state.hydc_property_overview_volume,
            overview_amount=state.hydc_property_overview_amount,
            property_data=state.hydc_property_data,
            top_goods=state.hydc_property_top_goods,
            top_goods_raw=state.hydc_property_top_goods_raw,
        )

        return {"aggregated_data": aggregated}

    # === 大盘颜色分析 (table_type=2, dimension=1) ===

    def _query_hydc_color_start(self, state: ZhiyiDeepresearchState):
        """大盘颜色分析开始"""
        return

    def _query_hydc_color_overview(self, state: ZhiyiDeepresearchState):
        """查询大盘颜色概览数据"""
        req = state.request
        param = state.api_parse_param
        date_type = "month" if param.date_type == "month" else "week"

        api_client = get_zhiyi_api_client()

        # 使用统一的 trend-list 接口，groupType=overview
        overview_request = ZhiyiHydcTrendListRequest(
            rootCategoryId=self._safe_int(param.root_category_id),
            categoryId=self._safe_int(param.category_id) if param.category_id else None,
            startDate=param.start_date,
            endDate=param.end_date,
            groupType="overview",
            dateType=date_type,
        )
        overview_resp = api_client.get_hydc_trend_list(
            user_id=req.user_id,
            team_id=req.team_id,
            params=overview_request,
        )

        return {
            "hydc_color_overview_volume": self._clean_hydc_overview_data(overview_resp),
            "hydc_color_overview_amount": self._clean_hydc_overview_data(overview_resp),
        }

    def _query_hydc_color_data(self, state: ZhiyiDeepresearchState):
        """查询大盘颜色分布数据"""
        req = state.request
        param = state.api_parse_param
        date_type = "month" if param.date_type == "month" else "week"

        api_client = get_zhiyi_api_client()

        # groupType=color，添加 priceBindList
        color_request = ZhiyiHydcTrendListRequest(
            rootCategoryId=self._safe_int(param.root_category_id),
            categoryId=self._safe_int(param.category_id) if param.category_id else None,
            startDate=param.start_date,
            endDate=param.end_date,
            groupType="color",
            dateType=date_type,
            priceBindList=[","],
        )
        color_resp = api_client.get_hydc_trend_list(
            user_id=req.user_id,
            team_id=req.team_id,
            params=color_request,
        )

        return {
            "hydc_color_data": self._clean_hydc_color_data(color_resp),
        }

    def _query_hydc_color_top_goods(self, state: ZhiyiDeepresearchState):
        """查询大盘颜色Top10商品"""
        req = state.request
        param = state.api_parse_param

        api_client = get_zhiyi_api_client()

        root_category_id = self._safe_int(param.root_category_id)
        category_id = self._safe_int(param.category_id) if param.category_id else None

        # 使用 /v2-5-6/rank/item-list-v3 接口
        rank_request = ZhiyiHydcRankItemRequest(
            rootCategoryId=root_category_id,
            categoryId=category_id,
            startDate=param.start_date,
            endDate=param.end_date,
        )
        rank_resp = api_client.get_hydc_rank_items(
            user_id=req.user_id,
            team_id=req.team_id,
            params=rank_request,
        )

        return {
            "hydc_color_top_goods": self._clean_top10_items_data(rank_resp),
            "hydc_color_top_goods_raw": rank_resp,
        }

    def _assemble_hydc_color_result(self, state: ZhiyiDeepresearchState):
        """聚合大盘颜色分析结果并构建 ZhiyiAggregatedData"""
        parse_param = state.api_parse_param

        aggregated = ZhiyiAggregatedData(
            is_shop=False,
            table_type="2",
            hydc_dimension_type=1,  # 颜色分析
            category_path=f"{parse_param.root_category_id_name}-{parse_param.category_id_name}",
            start_date=parse_param.start_date,
            end_date=parse_param.end_date,
            overview_volume=state.hydc_color_overview_volume,
            overview_amount=state.hydc_color_overview_amount,
            color_data=state.hydc_color_data,
            top_goods=state.hydc_color_top_goods,
            top_goods_raw=state.hydc_color_top_goods_raw,
        )

        return {"aggregated_data": aggregated}

    # === 大盘品牌分析 (table_type=2, dimension=2) ===

    def _query_hydc_brand_start(self, state: ZhiyiDeepresearchState):
        """大盘品牌分析开始"""
        return

    def _query_hydc_brand_overview(self, state: ZhiyiDeepresearchState):
        """查询大盘品牌概览数据"""
        req = state.request
        param = state.api_parse_param
        date_type = "month" if param.date_type == "month" else "week"

        api_client = get_zhiyi_api_client()

        # 使用统一的 trend-list 接口，groupType=overview
        overview_request = ZhiyiHydcTrendListRequest(
            rootCategoryId=self._safe_int(param.root_category_id),
            categoryId=self._safe_int(param.category_id) if param.category_id else None,
            startDate=param.start_date,
            endDate=param.end_date,
            groupType="overview",
            dateType=date_type,
        )
        overview_resp = api_client.get_hydc_trend_list(
            user_id=req.user_id,
            team_id=req.team_id,
            params=overview_request,
        )

        return {
            "hydc_brand_overview_volume": self._clean_hydc_overview_data(overview_resp),
            "hydc_brand_overview_amount": self._clean_hydc_overview_data(overview_resp),
        }

    def _query_hydc_brand_data(self, state: ZhiyiDeepresearchState):
        """查询大盘品牌分布数据"""
        req = state.request
        param = state.api_parse_param
        date_type = "month" if param.date_type == "month" else "week"

        api_client = get_zhiyi_api_client()

        # groupType=brand，添加 type 和 propertyList
        brand_request = ZhiyiHydcTrendListRequest(
            rootCategoryId=self._safe_int(param.root_category_id),
            categoryId=self._safe_int(param.category_id) if param.category_id else None,
            startDate=param.start_date,
            endDate=param.end_date,
            groupType="brand",
            dateType=date_type,
            type="category-insight",
            propertyList=[],
            brandName="",
        )
        brand_resp = api_client.get_hydc_trend_list(
            user_id=req.user_id,
            team_id=req.team_id,
            params=brand_request,
        )

        return {
            "hydc_brand_data": self._clean_hydc_brand_data(brand_resp),
        }

    def _clean_brand_data(self, raw_data: BrandRawResponse) -> BrandCleanResponse:
        """清洗品牌数据"""
        brand_slim = [
            BrandSlimItem(
                brandName=it.brand_name,
                salesVolume=it.sum_day_sales_volume_by_brand or 0,
                salesAmount=it.sum_day_sale_by_brand,
                rate=it.sum_day_sales_volume_by_brand_rate,
                otherFlag=it.other_flag,
            )
            for it in raw_data.result
        ]
        return BrandCleanResponse(
            success=raw_data.success,
            result=BrandSlimResult(brand_slim=brand_slim)
        )

    def _query_hydc_brand_top_goods(self, state: ZhiyiDeepresearchState):
        """查询大盘品牌Top10商品"""
        req = state.request
        param = state.api_parse_param

        api_client = get_zhiyi_api_client()

        root_category_id = self._safe_int(param.root_category_id)
        category_id = self._safe_int(param.category_id) if param.category_id else None

        # 使用 /v2-5-6/rank/item-list-v3 接口
        rank_request = ZhiyiHydcRankItemRequest(
            rootCategoryId=root_category_id,
            categoryId=category_id,
            startDate=param.start_date,
            endDate=param.end_date,
        )
        rank_resp = api_client.get_hydc_rank_items(
            user_id=req.user_id,
            team_id=req.team_id,
            params=rank_request,
        )

        return {
            "hydc_brand_top_goods": self._clean_top10_items_data(rank_resp),
            "hydc_brand_top_goods_raw": rank_resp,
        }

    def _assemble_hydc_brand_result(self, state: ZhiyiDeepresearchState):
        """聚合大盘品牌分析结果并构建 ZhiyiAggregatedData"""
        parse_param = state.api_parse_param

        aggregated = ZhiyiAggregatedData(
            is_shop=False,
            table_type="2",
            hydc_dimension_type=2,  # 品牌分析
            category_path=f"{parse_param.root_category_id_name}-{parse_param.category_id_name}",
            start_date=parse_param.start_date,
            end_date=parse_param.end_date,
            overview_volume=state.hydc_brand_overview_volume,
            overview_amount=state.hydc_brand_overview_amount,
            brand_data=state.hydc_brand_data,
            top_goods=state.hydc_brand_top_goods,
            top_goods_raw=state.hydc_brand_top_goods_raw,
        )

        return {"aggregated_data": aggregated}

    # === 大盘价格带分析 (table_type=3) ===

    def _query_hydc_price_start(self, state: ZhiyiDeepresearchState):
        """大盘价格带分析开始"""
        return

    def _query_hydc_price_overview(self, state: ZhiyiDeepresearchState):
        """查询大盘价格带概览数据"""
        req = state.request
        param = state.api_parse_param
        date_type = "month" if param.date_type == "month" else "week"

        api_client = get_zhiyi_api_client()

        # 使用统一的 trend-list 接口，groupType=overview
        overview_request = ZhiyiHydcTrendListRequest(
            rootCategoryId=self._safe_int(param.root_category_id),
            categoryId=self._safe_int(param.category_id) if param.category_id else None,
            startDate=param.start_date,
            endDate=param.end_date,
            groupType="overview",
            dateType=date_type,
        )
        overview_resp = api_client.get_hydc_trend_list(
            user_id=req.user_id,
            team_id=req.team_id,
            params=overview_request,
        )

        return {
            "hydc_price_overview_volume": self._clean_hydc_overview_data(overview_resp),
            "hydc_price_overview_amount": self._clean_hydc_overview_data(overview_resp),
        }

    def _query_hydc_price_data(self, state: ZhiyiDeepresearchState):
        """查询大盘价格带分布数据"""
        req = state.request
        param = state.api_parse_param
        date_type = "month" if param.date_type == "month" else "week"

        api_client = get_zhiyi_api_client()

        # groupType=cprice，添加 priceBindList
        price_request = ZhiyiHydcTrendListRequest(
            rootCategoryId=param.root_category_id,
            categoryId=param.category_id if param.category_id else None,
            startDate=param.start_date,
            endDate=param.end_date,
            groupType="cprice",
            dateType=date_type,
            priceBindList=state.price_band_list,
        )
        price_resp = api_client.get_hydc_trend_list(
            user_id=req.user_id,
            team_id=req.team_id,
            params=price_request,
        )

        return {
            "hydc_price_data": self._clean_hydc_price_data(price_resp),
        }

    def _query_hydc_price_top_goods(self, state: ZhiyiDeepresearchState):
        """查询大盘价格带Top10商品"""
        req = state.request
        param = state.api_parse_param

        api_client = get_zhiyi_api_client()

        root_category_id = self._safe_int(param.root_category_id)
        category_id = self._safe_int(param.category_id) if param.category_id else None

        # 使用 /v2-5-6/rank/item-list-v3 接口
        rank_request = ZhiyiHydcRankItemRequest(
            rootCategoryId=root_category_id,
            categoryId=category_id,
            startDate=param.start_date,
            endDate=param.end_date,
        )
        rank_resp = api_client.get_hydc_rank_items(
            user_id=req.user_id,
            team_id=req.team_id,
            params=rank_request,
        )

        return {
            "hydc_price_top_goods": self._clean_top10_items_data(rank_resp),
            "hydc_price_top_goods_raw": rank_resp,
        }

    def _assemble_hydc_price_result(self, state: ZhiyiDeepresearchState):
        """聚合大盘价格带分析结果并构建 ZhiyiAggregatedData"""
        parse_param = state.api_parse_param

        aggregated = ZhiyiAggregatedData(
            is_shop=False,
            table_type="3",
            category_path=f"{parse_param.root_category_id_name}-{parse_param.category_id_name}",
            start_date=parse_param.start_date,
            end_date=parse_param.end_date,
            overview_volume=state.hydc_price_overview_volume,
            overview_amount=state.hydc_price_overview_amount,
            price_data=state.hydc_price_data,
            top_goods=state.hydc_price_top_goods,
            top_goods_raw=state.hydc_price_top_goods_raw,
        )

        return {"aggregated_data": aggregated}

    # === 大盘通用报告生成 ===

    def _generate_hydc_normal_report(self, state: ZhiyiDeepresearchState):
        """生成大盘通用分析报告（属性/颜色/品牌/价格带）"""
        req = state.request
        parse_param = state.api_parse_param
        pusher = state.message_pusher

        dimension_type = state.hydc_dimension_type
        table_type = parse_param.table_type

        pusher.push_phase("洞察生成中")

        # 根据分析类型获取对应数据
        if table_type == "3":  # 价格带分析
            dimension_data_name = "价格带分布数据"
            overview_volume = state.hydc_price_overview_volume
            overview_amount = state.hydc_price_overview_amount
            dimension_data = state.hydc_price_data
            top_goods = state.hydc_price_top_goods
            top_goods_raw = state.hydc_price_top_goods_raw
        elif dimension_type == 0:  # 属性分析
            dimension_data_name = state.hydc_property_name + "分布数据"
            overview_volume = state.hydc_property_overview_volume
            overview_amount = state.hydc_property_overview_amount
            dimension_data = state.hydc_property_data
            top_goods = state.hydc_property_top_goods
            top_goods_raw = state.hydc_property_top_goods_raw
        elif dimension_type == 1:  # 颜色分析
            dimension_data_name = "颜色分布数据"
            overview_volume = state.hydc_color_overview_volume
            overview_amount = state.hydc_color_overview_amount
            dimension_data = state.hydc_color_data
            top_goods = state.hydc_color_top_goods
            top_goods_raw = state.hydc_color_top_goods_raw
        else:  # 品牌分析
            dimension_data_name = "品牌分布数据"
            overview_volume = state.hydc_brand_overview_volume
            overview_amount = state.hydc_brand_overview_amount
            dimension_data = state.hydc_brand_data
            top_goods = state.hydc_brand_top_goods
            top_goods_raw = state.hydc_brand_top_goods_raw

        llm = llm_factory.get_llm(LlmProvider.HUANXIN.name, LlmModelName.HUANXIN_GEMINI_3_PRO_PREVIEW.value)
        chain = llm.with_retry(stop_after_attempt=2) | StrOutputParser()

        invoke_params = {
            "user_query": req.user_query,
            "date_range": f"{parse_param.start_date}至{parse_param.end_date}",
            "category_path": f"{parse_param.root_category_id_name}-{parse_param.category_id_name}",
            "sale_volume_trend_data": overview_volume.model_dump_json(ensure_ascii=False) if overview_volume else "{}",
            "sale_amount_trend_data": overview_amount.model_dump_json(ensure_ascii=False) if overview_amount else "{}",
            # 当前分析维度的趋势数据
            "dimension_data_name": dimension_data_name,
            "dimension_trend_data": dimension_data.model_dump_json(ensure_ascii=False),
            "top_products": top_goods.model_dump_json(ensure_ascii=False),
        }
        messages = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.DR_ZHIYI_HYDC_NORMAL_REPORT_PROMPT.value,
            variables=invoke_params,
        )

        report_text: str = chain.with_config(run_name="生成大盘通用分析报告").invoke(messages)

        # 替换商品卡数据
        replaced_text = self._replace_report_product_cards(report_text=report_text, origin_product_resp=top_goods_raw)

        # 构建 Excel 导出数据
        exporter = ZhiyiExcelExporter(
            aggregated_data=state.aggregated_data,
            param=state.api_parse_param,
            is_thinking=state.is_thinking,
        )
        excel_data_list = exporter.build_excel_data_list()

        pusher.push_phase("洞察生成完成")
        pusher.push_report_and_excel_data(
            entity_type=WorkflowEntityType.TAOBAO_ITEM.code,
            report_text=replaced_text,
            excel_data_list=excel_data_list
        )

        return {"report_text": replaced_text}

    def _assemble_output_response(self, state: ZhiyiDeepresearchState):
        """处理生成完成的消息等内容"""
        pusher = state.message_pusher

        pusher.push_task_finish_status_msg()
        return

    # === 非深度思考报告节点 ===

    def _generate_shop_category_nonthinking_report(self, state: ZhiyiDeepresearchState):
        """生成店铺品类分析非深度思考报告"""
        req = state.request
        parse_param = state.api_parse_param
        pusher = state.message_pusher

        pusher.push_phase("正在生成数据洞察报告")

        # 获取LLM
        llm = llm_factory.get_llm(
            LlmProvider.HUANXIN.name,
            LlmModelName.HUANXIN_GEMINI_3_PRO_PREVIEW.value
        )
        chain = llm.with_retry(stop_after_attempt=2) | StrOutputParser()

        # 准备参数（不包含商品卡完整数据）
        invoke_params = {
            "user_query": req.user_query,
            "shop_name": state.target_shop_name,
            "date_range": f"{parse_param.start_date}至{parse_param.end_date}",
            "category_path": f"{parse_param.root_category_id_name}-{parse_param.category_id_name}",
            "sale_volume_trend_data": state.shop_overview_volume.model_dump_json(ensure_ascii=False) if state.shop_overview_volume else "{}",
            "sale_amount_trend_data": state.shop_overview_amount.model_dump_json(ensure_ascii=False) if state.shop_overview_amount else "{}",
            "category_trend_data": state.shop_category_trend.model_dump_json(ensure_ascii=False) if state.shop_category_trend else "{}",
            "price_trend_data": state.shop_price_trend.model_dump_json(ensure_ascii=False) if state.shop_price_trend else "{}",
            "color_trend_data": state.shop_color_data.model_dump_json(ensure_ascii=False) if state.shop_color_data else "{}",
        }

        # 从CozePromptHub获取提示词
        messages = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.DR_ZHIYI_SHOP_NONTHINKING_CATEGORY_REPORT_PROMPT.value,
            variables=invoke_params,
        )

        report_text = chain.with_config(run_name="生成店铺品类非思考报告").invoke(messages)

        # 构建 Excel 导出数据
        exporter = ZhiyiExcelExporter(
            aggregated_data=state.aggregated_data,
            param=state.api_parse_param,
            is_thinking=state.is_thinking,
        )
        excel_data_list = exporter.build_excel_data_list()

        pusher.push_phase("报告已生成")
        pusher.push_report_and_excel_data(
            entity_type=WorkflowEntityType.TAOBAO_ITEM.code,
            report_text=report_text,
            excel_data_list=excel_data_list
        )

        return {"report_text": report_text}

    def _generate_shop_normal_nonthinking_report(self, state: ZhiyiDeepresearchState):
        """生成店铺非品类分析非深度思考报告（属性/价格带）"""
        req = state.request
        parse_param = state.api_parse_param
        pusher = state.message_pusher

        pusher.push_phase("正在生成数据洞察报告")

        # 判断维度
        table_type = parse_param.table_type
        if table_type == "3":
            dimension_data_name = "价格带分布数据"
            dimension_trend_data = state.shop_price_trend
        else:
            dimension_data_name = "属性分布数据"
            dimension_trend_data = state.shop_property_data

        # 获取LLM
        llm = llm_factory.get_llm(
            LlmProvider.HUANXIN.name,
            LlmModelName.HUANXIN_GEMINI_3_PRO_PREVIEW.value
        )
        chain = llm.with_retry(stop_after_attempt=2) | StrOutputParser()

        # 准备参数（不包含商品卡完整数据）
        invoke_params = {
            "user_query": req.user_query,
            "shop_name": state.target_shop_name,
            "date_range": f"{parse_param.start_date}至{parse_param.end_date}",
            "category_path": f"{parse_param.root_category_id_name}-{parse_param.category_id_name}",
            "sale_volume_trend_data": state.shop_overview_volume.model_dump_json(ensure_ascii=False) if state.shop_overview_volume else "{}",
            "sale_amount_trend_data": state.shop_overview_amount.model_dump_json(ensure_ascii=False) if state.shop_overview_amount else "{}",
            "dimension_data_name": dimension_data_name,
            "dimension_trend_data": dimension_trend_data.model_dump_json(ensure_ascii=False) if dimension_trend_data else "{}",
        }

        messages = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.DR_ZHIYI_SHOP_NONTHINKING_NORMAL_REPORT_PROMPT.value,
            variables=invoke_params,
        )

        report_text = chain.with_config(run_name="生成店铺非品类非思考报告").invoke(messages)

        # 构建 Excel 导出数据
        exporter = ZhiyiExcelExporter(
            aggregated_data=state.aggregated_data,
            param=state.api_parse_param,
            is_thinking=state.is_thinking,
        )
        excel_data_list = exporter.build_excel_data_list()

        pusher.push_phase("报告已生成")
        pusher.push_report_and_excel_data(
            entity_type=WorkflowEntityType.TAOBAO_ITEM.code,
            report_text=report_text,
            excel_data_list=excel_data_list
        )

        return {"report_text": report_text}

    def _generate_hydc_category_nonthinking_report(self, state: ZhiyiDeepresearchState):
        """生成大盘品类分析非深度思考报告"""
        req = state.request
        parse_param = state.api_parse_param
        pusher = state.message_pusher

        pusher.push_phase("正在生成数据洞察报告")

        # 获取LLM（大盘使用Flash模型以提高速度）
        llm = llm_factory.get_llm(
            LlmProvider.HUANXIN.name,
            LlmModelName.HUANXIN_GEMINI_3_FLASH_PREVIEW.value
        )
        chain = llm.with_retry(stop_after_attempt=2) | StrOutputParser()

        # 准备参数（不包含商品卡完整数据）
        invoke_params = {
            "user_query": req.user_query,
            "date_range": f"{parse_param.start_date}至{parse_param.end_date}",
            "category_path": f"{parse_param.root_category_id_name}-{parse_param.category_id_name}",
            "sale_volume_trend_data": state.hydc_category_overview_volume.model_dump_json(ensure_ascii=False) if state.hydc_category_overview_volume else "{}",
            "sale_amount_trend_data": state.hydc_category_overview_amount.model_dump_json(ensure_ascii=False) if state.hydc_category_overview_amount else "{}",
            "price_trend_data": state.hydc_category_price_data.model_dump_json(ensure_ascii=False) if state.hydc_category_price_data else "{}",
            "color_trend_data": state.hydc_category_color_data.model_dump_json(ensure_ascii=False) if state.hydc_category_color_data else "{}",
        }

        messages = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.DR_ZHIYI_HYDC_NONTHINKING_CATEGORY_REPORT_PROMPT.value,
            variables=invoke_params,
        )

        report_text = chain.with_config(run_name="生成大盘品类非思考报告").invoke(messages)

        # 构建 Excel 导出数据
        exporter = ZhiyiExcelExporter(
            aggregated_data=state.aggregated_data,
            param=state.api_parse_param,
            is_thinking=state.is_thinking,
        )
        excel_data_list = exporter.build_excel_data_list()

        pusher.push_phase("报告已生成")
        pusher.push_report_and_excel_data(
            entity_type=WorkflowEntityType.TAOBAO_ITEM.code,
            report_text=report_text,
            excel_data_list=excel_data_list
        )

        return {"report_text": report_text}

    def _generate_hydc_normal_nonthinking_report(self, state: ZhiyiDeepresearchState):
        """生成大盘通用分析非深度思考报告（属性/颜色/品牌/价格带）"""
        req = state.request
        parse_param = state.api_parse_param
        pusher = state.message_pusher
        dimension_type = state.hydc_dimension_type
        table_type = parse_param.table_type

        pusher.push_phase("正在生成数据洞察报告")

        # 根据分析类型获取对应数据
        if table_type == "3":  # 价格带分析
            dimension_data_name = "价格带分布数据"
            overview_volume = state.hydc_price_overview_volume
            overview_amount = state.hydc_price_overview_amount
            dimension_data = state.hydc_price_data
        elif dimension_type == 0:  # 属性分析
            dimension_data_name = state.hydc_property_name + "分布数据"
            overview_volume = state.hydc_property_overview_volume
            overview_amount = state.hydc_property_overview_amount
            dimension_data = state.hydc_property_data
        elif dimension_type == 1:  # 颜色分析
            dimension_data_name = "颜色分布数据"
            overview_volume = state.hydc_color_overview_volume
            overview_amount = state.hydc_color_overview_amount
            dimension_data = state.hydc_color_data
        else:  # 品牌分析
            dimension_data_name = "品牌分布数据"
            overview_volume = state.hydc_brand_overview_volume
            overview_amount = state.hydc_brand_overview_amount
            dimension_data = state.hydc_brand_data

        # 获取LLM
        llm = llm_factory.get_llm(
            LlmProvider.HUANXIN.name,
            LlmModelName.HUANXIN_GEMINI_3_PRO_PREVIEW.value
        )
        chain = llm.with_retry(stop_after_attempt=2) | StrOutputParser()

        # 准备参数（不包含商品卡完整数据）
        invoke_params = {
            "user_query": req.user_query,
            "date_range": f"{parse_param.start_date}至{parse_param.end_date}",
            "category_path": f"{parse_param.root_category_id_name}-{parse_param.category_id_name}",
            "sale_volume_trend_data": overview_volume.model_dump_json(ensure_ascii=False) if overview_volume else "{}",
            "sale_amount_trend_data": overview_amount.model_dump_json(ensure_ascii=False) if overview_amount else "{}",
            "dimension_data_name": dimension_data_name,
            "dimension_trend_data": dimension_data.model_dump_json(ensure_ascii=False) if dimension_data else "{}",
        }

        messages = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.DR_ZHIYI_HYDC_NONTHINKING_NORMAL_REPORT_PROMPT.value,
            variables=invoke_params,
        )

        report_text = chain.with_config(run_name="生成大盘通用非思考报告").invoke(messages)

        # 构建 Excel 导出数据
        exporter = ZhiyiExcelExporter(
            aggregated_data=state.aggregated_data,
            param=state.api_parse_param,
            is_thinking=state.is_thinking,
        )
        excel_data_list = exporter.build_excel_data_list()

        pusher.push_phase("报告已生成")
        pusher.push_report_and_excel_data(
            entity_type=WorkflowEntityType.TAOBAO_ITEM.code,
            report_text=report_text,
            excel_data_list=excel_data_list
        )

        return {"report_text": report_text}



zhiyi_thinking_subgraph = ZhiyiDeepresearchGraph()

__all__ = [
    "ZhiyiDeepresearchGraph",
    "zhiyi_thinking_subgraph"
]
