# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2026/1/15 15:17
# @File     : douyi_deepresearch_graph.py
"""
抖衣数据洞察深度思考工作流
"""
import re
from datetime import datetime, timedelta
from enum import StrEnum

from dateutil.relativedelta import relativedelta
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.utils.json_schema import dereference_refs
from langgraph.graph.state import CompiledStateGraph, StateGraph
from loguru import logger

from app.core.clients.coze_loop_client import coze_loop_client_provider
from app.core.config.constants import VolcKnowledgeServiceId, LlmProvider, LlmModelName, CozePromptHubKey, \
    WorkflowEntityType
from app.core.errors import AppException, ErrorCode
from app.core.tools import llm_factory
from app.schemas.response.common import PageResult
from app.service.chains.workflow.base_graph import BaseWorkflowGraph
from app.service.chains.workflow.deepresearch.deepresearch_graph_state import DouyiDeepresearchState
from app.service.chains.workflow.deepresearch.deepresearch_message_pusher import DeepresearchMessagePusher
from app.service.chains.workflow.deepresearch.douyi.schema import (
    DouyiTableType,
    DouyiCategoryFormatItem,
    DouyiParsedCategory,
    DouyiMainParseParam,
    DouyiGoodsRelateType,
    DouyiTrendCleanResponse,
    DouyiTrendSlimResult,
    DouyiTrendSlimItem,
    DouyiPriceCleanResponse,
    DouyiPriceSlimResult,
    DouyiPriceSlimItem,
    DouyiPropertyCleanResponse,
    DouyiPropertySlimResult,
    DouyiPropertySlimItem,
    DouyiTopGoodsCleanResponse,
    DouyiTopGoodsSlimResult,
    DouyiTopGoodsSlimItem,
    DouyiAggregatedData,
)
from app.service.chains.workflow.deepresearch.douyi.llm_output import DouyiExtractedProperty, DouyiExtractedPriceRange
from app.service.chains.workflow.deepresearch.douyi.excel_exporter import DouyiExcelExporter
from app.service.rpc.volcengine_kb_api import get_volcengine_kb_api
from app.service.rpc.zhiyi.client import get_zhiyi_api_client
from app.service.rpc.zhiyi.schemas import (
    DouyiTrendAnalysisRequest,
    DouyiTrendAnalysisRawResponse,
    DouyiTopItemsRequest,
    DouyiPropertySelectorRequest,
    DouyiPropertySelectorRawResponse,
    DouyiGoodsNested, DouyiSearchRequest, DouyiGoodsEntity,
)


class NodeKey(StrEnum):
    # 通用准备
    INIT = "初始化"
    CATEGORY_SEARCH = "类目检索"
    MAIN_PARAM_PARSE = "主要参数解析"

    # 数据查询
    CATEGORY_BRANCH_START = "品类分析分支开始"
    PRICE_BRANCH_START = "价格分析分支开始"
    PROPERTY_BRANCH_START = "属性分析分支开始"

    CATEGORY_CATEGORY_QUERY = "品类分支-品类分析"
    CATEGORY_PRICE_QUERY = "品类分支-价格分析"
    CATEGORY_TOP_GOODS_QUERY = "品类分支-Top商品查询"
    CATEGORY_ANALYZE_AGGREGATE = "品类分支-分析结果聚合"

    PROPERTY_PROPERTY_EXTRACT = "属性分支-属性提取"
    PROPERTY_CATEGORY_QUERY = "属性分支-品类分析"
    PROPERTY_PROPERTY_QUERY = "属性分支-属性分析"
    PROPERTY_TOP_GOODS_QUERY = "属性分支-Top商品查询"
    PROPERTY_ANALYZE_AGGREGATE = "属性分支-分析结果聚合"

    PRICE_CATEGORY_QUERY = "价格分支-品类分析"
    PRICE_PRICE_QUERY = "价格分支-价格分析"
    PRICE_TOP_GOODS_QUERY = "价格分支-Top商品查询"
    PRICE_RANGE_EXTRACT = "价格分支-价格带提取"
    PRICE_ANALYZE_AGGREGATE = "价格分支-分析结果聚合"

    # 报告生成
    NORMAL_REPORT_GENERATE = "通用报告生成"
    NONTHINKING_REPORT_GENERATE = "非深度思考报告生成"

    # 结果输出
    ASSEMBLE_OUTPUT = "组装输出"


class DouyiDeepresearchGraph(BaseWorkflowGraph):
    """
    数据洞察-抖衣数据源-深度思考工作流
    """
    span_name: str = "抖衣数据洞察工作流"

    def __init__(self):
        super().__init__()
        self.compiled_graph = self._build_graph()
        self.product_area_match_prog = re.compile(r'<custom-productcards>(.*?)</custom-productcards>', re.S)

    def _build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(DouyiDeepresearchState)

        # =============定义节点=============
        # 前置准备相关节点
        graph.add_node(NodeKey.INIT, self._init_state_node)
        graph.add_node(NodeKey.CATEGORY_SEARCH, self._category_search_node)
        graph.add_node(NodeKey.MAIN_PARAM_PARSE, self._main_param_parse_node)
        # 品类分析
        graph.add_node(NodeKey.CATEGORY_BRANCH_START, self._query_analyze_data_start)
        graph.add_node(NodeKey.CATEGORY_CATEGORY_QUERY, self._query_category_analyze_data)
        graph.add_node(NodeKey.CATEGORY_PRICE_QUERY, self._query_price_analyze_data)
        graph.add_node(NodeKey.CATEGORY_TOP_GOODS_QUERY, self._query_top_goods_data)
        graph.add_node(NodeKey.CATEGORY_ANALYZE_AGGREGATE, self._agg_analyze_data)
        # 属性分析
        graph.add_node(NodeKey.PROPERTY_BRANCH_START, self._query_analyze_data_start)
        graph.add_node(NodeKey.PROPERTY_PROPERTY_EXTRACT, self._extract_property_node)
        graph.add_node(NodeKey.PROPERTY_CATEGORY_QUERY, self._query_category_analyze_data)
        graph.add_node(NodeKey.PROPERTY_PROPERTY_QUERY, self._query_property_analyze_data)
        graph.add_node(NodeKey.PROPERTY_TOP_GOODS_QUERY, self._query_top_goods_data)
        graph.add_node(NodeKey.PROPERTY_ANALYZE_AGGREGATE, self._agg_analyze_data)
        # 价格带分析
        graph.add_node(NodeKey.PRICE_BRANCH_START, self._query_analyze_data_start)
        graph.add_node(NodeKey.PRICE_RANGE_EXTRACT, self._extract_price_range_node)
        graph.add_node(NodeKey.PRICE_CATEGORY_QUERY, self._query_category_analyze_data)
        graph.add_node(NodeKey.PRICE_PRICE_QUERY, self._query_price_analyze_data)
        graph.add_node(NodeKey.PRICE_TOP_GOODS_QUERY, self._query_top_goods_data)
        graph.add_node(NodeKey.PRICE_ANALYZE_AGGREGATE, self._agg_analyze_data)

        # 生成报告
        graph.add_node(NodeKey.NORMAL_REPORT_GENERATE, self._generate_normal_report)
        graph.add_node(NodeKey.NONTHINKING_REPORT_GENERATE, self._generate_nonthinking_report)

        graph.add_node(NodeKey.ASSEMBLE_OUTPUT, self._assemble_output_node)

        # =============定义入口和边=============
        graph.set_entry_point(NodeKey.INIT)

        # 通用处理
        graph.add_edge(NodeKey.INIT, NodeKey.CATEGORY_SEARCH)
        graph.add_edge(NodeKey.CATEGORY_SEARCH, NodeKey.MAIN_PARAM_PARSE)
        graph.add_conditional_edges(NodeKey.MAIN_PARAM_PARSE, self._route_analyze_dimension, {
            DouyiTableType.CATEGORY_ANALYSIS.code: NodeKey.CATEGORY_BRANCH_START,
            DouyiTableType.PROPERTY_ANALYSIS.code: NodeKey.PROPERTY_BRANCH_START,
            DouyiTableType.PRICE_ANALYSIS.code: NodeKey.PRICE_BRANCH_START,
        })
        # 品类分支数据查询
        graph.add_edge(NodeKey.CATEGORY_BRANCH_START, NodeKey.CATEGORY_CATEGORY_QUERY)
        graph.add_edge(NodeKey.CATEGORY_BRANCH_START, NodeKey.CATEGORY_PRICE_QUERY)
        graph.add_edge(NodeKey.CATEGORY_BRANCH_START, NodeKey.CATEGORY_TOP_GOODS_QUERY)
        graph.add_edge([
            NodeKey.CATEGORY_CATEGORY_QUERY,
            NodeKey.CATEGORY_PRICE_QUERY,
            NodeKey.CATEGORY_TOP_GOODS_QUERY,
        ], NodeKey.CATEGORY_ANALYZE_AGGREGATE)

        # 属性分支数据查询
        # 品类查询和Top商品查询并行执行，属性提取后再执行属性查询
        graph.add_edge(NodeKey.PROPERTY_BRANCH_START, NodeKey.PROPERTY_CATEGORY_QUERY)
        graph.add_edge(NodeKey.PROPERTY_BRANCH_START, NodeKey.PROPERTY_TOP_GOODS_QUERY)
        graph.add_edge(NodeKey.PROPERTY_BRANCH_START, NodeKey.PROPERTY_PROPERTY_EXTRACT)
        graph.add_edge(NodeKey.PROPERTY_PROPERTY_EXTRACT, NodeKey.PROPERTY_PROPERTY_QUERY)
        graph.add_edge([
            NodeKey.PROPERTY_CATEGORY_QUERY,
            NodeKey.PROPERTY_PROPERTY_QUERY,
            NodeKey.PROPERTY_TOP_GOODS_QUERY,
        ], NodeKey.PROPERTY_ANALYZE_AGGREGATE)

        # 价格带分支数据查询
        # 品类查询和Top商品查询并行执行，价格带提取后再执行价格查询
        graph.add_edge(NodeKey.PRICE_BRANCH_START, NodeKey.PRICE_CATEGORY_QUERY)
        graph.add_edge(NodeKey.PRICE_BRANCH_START, NodeKey.PRICE_TOP_GOODS_QUERY)
        graph.add_edge(NodeKey.PRICE_BRANCH_START, NodeKey.PRICE_RANGE_EXTRACT)
        graph.add_edge(NodeKey.PRICE_RANGE_EXTRACT, NodeKey.PRICE_PRICE_QUERY)
        graph.add_edge([
            NodeKey.PRICE_CATEGORY_QUERY,
            NodeKey.PRICE_PRICE_QUERY,
            NodeKey.PRICE_TOP_GOODS_QUERY,
        ], NodeKey.PRICE_ANALYZE_AGGREGATE)

        # 报告生成
        graph.add_conditional_edges(NodeKey.CATEGORY_ANALYZE_AGGREGATE, self._route_if_thinking_report, {
            True: NodeKey.NORMAL_REPORT_GENERATE, False: NodeKey.NONTHINKING_REPORT_GENERATE,
        })
        graph.add_conditional_edges(NodeKey.PROPERTY_ANALYZE_AGGREGATE, self._route_if_thinking_report, {
            True: NodeKey.NORMAL_REPORT_GENERATE, False: NodeKey.NONTHINKING_REPORT_GENERATE,
        })
        graph.add_conditional_edges(NodeKey.PRICE_ANALYZE_AGGREGATE, self._route_if_thinking_report, {
            True: NodeKey.NORMAL_REPORT_GENERATE, False: NodeKey.NONTHINKING_REPORT_GENERATE,
        })

        graph.add_edge(NodeKey.NONTHINKING_REPORT_GENERATE, NodeKey.ASSEMBLE_OUTPUT)
        graph.add_edge(NodeKey.NORMAL_REPORT_GENERATE, NodeKey.ASSEMBLE_OUTPUT)
        graph.set_finish_point(NodeKey.ASSEMBLE_OUTPUT)

        return graph.compile()

    # ===节点实现===

    def _init_state_node(self, state: DouyiDeepresearchState):
        """开始节点，处理初始化"""
        req = state.request
        is_thinking = req.thinking if req.thinking else False

        # 创建消息推送器
        pusher = DeepresearchMessagePusher(
            request=req,
            is_thinking=is_thinking,
            workflow_type="douyi",
        )

        pusher.push_task_start_msg()

        # 非深度思考的推送
        if not is_thinking:
            pusher.push_phase("正在启用趋势分析助手")

        return {
            "is_thinking": is_thinking,
            "message_pusher": pusher,
        }

    def _category_search_node(self, state: DouyiDeepresearchState):
        """检索分析类目"""
        req = state.request

        # 向量搜索
        kb_client = get_volcengine_kb_api()
        resp = kb_client.simple_chat(
            query=req.user_query,
            service_resource_id=VolcKnowledgeServiceId.DOUYI_CATEGORY_VECTOR.value
        )
        parse_content_list: list[dict] = kb_client.parse_structure_chat_response(resp)

        # 结构处理
        parsed_category_list = []
        for content in parse_content_list:
            category_name_path: str = content['key']  # format: '女装,风衣'
            category_id_path: str = content['value']  # format: '16,50008901'
            category_id_arr = category_id_path.split(',')
            category_name_arr = category_name_path.split(',')
            root_category_id = category_id_arr[0] if len(category_id_arr) >= 1 else None
            leaf_category_id = category_id_arr[1] if len(category_id_arr) >= 2 else None
            root_category_name = category_name_arr[0] if len(category_name_arr) >= 1 else None
            leaf_category_name = category_name_arr[1] if len(category_name_arr) >= 2 else None

            item = DouyiCategoryFormatItem(
                root_category_id=root_category_id,
                category_id=leaf_category_id,
                root_category_id_name=root_category_name,
                category_name=leaf_category_name,
            )
            parsed_category_list.append(item)


        return {
            "recall_category": DouyiParsedCategory(category_list=parsed_category_list)
        }

    def _main_param_parse_node(self, state: DouyiDeepresearchState):
        """主要参数解析节点"""
        req = state.request
        is_thinking = state.is_thinking
        recall_category = state.recall_category
        pusher = state.message_pusher

        # 获取llm
        llm = llm_factory.get_llm(LlmProvider.HUANXIN.name, LlmModelName.HUANXIN_GEMINI_3_FLASH_PREVIEW.value)
        struct_llm = llm.with_structured_output(schema=DouyiMainParseParam).with_retry(stop_after_attempt=2)

        # 计算日期相关参数
        today = datetime.now()
        t_minus_1 = today - timedelta(days=1)
        last_month_end = today.replace(day=1) - timedelta(days=1)
        six_months_ago_first = (today - relativedelta(months=6)).replace(day=1)
        recent_30day_start = t_minus_1 - timedelta(days=29)  # 近30天开始日期

        invoke_params = {
            "current_date": t_minus_1.strftime("%Y-%m-%d"),
            "default_end_date": last_month_end.strftime("%Y-%m-%d"),
            "default_start_date": six_months_ago_first.strftime("%Y-%m-%d"),
            "recent_30day_start_date": recent_30day_start.strftime("%Y-%m-%d"),
            "user_query": req.user_query,
            "industry": req.industry,
            "category_list": recall_category.model_dump_json(ensure_ascii=False) if recall_category else "[]",
        }
        messages: list[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.DR_DOUYI_MAIN_PARAM_PARSE_PROMPT.value,
            variables=invoke_params,
        )
        # 执行llm
        parse_param: DouyiMainParseParam = struct_llm.with_config(run_name="解析主要参数").invoke(messages)

        # 分析商品/销量类型
        relate_type = None
        target_sale_volume = None
        target_sale_amount = None
        if parse_param.frontTitleTabValue == DouyiGoodsRelateType.WINDOW_GOODS.code:
            relate_type = DouyiGoodsRelateType.WINDOW_GOODS.code
            target_sale_volume = "saleVolume"
            target_sale_amount = "saleAmount"
        elif parse_param.frontTitleTabValue == DouyiGoodsRelateType.LIVE_GOODS.code:
            relate_type = DouyiGoodsRelateType.LIVE_GOODS.code
            target_sale_volume = "saleVolume"
            target_sale_amount = "saleAmount"
        elif parse_param.frontTitleTabValue == DouyiGoodsRelateType.VIDEO_GOODS.code:
            relate_type = DouyiGoodsRelateType.WINDOW_GOODS.code
            target_sale_volume = "videoSaleVolume"
            target_sale_amount = "videoSaleAmount"
        elif parse_param.frontTitleTabValue == DouyiGoodsRelateType.CARD_GOODS.code:
            relate_type = DouyiGoodsRelateType.WINDOW_GOODS.code
            target_sale_volume = "cardSaleVolume"
            target_sale_amount = "cardSaleAmount"
        else:
            logger.error(f"不受支持的frontTitleTabValue类型: {parse_param.frontTitleTabValue}")
            raise AppException(ErrorCode.WORKFLOW_ERROR, f"不受支持的frontTitleTabValue类型: {parse_param.frontTitleTabValue}")

        # 深度思考的推送
        if is_thinking:
            pusher.push_phase("数据收集", variables={"industry_name": parse_param.root_category_id_name})

        return {
            "main_parse_param": parse_param,
            "goods_relate_type": relate_type,
            "target_sale_volume": target_sale_volume,
            "target_sale_amount": target_sale_amount,
        }

    def _route_analyze_dimension(self, state: DouyiDeepresearchState) -> str:
        """路由分析维度 - 根据 table_type 返回对应分支标识"""
        if state.main_parse_param:
            return state.main_parse_param.table_type
        return DouyiTableType.CATEGORY_ANALYSIS.code  # 默认品类分析

    def _query_analyze_data_start(self, state: DouyiDeepresearchState):
        """分支开始节点 - 准备查询参数"""
        return {}

    def _extract_price_range_node(self, state: DouyiDeepresearchState):
        """价格带提取节点 - 从用户问题中提取价格范围"""
        req = state.request
        extracted = self._extract_price_range_with_llm(req.user_query)
        return {"extracted_price_range": extracted}

    def _extract_property_node(self, state: DouyiDeepresearchState):
        """属性提取节点 - 获取可用属性列表并使用 LLM 提取属性"""
        req = state.request
        param = state.main_parse_param
        if not param:
            return {}

        api_client = get_zhiyi_api_client()

        # 1. 先获取可用属性列表
        property_selector_request = DouyiPropertySelectorRequest(
            rootCategoryId=param.root_category_id,
            categoryId=param.category_id,
        )
        selector_resp = None
        available_properties_json = "[]"
        try:
            selector_resp = api_client.get_douyi_property_selector(
                user_id=req.user_id,
                team_id=req.team_id,
                params=property_selector_request,
            )
            if selector_resp and selector_resp.result:
                available_properties_json = selector_resp.model_dump_json(ensure_ascii=False)
        except Exception as e:
            logger.warning(f"获取属性选择器失败: {e}")

        # 2. 使用LLM从用户问题中提取属性
        extracted_property = self._extract_property_with_llm(
            user_query=req.user_query,
            available_properties_json=available_properties_json,
            selector_resp=selector_resp,
        )

        return {
            "extracted_property": extracted_property,
            "available_property_list": selector_resp.model_dump_json(ensure_ascii=False) if selector_resp else None,
        }

    # ===数据查询节点===

    def _query_category_analyze_data(self, state: DouyiDeepresearchState):
        """查询品类分析数据（销量+销售额趋势）"""
        req = state.request
        param = state.main_parse_param
        relate_type = state.goods_relate_type
        target_sale_volume = state.target_sale_volume
        target_sale_amount = state.target_sale_amount
        if not param:
            return {}

        api_client = get_zhiyi_api_client()

        # 构建品类ID列表
        category_id_list = [param.category_id] if param.category_id else None

        # 查询销量趋势
        volume_request = DouyiTrendAnalysisRequest(
            queryType="categoryAnalysis",
            rootCategoryId=param.root_category_id,
            categoryIdList=category_id_list,
            startDate=param.start_date,
            endDate=param.end_date,
            trendType=param.date_type or "monthly",
            frontTitleTabValue=param.frontTitleTabValue,
            relateType=relate_type,
            target=target_sale_volume,
        )
        volume_resp = api_client.get_douyi_trend_analysis(
            user_id=req.user_id,
            team_id=req.team_id,
            params=volume_request,
        )

        # 查询销售额趋势
        amount_request = DouyiTrendAnalysisRequest(
            queryType="categoryAnalysis",
            rootCategoryId=param.root_category_id,
            categoryIdList=category_id_list,
            startDate=param.start_date,
            endDate=param.end_date,
            trendType=param.date_type or "monthly",
            frontTitleTabValue=param.frontTitleTabValue,
            relateType=relate_type,
            target=target_sale_amount,
        )
        amount_resp = api_client.get_douyi_trend_analysis(
            user_id=req.user_id,
            team_id=req.team_id,
            params=amount_request,
        )

        # 清洗数据
        volume_clean = self._clean_trend_data(volume_resp)
        amount_clean = self._clean_trend_data(amount_resp)

        # 根据当前分支返回不同的state字段
        table_type = param.table_type
        if table_type == DouyiTableType.CATEGORY_ANALYSIS.code:
            return {
                "category_volume_trend": volume_clean,
                "category_amount_trend": amount_clean,
            }
        elif table_type == DouyiTableType.PROPERTY_ANALYSIS.code:
            return {
                "property_volume_trend": volume_clean,
                "property_amount_trend": amount_clean,
            }
        else:
            return {
                "price_volume_trend": volume_clean,
                "price_amount_trend": amount_clean,
            }

    def _query_price_analyze_data(self, state: DouyiDeepresearchState):
        """查询价格带分析数据"""
        req = state.request
        param = state.main_parse_param
        relate_type = state.goods_relate_type
        target_sale_volume = state.target_sale_volume
        target_sale_amount = state.target_sale_amount
        if not param:
            return {}

        api_client = get_zhiyi_api_client()

        # 构建品类ID列表
        category_id_list = [param.category_id] if param.category_id else None

        # 根据分支类型决定是否传入priceRangeList
        table_type = param.table_type
        price_range_list = None
        if table_type == DouyiTableType.PRICE_ANALYSIS.code and state.extracted_price_range:
            # 价格带分支：使用LLM提取的价格范围
            price_range_list = state.extracted_price_range.to_api_format()

        # 查询价格带数据
        price_request = DouyiTrendAnalysisRequest(
            queryType="priceAnalysis",
            rootCategoryId=param.root_category_id,
            categoryIdList=category_id_list,
            startDate=param.start_date,
            endDate=param.end_date,
            trendType=param.date_type or "monthly",
            frontTitleTabValue=param.frontTitleTabValue,
            relateType=relate_type,
            target=target_sale_volume,
            priceRangeList=price_range_list,
        )
        price_resp = api_client.get_douyi_trend_analysis(
            user_id=req.user_id,
            team_id=req.team_id,
            params=price_request,
        )

        # 清洗数据
        price_clean = self._clean_price_data(price_resp)

        # 根据当前分支返回不同的state字段
        if table_type == DouyiTableType.CATEGORY_ANALYSIS.code:
            return {"category_price_data": price_clean}
        else:
            return {"price_price_data": price_clean}

    def _query_property_analyze_data(self, state: DouyiDeepresearchState):
        """查询属性分析数据"""
        req = state.request
        param = state.main_parse_param
        relate_type = state.goods_relate_type
        target_sale_volume = state.target_sale_volume
        target_sale_amount = state.target_sale_amount
        if not param or not state.extracted_property:
            return {}

        api_client = get_zhiyi_api_client()

        # 查询属性分布数据
        category_id_list = [param.category_id] if param.category_id else None
        property_request = DouyiTrendAnalysisRequest(
            queryType="propertyAnalysis",
            rootCategoryId=param.root_category_id,
            categoryIdList=category_id_list,
            startDate=param.start_date,
            endDate=param.end_date,
            trendType=param.date_type or "monthly",
            frontTitleTabValue="windowGoods",
            relateType=relate_type,
            target=target_sale_volume,
            # 属性参数 - 从 state 中获取
            frontProperty=state.extracted_property.front_property,
            properties=state.extracted_property.properties,
        )
        property_resp = api_client.get_douyi_trend_analysis(
            user_id=req.user_id,
            team_id=req.team_id,
            params=property_request,
        )

        # 清洗属性数据，传入提取的属性名
        property_clean = self._clean_property_data(
            property_resp,
            property_name=state.extracted_property.front_property[0] if state.extracted_property.front_property else None
        )

        return {"property_data": property_clean}

    def _extract_property_with_llm(
        self,
        user_query: str,
        available_properties_json: str,
        selector_resp: DouyiPropertySelectorRawResponse | None,
    ) -> DouyiExtractedProperty:
        """使用LLM从用户问题中提取属性"""
        # 获取LLM
        llm = llm_factory.get_llm(LlmProvider.HUANXIN.name, LlmModelName.HUANXIN_GEMINI_3_FLASH_PREVIEW.value)
        struct_llm = llm.with_structured_output(schema=DouyiExtractedProperty).with_retry(stop_after_attempt=2)

        invoke_params = {
            "user_query": user_query,
            "available_properties": available_properties_json,
        }

        try:
            messages: list[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
                prompt_key=CozePromptHubKey.DR_DOUYI_PROPERTY_EXTRACT_PROMPT.value,
                variables=invoke_params,
            )
            result: DouyiExtractedProperty = struct_llm.with_config(run_name="提取属性参数").invoke(messages)
            return result
        except Exception as e:
            logger.warning(f"LLM属性提取失败: {e}，使用默认属性")
            # 回退：使用第一个可用属性
            default_property = self._get_default_property(selector_resp)
            return default_property

    def _get_default_property(self, selector_resp: DouyiPropertySelectorRawResponse | None) -> DouyiExtractedProperty:
        """获取默认属性（第一个可用属性的所有值）"""
        if not selector_resp or not selector_resp.result:
            return DouyiExtractedProperty(front_property=["颜色"], properties=[])

        first_prop = selector_resp.result[0]
        property_name = first_prop.parameter_name or "颜色"

        # 构建属性值列表
        properties = []
        if first_prop.parameter_values:
            for val in first_prop.parameter_values:
                if val.property_value_name:
                    properties.append(f"{property_name}:{val.property_value_name}")

        return DouyiExtractedProperty(
            front_property=[property_name],
            properties=properties
        )

    def _extract_price_range_with_llm(self, user_query: str) -> DouyiExtractedPriceRange:
        """使用LLM从用户问题中提取价格带"""
        # gemini不支持嵌套的默认schema定义，需要展平
        schema = DouyiExtractedPriceRange.model_json_schema()
        flat_schema = dereference_refs(schema)
        flat_schema.pop("$defs", None)
        flat_schema.pop("definitions", None)
        # 获取LLM
        llm = llm_factory.get_llm(LlmProvider.HUANXIN.name, LlmModelName.HUANXIN_GEMINI_3_FLASH_PREVIEW.value)
        struct_llm = llm.with_structured_output(schema=flat_schema).with_retry(stop_after_attempt=2)

        invoke_params = {
            "user_query": user_query,
        }

        try:
            messages: list[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
                prompt_key=CozePromptHubKey.DR_DOUYI_PRICE_EXTRACT_PROMPT.value,
                variables=invoke_params,
            )
            result: DouyiExtractedPriceRange = struct_llm.with_config(run_name="提取价格带参数").invoke(messages)
            return result
        except Exception as e:
            logger.warning(f"LLM价格带提取失败: {e}，使用默认价格带")
            # 回退：返回默认价格带
            return self._get_default_price_range()

    def _get_default_price_range(self) -> DouyiExtractedPriceRange:
        """获取默认价格带"""
        from app.service.chains.workflow.deepresearch.douyi.llm_output import DouyiPriceRangeItem
        default_ranges = [
            DouyiPriceRangeItem(min_price=0, max_price=5000),
            DouyiPriceRangeItem(min_price=5000, max_price=10000),
            DouyiPriceRangeItem(min_price=10000, max_price=20000),
            DouyiPriceRangeItem(min_price=20000, max_price=50000),
            DouyiPriceRangeItem(min_price=50000, max_price=100000),
            DouyiPriceRangeItem(min_price=100000, max_price=150000),
            DouyiPriceRangeItem(min_price=150000, max_price=None),
        ]
        return DouyiExtractedPriceRange(price_ranges=default_ranges)

    def _query_top_goods_data(self, state: DouyiDeepresearchState):
        """查询Top商品数据"""
        req = state.request
        param = state.main_parse_param
        if not param:
            return {}

        api_client = get_zhiyi_api_client()

        # 构建品类ID列表
        category_id_list = [param.category_id] if param.category_id else None

        # 查询Top商品
        goods_request = DouyiSearchRequest(
            pageNo=1,
            pageSize=10,
            limit=10,
            rootCategoryId=param.root_category_id,
            categoryIdList=category_id_list,
            sortStartDate=param.start_date,
            sortEndDate=param.end_date,
            sortDateType="month",
            sortField="saleVolumeDaily",
            sortType="desc",
            firstRecordDateType="recent",
            queryType="GoodsLibraryHotSale"
        )

        goods_resp: PageResult[DouyiGoodsEntity] = api_client.common_search_douyi(
            user_id=req.user_id,
            team_id=req.team_id,
            params=goods_request
        )

        # 清洗数据
        top_clean = self._clean_top_goods_data(goods_resp)

        # 根据当前分支返回不同的state字段
        table_type = param.table_type
        if table_type == DouyiTableType.CATEGORY_ANALYSIS.code:
            return {"category_top_goods": top_clean, "category_top_resp": goods_resp}
        elif table_type == DouyiTableType.PROPERTY_ANALYSIS.code:
            return {"property_top_goods": top_clean, "property_top_resp": goods_resp}
        else:
            return {"price_top_goods": top_clean, "price_top_resp": goods_resp}

    # ===数据清洗方法===

    def _clean_trend_data(self, raw_response: DouyiTrendAnalysisRawResponse) -> DouyiTrendCleanResponse:
        """清洗趋势数据 - 对应n8n的Code in JavaScript"""
        if not raw_response.result or not raw_response.result.sub_list:
            return DouyiTrendCleanResponse(
                success=raw_response.success,
                result=DouyiTrendSlimResult()
            )

        # 提取第一个分类的数据
        category_data = raw_response.result.sub_list[0] if raw_response.result.sub_list else None
        if not category_data:
            return DouyiTrendCleanResponse(success=True, result=DouyiTrendSlimResult())

        # 转换时间周期数据
        trend_items = []
        if category_data.sub_list:
            for item in category_data.sub_list:
                trend_items.append(DouyiTrendSlimItem(
                    granularityDate=item.key,
                    dailySumDaySalesVolume=item.sum or 0,
                    dailySumDaySale=item.live_sum or 0,  # 映射到销售额
                    dailyLiveSum=item.live_sum or 0,
                    dailyVideoSum=item.video_sum or 0,
                    dailyCardSum=item.card_sum or 0,
                ))

        result = DouyiTrendSlimResult(
            sumSalesVolume=category_data.sum,
            sumSale=raw_response.result.sum,
            top3CategoryList=[category_data.name] if category_data.name else [],
            hasShuang11Presale=False,
            trendDTOS=trend_items,
        )

        return DouyiTrendCleanResponse(success=True, result=result)

    def _clean_price_data(self, raw_response: DouyiTrendAnalysisRawResponse) -> DouyiPriceCleanResponse:
        """清洗价格带数据 - 对应n8n的Code in JavaScript1"""
        if not raw_response.result or not raw_response.result.sub_list:
            return DouyiPriceCleanResponse(
                success=raw_response.success,
                result=DouyiPriceSlimResult()
            )

        price_items = []
        for item in raw_response.result.sub_list:
            # 解析价格区间字符串 (例如 "5000-10000" 或 "150000-")
            price_key = item.key or ""
            left_price = 0
            right_price = None

            if "-" in price_key:
                parts = price_key.split("-")
                try:
                    left_price = int(parts[0]) if parts[0] else 0
                    right_price = int(parts[1]) if len(parts) > 1 and parts[1] else None
                except ValueError:
                    pass

            price_items.append(DouyiPriceSlimItem(
                leftPrice=left_price,
                rightPrice=right_price,
                salesVolume=item.sum or 0,
                rate=item.percentage,
            ))

        result = DouyiPriceSlimResult(price_slim=price_items)
        return DouyiPriceCleanResponse(success=True, result=result)

    def _clean_property_data(
        self,
        raw_response: DouyiTrendAnalysisRawResponse,
        property_name: str | None = None,
    ) -> DouyiPropertyCleanResponse:
        """清洗属性分布数据"""
        if not raw_response.result or not raw_response.result.sub_list:
            return DouyiPropertyCleanResponse(
                success=raw_response.success,
                result=DouyiPropertySlimResult()
            )

        property_items = []
        for item in raw_response.result.sub_list:
            property_items.append(DouyiPropertySlimItem(
                propertyName=property_name,
                propertyValue=item.name,
                salesVolume=item.sum or 0,
                salesAmount=None,
                rate=item.percentage,
                otherFlag=False,
            ))

        result = DouyiPropertySlimResult(property_slim=property_items)
        return DouyiPropertyCleanResponse(success=True, result=result)

    def _clean_top_goods_data(self, raw_response: PageResult[DouyiGoodsEntity]) -> DouyiTopGoodsCleanResponse:
        """清洗Top商品数据 - 对应n8n的Code in JavaScript2，参照JS清洗代码逻辑"""
        if not raw_response.result_list:
            return DouyiTopGoodsCleanResponse(
                success=True,
                result=DouyiTopGoodsSlimResult()
            )

        top_items = []
        for idx, item in enumerate(raw_response.result_list[:10]):
            # 获取抖音商品嵌套数据
            douyin_data = item.douyin_goods or DouyiGoodsNested()

            # 构建品类路径
            category_detail = f"{item.root_category_name}=>{item.category_name}" if item.root_category_name and item.category_name else item.category_name

            # 参照 JS 清洗代码逻辑构建数据
            top_items.append(DouyiTopGoodsSlimItem(
                itemId=item.item_id,
                # 标题：优先 title，否则 goods_title
                title=item.title or item.goods_title,
                goodsTitle=item.goods_title,
                picUrl=item.pic_url,
                imageEntityExtend=None,  # JSON中可能有，但通常不需要
                categoryName=item.category_name,
                categoryDetail=category_detail,
                # 价格：couponCprice 优先级 periodLatestCPrice > goodsPrice > cprice
                couponCprice=item.period_latest_c_price or item.goods_price or item.cprice,
                cprice=item.cprice or item.goods_price,
                goodsPrice=item.goods_price,
                minPrice=str(item.min_price) if item.min_price else None,
                maxSPrice=item.sprice,
                # 店铺信息
                shopId=item.shop_id,
                shopName=item.shop_name,
                brand=item.brand,
                # 时间
                firstRecordTime=item.first_record_time,
                # 销量数据
                totalSaleVolume=item.total_sale_volume,
                saleVolume30day=item.sale_volume_30day,
                newSaleVolume=item.new_sale_volume,
                # 销量/销售额（优先使用周期数据）
                salesVolume=item.new_sale_volume or item.sale_volume or 0,
                salesAmount=item.new_sale_amount or item.sale_amount or 0,
                newSaleAmount=item.new_sale_amount,
                # 互动数据：commentCount 优先 totalCommentNum > douyinGoods.totalViewNum
                commentCount=item.total_comment_num or (douyin_data.total_view_num if douyin_data else 0) or 0,
                commentNum30day=douyin_data.comment_num_30day if douyin_data else 0,
                relateLiveNum=douyin_data.relate_live_num if douyin_data else 0,
                relateProductNum=douyin_data.relate_product_num if douyin_data else 0,
            ))

        result = DouyiTopGoodsSlimResult(
            start=raw_response.start or 0,
            pageSize=raw_response.page_size or 10,
            resultCount=len(raw_response.result_list),
            top10_slim=top_items,
        )
        return DouyiTopGoodsCleanResponse(success=True, result=result)

    # ===聚合节点===

    def _agg_analyze_data(self, state: DouyiDeepresearchState):
        """聚合分析数据 - 根据当前分支聚合对应的数据"""
        param = state.main_parse_param
        if not param:
            return {}

        table_type = param.table_type

        # 根据分支类型聚合数据
        if table_type == DouyiTableType.CATEGORY_ANALYSIS.code:
            # 品类分析分支
            aggregated = DouyiAggregatedData(
                table_type=table_type,
                shopSalesVolume=state.category_volume_trend,
                shopSalesAmount=state.category_amount_trend,
                priceData=state.category_price_data,
                propertyData=None,
                top10Products=state.category_top_goods,
                topProductsOrigin=state.category_top_resp,
            )
        elif table_type == DouyiTableType.PROPERTY_ANALYSIS.code:
            # 属性分析分支
            aggregated = DouyiAggregatedData(
                table_type=table_type,
                shopSalesVolume=state.property_volume_trend,
                shopSalesAmount=state.property_amount_trend,
                priceData=None,
                propertyData=state.property_data,
                top10Products=state.property_top_goods,
                topProductsOrigin=state.property_top_resp,
            )
        else:
            # 价格带分析分支
            aggregated = DouyiAggregatedData(
                table_type=table_type,
                shopSalesVolume=state.price_volume_trend,
                shopSalesAmount=state.price_amount_trend,
                priceData=state.price_price_data,
                propertyData=None,
                top10Products=state.price_top_goods,
                topProductsOrigin=state.price_top_resp,
            )

        return {"aggregated_data": aggregated}

    # ===报告生成节点===
    def _route_if_thinking_report(self, state: DouyiDeepresearchState) -> bool:
        return state.is_thinking

    def _generate_normal_report(self, state: DouyiDeepresearchState):
        """生成分析报告 - 使用统一的报告生成prompt"""
        req = state.request
        param = state.main_parse_param
        aggregated_data = state.aggregated_data
        pusher = state.message_pusher

        if not param or not aggregated_data:
            return {"report_text": "数据不足，无法生成报告。"}
        pusher.push_phase("洞察生成中")

        # 深度思考模式：添加报告生成步骤
        dimension_type = DouyiTableType.from_code(param.table_type)
        match dimension_type:
            case DouyiTableType.CATEGORY_ANALYSIS:
                dimension_data_name = "价格带分布数据"
                dimension_trend_data = aggregated_data.price_data
            case DouyiTableType.PROPERTY_ANALYSIS:
                dimension_data_name = "属性分布数据"
                dimension_trend_data = aggregated_data.property_data
            case DouyiTableType.PRICE_ANALYSIS:
                dimension_data_name = "价格带分布数据"
                dimension_trend_data = aggregated_data.price_data


        # 获取LLM
        llm = llm_factory.get_llm(LlmProvider.HUANXIN.name, LlmModelName.HUANXIN_GEMINI_3_PRO_PREVIEW.value)
        chain = llm.with_retry(stop_after_attempt=2) | StrOutputParser()

        # 准备报告生成参数
        invoke_params = {
            # 基础数据
            "user_query": req.user_query,
            "date_range": f"{param.start_date}至{param.end_date}",
            "category_path": f"{param.root_category_id_name}-{param.category_id_name}",
            # 分析结果数据
            "sale_volume_trend_data": aggregated_data.sale_volume_data.model_dump_json(ensure_ascii=False) if aggregated_data.sale_volume_data else "{}",
            "sale_amount_trend_data": aggregated_data.sale_amount_data.model_dump_json(ensure_ascii=False) if aggregated_data.sale_amount_data else "{}",
            "dimension_data_name": dimension_data_name,
            "dimension_trend_data": dimension_trend_data.model_dump_json(ensure_ascii=False) if dimension_trend_data else "{}",
            "price_data": aggregated_data.price_data.model_dump_json(ensure_ascii=False) if aggregated_data.price_data else "{}",
            "property_data": aggregated_data.property_data.model_dump_json(ensure_ascii=False) if aggregated_data.property_data else "{}",
            "top10_products": aggregated_data.top10_products.model_dump_json(ensure_ascii=False) if aggregated_data.top10_products else "{}",
        }

        messages: list[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.DR_DOUYI_REPORT_GENERATE_PROMPT.value,
            variables=invoke_params,
        )

        # 执行LLM生成报告
        report_text = chain.with_config(run_name="生成分析报告").invoke(messages)

        # 替换top商品卡内容为原始请求结果
        replaced_text = self._replace_report_product_cards(report_text=report_text, origin_product_resp=aggregated_data.top_products_origin)

        # 深度思考模式：完成洞察生成
        pusher.push_phase("洞察生成完成")

        return {"report_text": replaced_text}

    def _generate_nonthinking_report(self, state: DouyiDeepresearchState):
        req = state.request
        param = state.main_parse_param
        aggregated_data = state.aggregated_data
        pusher = state.message_pusher

        pusher.push_phase("正在生成数据洞察报告")

        # 获取LLM
        llm = llm_factory.get_llm(LlmProvider.HUANXIN.name, LlmModelName.HUANXIN_GEMINI_3_PRO_PREVIEW.value)
        chain = llm.with_retry(stop_after_attempt=2) | StrOutputParser()

        dimension_type = DouyiTableType.from_code(param.table_type)
        match dimension_type:
            case DouyiTableType.CATEGORY_ANALYSIS:
                dimension_data_name = "价格带分布数据"
                dimension_trend_data = aggregated_data.price_data
            case DouyiTableType.PROPERTY_ANALYSIS:
                dimension_data_name = "属性分布数据"
                dimension_trend_data = aggregated_data.property_data
            case DouyiTableType.PRICE_ANALYSIS:
                dimension_data_name = "价格带分布数据"
                dimension_trend_data = aggregated_data.price_data

        # 准备报告生成参数
        invoke_params = {
            # 基础数据
            "date_range": f"{param.start_date}至{param.end_date}",
            "category_path": f"{param.root_category_id_name}-{param.category_id_name}",
            # 分析结果数据
            "sale_volume_trend_data": aggregated_data.sale_volume_data.model_dump_json(ensure_ascii=False) if aggregated_data.sale_volume_data else "{}",
            "sale_amount_trend_data": aggregated_data.sale_amount_data.model_dump_json(ensure_ascii=False) if aggregated_data.sale_amount_data else "{}",
            "dimension_data_name": dimension_data_name,
            "dimension_trend_data": dimension_trend_data.model_dump_json(ensure_ascii=False) if dimension_trend_data else "{}",
        }

        messages: list[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.DR_DOUYI_NONTHINKING_REPORT_GENERATE_PROMPT.value,
            variables=invoke_params,
        )

        # 执行LLM生成报告
        report_text = chain.with_config(run_name="生成非思考分析报告").invoke(messages)

        pusher.push_phase("报告已生成")

        return {
            "report_text": report_text,
        }

    def _replace_report_product_cards(self, report_text: str, origin_product_resp: PageResult[DouyiGoodsEntity]) -> str:
        new_product_card_content = f"<custom-productcards>\n{origin_product_resp.model_dump_json(by_alias=True, ensure_ascii=False)}\n</custom-productcards>"
        replaced_text = self.product_area_match_prog.sub(new_product_card_content, report_text)
        return replaced_text

    # ===输出组装节点===

    def _assemble_output_node(self, state: DouyiDeepresearchState):
        """组装最终输出 - 包括 Excel 导出"""
        req = state.request
        is_thinking = state.is_thinking
        param = state.main_parse_param
        aggregated_data = state.aggregated_data
        report_text = state.report_text
        pusher = state.message_pusher

        # 构建 Excel 导出数据
        exporter = DouyiExcelExporter(aggregated_data, param, is_thinking)
        excel_data_list = exporter.build_excel_data_list()
        pusher.push_report_and_excel_data(entity_type=WorkflowEntityType.DY_ITEM.code, report_text=report_text, excel_data_list=excel_data_list)

        pusher.push_task_finish_status_msg()

        return {}


douyi_deepresearch_graph = DouyiDeepresearchGraph()
__all__ = [
    "DouyiDeepresearchGraph",
    "douyi_deepresearch_graph"
]
