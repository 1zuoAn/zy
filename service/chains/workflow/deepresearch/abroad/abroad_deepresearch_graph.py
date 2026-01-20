# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2026/1/15 22:48
# @File     : abroad_deepresearch_graph.py
"""
海外探款数据洞察深度思考工作流
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from enum import StrEnum

from dateutil.relativedelta import relativedelta
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph.state import CompiledStateGraph, StateGraph
from loguru import logger

from app.core.clients.coze_loop_client import coze_loop_client_provider
from app.core.config.constants import CozePromptHubKey, LlmProvider, LlmModelName, VolcKnowledgeServiceId, \
    WorkflowEntityType
from app.core.tools import llm_factory
from app.service.chains.workflow.base_graph import BaseWorkflowGraph
from app.service.chains.workflow.deepresearch.deepresearch_graph_state import AbroadDeepresearchState
from app.service.chains.workflow.deepresearch.deepresearch_message_pusher import DeepresearchMessagePusher
from app.service.chains.workflow.deepresearch.abroad.schema import (
    AbroadMainParseParam,
    AbroadPlatformType,
    AbroadDimensionType,
    AbroadCategoryFormatItem,
    AbroadParsedCategory,
    parse_abroad_category_list,
    AbroadTrendCleanResponse,
    AbroadTrendSlimResult,
    AbroadTrendSlimItem,
    AbroadDimensionInfoCleanResponse,
    AbroadDimensionInfoResult,
    AbroadDimensionInfoItem,
    AbroadColorCleanResponse,
    AbroadColorSlimItem,
    AbroadPriceCleanResponse,
    AbroadPriceSlimItem,
    AbroadPropertyCleanResponse,
    AbroadPropertySlimItem,
    AbroadPropertyTrendItem,
    AbroadTopGoodsCleanResponse,
    AbroadTopGoodsSlimResult,
    AbroadTopGoodsSlimItem,
    AbroadAggregatedData,
    # API 原始响应模型
    AbroadTrendSummaryRawResponse,
    AbroadDimensionInfoRawResponse,
    AbroadPropertyRawResponse,
    AbroadTopGoodsRawResponse,
    AbroadPropertyListRawResponse,
)
from app.service.chains.workflow.deepresearch.abroad.llm_output import (
    AbroadExtractedProperty,
    AbroadDimensionAnalysis,
)
from app.service.chains.workflow.deepresearch.abroad.excel_exporter import AbroadExcelExporter
from app.service.rpc.volcengine_kb_api import get_volcengine_kb_api
from app.service.rpc.abroad.client import get_abroad_api
from app.service.rpc.abroad.schemas import (
    AbroadTrendSummaryRequest,
    AbroadDimensionInfoRequest,
    AbroadPropertyTrendRequest,
    AbroadPropertyListRequest,
    AbroadTopGoodsAnalysisRequest,
)


class NodeKey(StrEnum):
    """工作流节点标识"""
    # ========== 前置准备 ==========
    INIT = "初始化"
    CATEGORY_SEARCH = "品类检索"
    MAIN_PARAM_PARSE = "API参数解析"
    SITE_SEARCH = "站点检索"
    SITE_JUDGE = "站点判断"

    # ========== 品类分析分支（table_type=1）==========
    CATEGORY_BRANCH_START = "品类分析_分支开始"
    CATEGORY_OVERVIEW_QUERY = "品类分析_概览查询"
    CATEGORY_CATEGORY_QUERY = "品类分析_品类查询"
    CATEGORY_COLOR_QUERY = "品类分析_颜色查询"
    CATEGORY_PRICE_QUERY = "品类分析_价格带查询"
    CATEGORY_TOP_GOODS_QUERY = "品类分析_Top商品查询"
    CATEGORY_AGGREGATE = "品类分析_数据聚合"

    # ========== 非品类分析分支（table_type=2）==========
    PROPERTY_BRANCH_START = "非品类分析_分支开始"
    PROPERTY_OVERVIEW_QUERY = "非品类分析_概览查询"
    PROPERTY_CATEGORY_QUERY = "非品类分析_品类查询"
    PROPERTY_TOP_GOODS_QUERY = "非品类分析_Top商品查询"
    PROPERTY_DIMENSION_ANALYZE = "非品类分析_维度分析"
    # 维度查询子分支（通过 conditional_edges 路由）
    PROPERTY_DIM_OTHER_QUERY = "非品类分析_属性查询"
    PROPERTY_DIM_COLOR_QUERY = "非品类分析_颜色查询"
    PROPERTY_DIM_FABRIC_QUERY = "非品类分析_面料查询"
    PROPERTY_DIM_PRICE_QUERY = "非品类分析_价格带查询"
    PROPERTY_AGGREGATE = "非品类分析_数据聚合"

    # ========== 报告生成与输出 ==========
    CATEGORY_REPORT_GENERATE = "品类报告生成"
    NORMAL_REPORT_GENERATE = "通用报告生成"
    CATEGORY_NONTHINKING_REPORT_GENERATE = "非思考品类报告生成"
    NORMAL_NONTHINKING_REPORT_GENERATE = "非思考通用报告生成"
    ASSEMBLE_OUTPUT = "输出组装"


class AbroadDeepresearchGraph(BaseWorkflowGraph):
    """
    数据洞察-海外探款数据源-深度思考工作流
    """
    span_name: str = "海外探款数据洞察工作流"

    def __init__(self):
        super().__init__()

    def _build_graph(self) -> CompiledStateGraph:
        """构建工作流图"""
        graph = StateGraph(AbroadDeepresearchState)

        # ============= 注册节点 =============
        # 前置准备节点
        graph.add_node(NodeKey.INIT, self._init_state_node)
        graph.add_node(NodeKey.CATEGORY_SEARCH, self._category_search_node)
        graph.add_node(NodeKey.MAIN_PARAM_PARSE, self._main_param_parse_node)
        graph.add_node(NodeKey.SITE_SEARCH, self._site_search_node)
        graph.add_node(NodeKey.SITE_JUDGE, self._site_judge_node)

        # 品类分析分支节点
        graph.add_node(NodeKey.CATEGORY_BRANCH_START, self._branch_start_node)
        graph.add_node(NodeKey.CATEGORY_OVERVIEW_QUERY, self._query_overview_data)
        graph.add_node(NodeKey.CATEGORY_CATEGORY_QUERY, self._query_category_data)
        graph.add_node(NodeKey.CATEGORY_COLOR_QUERY, self._query_color_data)
        graph.add_node(NodeKey.CATEGORY_PRICE_QUERY, self._query_price_data)
        graph.add_node(NodeKey.CATEGORY_TOP_GOODS_QUERY, self._query_top_goods_data)
        graph.add_node(NodeKey.CATEGORY_AGGREGATE, self._aggregate_category_data)

        # 非品类分析分支节点
        graph.add_node(NodeKey.PROPERTY_BRANCH_START, self._branch_start_node)
        graph.add_node(NodeKey.PROPERTY_OVERVIEW_QUERY, self._query_overview_data)
        graph.add_node(NodeKey.PROPERTY_CATEGORY_QUERY, self._query_category_data)
        graph.add_node(NodeKey.PROPERTY_TOP_GOODS_QUERY, self._query_top_goods_data)
        graph.add_node(NodeKey.PROPERTY_DIMENSION_ANALYZE, self._analyze_dimension_node)
        # 维度查询子分支节点
        graph.add_node(NodeKey.PROPERTY_DIM_OTHER_QUERY, self._query_dim_other_data)
        graph.add_node(NodeKey.PROPERTY_DIM_COLOR_QUERY, self._query_dim_color_data)
        graph.add_node(NodeKey.PROPERTY_DIM_FABRIC_QUERY, self._query_dim_fabric_data)
        graph.add_node(NodeKey.PROPERTY_DIM_PRICE_QUERY, self._query_dim_price_data)
        graph.add_node(NodeKey.PROPERTY_AGGREGATE, self._aggregate_property_data)

        # 报告生成节点
        graph.add_node(NodeKey.CATEGORY_REPORT_GENERATE, self._generate_category_report_node)
        graph.add_node(NodeKey.NORMAL_REPORT_GENERATE, self._generate_normal_report_node)
        graph.add_node(NodeKey.CATEGORY_NONTHINKING_REPORT_GENERATE, self._generate_nonthinking_category_report_node)
        graph.add_node(NodeKey.NORMAL_NONTHINKING_REPORT_GENERATE, self._generate_nonthinking_normal_report_node)
        graph.add_node(NodeKey.ASSEMBLE_OUTPUT, self._assemble_output_node)

        # ============= 定义边 =============
        # 设置入口
        graph.set_entry_point(NodeKey.INIT)

        # 前置准备链
        graph.add_edge(NodeKey.INIT, NodeKey.CATEGORY_SEARCH)
        graph.add_edge(NodeKey.CATEGORY_SEARCH, NodeKey.MAIN_PARAM_PARSE)
        graph.add_edge(NodeKey.MAIN_PARAM_PARSE, NodeKey.SITE_SEARCH)
        graph.add_edge(NodeKey.SITE_SEARCH, NodeKey.SITE_JUDGE)

        # 条件分支：根据table_type分流
        graph.add_conditional_edges(
            NodeKey.SITE_JUDGE,
            self._route_by_table_type,
            {
                "1": NodeKey.CATEGORY_BRANCH_START,
                "2": NodeKey.PROPERTY_BRANCH_START,
            }
        )

        # 品类分析分支：并行查询 5 个维度
        graph.add_edge(NodeKey.CATEGORY_BRANCH_START, NodeKey.CATEGORY_OVERVIEW_QUERY)
        graph.add_edge(NodeKey.CATEGORY_BRANCH_START, NodeKey.CATEGORY_CATEGORY_QUERY)
        graph.add_edge(NodeKey.CATEGORY_BRANCH_START, NodeKey.CATEGORY_COLOR_QUERY)
        graph.add_edge(NodeKey.CATEGORY_BRANCH_START, NodeKey.CATEGORY_PRICE_QUERY)
        graph.add_edge(NodeKey.CATEGORY_BRANCH_START, NodeKey.CATEGORY_TOP_GOODS_QUERY)

        # 品类分析分支：聚合（等待所有并行查询完成）
        graph.add_edge([
            NodeKey.CATEGORY_OVERVIEW_QUERY,
            NodeKey.CATEGORY_CATEGORY_QUERY,
            NodeKey.CATEGORY_COLOR_QUERY,
            NodeKey.CATEGORY_PRICE_QUERY,
            NodeKey.CATEGORY_TOP_GOODS_QUERY,
        ], NodeKey.CATEGORY_AGGREGATE)

        # 品类分析分支：生成报告
        graph.add_conditional_edges(NodeKey.CATEGORY_AGGREGATE, self._route_if_thinking_report, {
            True: NodeKey.CATEGORY_REPORT_GENERATE, False: NodeKey.CATEGORY_NONTHINKING_REPORT_GENERATE
        })

        # 非品类分析分支：基础查询 + 维度分析
        graph.add_edge(NodeKey.PROPERTY_BRANCH_START, NodeKey.PROPERTY_OVERVIEW_QUERY)
        graph.add_edge(NodeKey.PROPERTY_BRANCH_START, NodeKey.PROPERTY_CATEGORY_QUERY)
        graph.add_edge(NodeKey.PROPERTY_BRANCH_START, NodeKey.PROPERTY_TOP_GOODS_QUERY)
        graph.add_edge(NodeKey.PROPERTY_BRANCH_START, NodeKey.PROPERTY_DIMENSION_ANALYZE)

        # 非品类分析分支：维度查询条件分支（根据dimension_type路由）
        graph.add_conditional_edges(
            NodeKey.PROPERTY_DIMENSION_ANALYZE,
            self._route_by_dimension_type,
            {
                "0": NodeKey.PROPERTY_DIM_OTHER_QUERY,   # 其他/属性
                "1": NodeKey.PROPERTY_DIM_COLOR_QUERY,   # 颜色
                "2": NodeKey.PROPERTY_DIM_FABRIC_QUERY,  # 面料
                "3": NodeKey.PROPERTY_DIM_PRICE_QUERY,   # 价格带
            }
        )

        # 非品类分析分支：聚合 (条件展开的分支不能用list作为start key)
        graph.add_edge([
            NodeKey.PROPERTY_OVERVIEW_QUERY,
            NodeKey.PROPERTY_CATEGORY_QUERY,
            NodeKey.PROPERTY_TOP_GOODS_QUERY,
        ], NodeKey.PROPERTY_AGGREGATE)
        graph.add_edge(NodeKey.PROPERTY_DIM_OTHER_QUERY, NodeKey.PROPERTY_AGGREGATE)
        graph.add_edge(NodeKey.PROPERTY_DIM_COLOR_QUERY, NodeKey.PROPERTY_AGGREGATE)
        graph.add_edge(NodeKey.PROPERTY_DIM_FABRIC_QUERY, NodeKey.PROPERTY_AGGREGATE)
        graph.add_edge(NodeKey.PROPERTY_DIM_PRICE_QUERY, NodeKey.PROPERTY_AGGREGATE)

        # 非品类分析分支：生成报告
        graph.add_conditional_edges(NodeKey.PROPERTY_AGGREGATE, self._route_if_thinking_report, {
            True: NodeKey.NORMAL_REPORT_GENERATE, False: NodeKey.NORMAL_NONTHINKING_REPORT_GENERATE
        })

        # 输出组装
        graph.add_edge(NodeKey.CATEGORY_REPORT_GENERATE, NodeKey.ASSEMBLE_OUTPUT)
        graph.add_edge(NodeKey.CATEGORY_NONTHINKING_REPORT_GENERATE, NodeKey.ASSEMBLE_OUTPUT)
        graph.add_edge(NodeKey.NORMAL_REPORT_GENERATE, NodeKey.ASSEMBLE_OUTPUT)
        graph.add_edge(NodeKey.NORMAL_NONTHINKING_REPORT_GENERATE, NodeKey.ASSEMBLE_OUTPUT)
        graph.set_finish_point(NodeKey.ASSEMBLE_OUTPUT)

        return graph.compile()

    # ==================== 前置准备节点 ====================

    def _init_state_node(self, state: AbroadDeepresearchState) -> dict:
        """初始化状态"""
        req = state.request
        is_thinking = req.thinking if req else False

        # 创建消息推送器
        pusher = DeepresearchMessagePusher(
            request=req,
            is_thinking=is_thinking,
            workflow_type="abroad",
        )

        pusher.push_task_start_msg()

        if not is_thinking:
            pusher.push_phase("正在启用趋势分析助手")

        return {
            "is_thinking": is_thinking,
            "message_pusher": pusher,
        }

    def _category_search_node(self, state: AbroadDeepresearchState) -> dict:
        """
        品类检索节点
        使用火山引擎知识库进行向量检索
        """
        req = state.request
        pusher = state.message_pusher
        logger.info(f"[海外探款] 品类检索 - 用户查询: {req.user_query if req else 'N/A'}")

        # 向量搜索
        kb_client = get_volcengine_kb_api()
        resp = kb_client.simple_chat(
            query=req.user_query,
            service_resource_id=VolcKnowledgeServiceId.ABROAD_CATEGORY_VECTOR.value
        )
        parse_content_list: list[dict] = kb_client.parse_structure_chat_response(resp)

        # 使用 parse_abroad_category_list 解析向量检索结果
        parsed_category = parse_abroad_category_list(parse_content_list)

        return {
            "recall_category": parsed_category  # 返回解析后的 AbroadParsedCategory 对象
        }

    def _main_param_parse_node(self, state: AbroadDeepresearchState) -> dict:
        """
        API参数解析节点
        使用LLM结构化输出解析用户问题中的查询参数
        """
        req = state.request
        recall_category = state.recall_category
        pusher = state.message_pusher

        # 获取LLM
        llm = llm_factory.get_llm(
            LlmProvider.OPENROUTER.name,
            LlmModelName.OPENROUTER_GEMINI_3_FLASH_PREVIEW.value
        )
        struct_llm = llm.with_structured_output(schema=AbroadMainParseParam).with_retry(stop_after_attempt=2)

        # 计算日期参数（参考douyi工作流）
        today = datetime.now()
        t_minus_1 = today - timedelta(days=1)  # T-1
        last_month_end = (today.replace(day=1) - timedelta(days=1))  # 上月最后一天
        six_months_ago_first = (today - relativedelta(months=6)).replace(day=1)  # 6个月前第一天
        recent_month_end = last_month_end  # 最近月末
        recent_month_start = last_month_end.replace(day=1)  # 最近月初

        invoke_params = {
            "current_date": t_minus_1.strftime("%Y-%m-%d"),
            "default_end_date": last_month_end.strftime("%Y-%m-%d"),
            "default_start_date": six_months_ago_first.strftime("%Y-%m-%d"),
            "recent_month_end_date": recent_month_end.strftime("%Y-%m-%d"),
            "recent_month_start_date": recent_month_start.strftime("%Y-%m-%d"),
            "user_query": req.user_query if req else "",
            "category_list": recall_category.model_dump_json(ensure_ascii=False) if recall_category else "[]",
        }

        # 从CozePromptHub获取提示词
        messages: list[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.DR_ABROAD_MAIN_PARAM_PARSE_PROMPT.value,
            variables=invoke_params,
        )

        # 执行LLM获取结构化输出
        parse_param: AbroadMainParseParam = struct_llm.with_config(run_name="解析主要参数").invoke(messages)

        pusher.push_phase("数据收集", variables={
            "industry_name": parse_param.root_category_id_name
        })

        return {
            "main_parse_param": parse_param
        }

    def _site_search_node(self, state: AbroadDeepresearchState) -> dict:
        """
        站点检索节点
        使用火山引擎知识库检索可用站点列表
        """
        req = state.request
        logger.info(f"[海外探款] 站点检索 - 用户查询: {req.user_query}")

        # 向量搜索站点
        kb_client = get_volcengine_kb_api()
        resp = kb_client.simple_chat(
            query=req.user_query,
            service_resource_id=VolcKnowledgeServiceId.ABROAD_SITE_VECTOR.value
        )
        parse_content_list: list[dict] = kb_client.parse_structure_chat_response(resp)

        format_recall_platform_list = [
            {
                "platform_name": item['key'],
                "platform_type": item['value']
            }
            for item in parse_content_list
        ]
        logger.debug(f"[海外探款数据洞察] 站点检索结果数量: {len(format_recall_platform_list)}")

        return {
            "recall_platforms": format_recall_platform_list  # 返回可用站点列表
        }

    def _site_judge_node(self, state: AbroadDeepresearchState) -> dict:
        """
        站点判断节点
        使用LLM从可选站点列表中匹配用户偏好的站点
        """
        req = state.request
        available_sites = state.recall_platforms
        logger.info("[海外探款] 站点判断 - 开始")

        # 获取LLM
        llm = llm_factory.get_llm(
            LlmProvider.OPENROUTER.name,
            LlmModelName.OPENROUTER_GEMINI_3_FLASH_PREVIEW.value
        )
        struct_llm = llm.with_structured_output(schema=AbroadPlatformType).with_retry(stop_after_attempt=2)

        invoke_params = {
            "user_query": req.user_query if req else "",
            "available_platform_list": json.dumps(available_sites, ensure_ascii=False) if available_sites else "[]",
        }

        # 从CozePromptHub获取提示词
        messages: list[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.DR_ABROAD_SITE_JUDGE_PROMPT.value,
            variables=invoke_params,
        )

        # 执行LLM
        result: AbroadPlatformType = struct_llm.with_config(run_name="站点判断").invoke(messages)
        logger.info(f"[海外探款] 站点判断完成 - platform_type={result.target_platform_type_list}")
        return {"target_platform_type_list": result.target_platform_type_list}

    # ==================== 路由方法 ====================

    def _route_by_table_type(self, state: AbroadDeepresearchState) -> str:
        """根据table_type路由到不同分支"""
        if state.main_parse_param:
            table_type = state.main_parse_param.table_type
            if table_type == 2:
                return "2"
        return "1"  # 默认品类分析

    def _route_by_dimension_type(self, state: AbroadDeepresearchState) -> str:
        """根据dimension_type路由到不同维度查询节点"""
        dimension_type = state.property_dimension_type or 0
        return str(dimension_type)

    # ==================== 分支共用节点 ====================

    def _branch_start_node(self, state: AbroadDeepresearchState) -> dict:
        """分支开始节点（无操作，仅用于并行分发）"""
        return {}

    # ==================== 数据查询节点 ====================

    def _get_api_params(self, state: AbroadDeepresearchState) -> dict:
        """提取通用 API 请求参数"""
        req = state.request
        parse_param = state.main_parse_param
        platform_type_list = state.target_platform_type_list or []

        single_category_id_list = []
        if parse_param.root_category_id:
            single_category_id_list.append(parse_param.root_category_id)
        if parse_param.category_id:
            single_category_id_list.append(parse_param.category_id)
        category_id_list = [single_category_id_list]

        return {
            "user_id": str(req.user_id) if req and req.user_id else "0",
            "team_id": str(req.team_id) if req and req.team_id else "0",
            "platform_type_list": platform_type_list,
            "category_id_list": category_id_list,
            "start_date": parse_param.start_date if parse_param else None,
            "end_date": parse_param.end_date if parse_param else None,
            "date_granularity": parse_param.date_type if parse_param else "MONTH",
        }

    def _query_overview_data(self, state: AbroadDeepresearchState) -> dict:
        """
        查询概览趋势数据
        调用 /overview/dimension-analyze/trend-summary 接口
        """
        parse_param = state.main_parse_param
        params = self._get_api_params(state)
        logger.info(f"[海外探款] 概览趋势查询 - platform_type_list={params['platform_type_list']}")

        # 构建 Request 对象
        request = AbroadTrendSummaryRequest(
            platformTypeList=params["platform_type_list"],
            categoryIdList=params["category_id_list"],
            startDate=params["start_date"],
            endDate=params["end_date"],
            dateGranularity=params["date_granularity"],
        )

        # 调用 API
        api = get_abroad_api()
        raw_response = api.get_trend_summary(
            user_id=params["user_id"],
            team_id=params["team_id"],
            request=request,
        )

        # 数据清洗
        clean_data = self._clean_trend_summary_data(raw_response)
        logger.info(f"[海外探款] 概览趋势查询完成 - success={clean_data.success}")

        # 根据 table_type 返回不同的状态字段
        table_type = parse_param.table_type if parse_param else 1
        if table_type == 1:
            return {"category_trend_data": clean_data}
        else:
            return {"property_trend_data": clean_data}

    def _query_category_data(self, state: AbroadDeepresearchState) -> dict:
        """
        查询品类维度数据
        调用 /overview/dimension-analyze/info 接口，dimension=KJ_CATEGORY
        """
        parse_param = state.main_parse_param
        params = self._get_api_params(state)
        logger.info(f"[海外探款] 品类维度查询 - platform_type_list={params['platform_type_list']}")

        # 构建 Request 对象（业务逻辑在应用层设置 dimension）
        request = AbroadDimensionInfoRequest(
            platformTypeList=params["platform_type_list"],
            dimension="KJ_CATEGORY",
            categoryIdList=params["category_id_list"],
            startDate=params["start_date"],
            endDate=params["end_date"],
            dateGranularity="MONTH",
        )

        # 调用 API
        api = get_abroad_api()
        raw_response = api.get_dimension_info(
            user_id=params["user_id"],
            team_id=params["team_id"],
            request=request,
        )

        # 数据清洗
        clean_data = self._clean_dimension_info_data(raw_response)
        logger.info(f"[海外探款] 品类维度查询完成 - success={clean_data.success}")

        # 根据 table_type 返回不同的状态字段
        table_type = parse_param.table_type if parse_param else 1
        if table_type == 1:
            return {"category_category_data": clean_data}
        else:
            return {"property_category_data": clean_data}

    def _query_color_data(self, state: AbroadDeepresearchState) -> dict:
        """
        查询颜色维度数据
        调用 /overview/dimension-analyze/info 接口，dimension=COLOR
        """
        params = self._get_api_params(state)
        logger.info(f"[海外探款] 颜色维度查询 - platform_type_list={params['platform_type_list']}")

        # 构建 Request 对象（业务逻辑在应用层设置 dimension）
        request = AbroadDimensionInfoRequest(
            platformTypeList=params["platform_type_list"],
            dimension="COLOR",
            categoryIdList=params["category_id_list"],
            startDate=params["start_date"],
            endDate=params["end_date"],
            dateGranularity="MONTH",
        )

        # 调用 API
        api = get_abroad_api()
        raw_response = api.get_dimension_info(
            user_id=params["user_id"],
            team_id=params["team_id"],
            request=request,
        )

        # 数据清洗
        clean_data = self._clean_color_data(raw_response)
        logger.info(f"[海外探款] 颜色维度查询完成 - success={clean_data.success}")
        return {"category_color_data": clean_data}

    def _query_price_data(self, state: AbroadDeepresearchState) -> dict:
        """
        查询价格带维度数据
        调用 /overview/dimension-analyze/info 接口，dimension=PRICE
        """
        params = self._get_api_params(state)
        logger.info(f"[海外探款] 价格带维度查询 - platform_type_list={params['platform_type_list']}")

        # 构建 Request 对象（业务逻辑在应用层设置 dimension）
        request = AbroadDimensionInfoRequest(
            platformTypeList=params["platform_type_list"],
            dimension="PRICE",
            categoryIdList=params["category_id_list"],
            startDate=params["start_date"],
            endDate=params["end_date"],
            dateGranularity="MONTH",
        )

        # 调用 API
        api = get_abroad_api()
        raw_response = api.get_dimension_info(
            user_id=params["user_id"],
            team_id=params["team_id"],
            request=request,
        )

        # 数据清洗
        clean_data = self._clean_price_data(raw_response)
        logger.info(f"[海外探款] 价格带维度查询完成 - success={clean_data.success}")
        return {"category_price_data": clean_data}

    def _query_top_goods_data(self, state: AbroadDeepresearchState) -> dict:
        """
        查询Top商品数据
        调用 /goods-center/goods-zone-list 接口
        """
        parse_param = state.main_parse_param
        params = self._get_api_params(state)
        logger.info(f"[海外探款] Top商品查询 - platform_type_list={params['platform_type_list']}")

        # 构建 Request 对象
        request = AbroadTopGoodsAnalysisRequest(
            platformTypeList=params["platform_type_list"],
            categoryIdList=params["category_id_list"],
            startDate=params["start_date"],
            endDate=params["end_date"],
            pageNo=1,
            pageSize=10,
            sortType=8,  # 近30天热销
            goodsZoneType="ALL",
            onSaleFlag=1,
        )

        # 调用 API
        api = get_abroad_api()
        raw_response = api.get_top_goods_for_analysis(
            user_id=params["user_id"],
            team_id=params["team_id"],
            request=request,
        )

        # 数据清洗
        clean_data = self._clean_top_goods_data(raw_response)
        logger.info(f"[海外探款] Top商品查询完成 - success={clean_data.success}")

        # 根据 table_type 返回不同的状态字段
        table_type = parse_param.table_type if parse_param else 1
        if table_type == 1:
            return {"category_top_goods": clean_data}
        else:
            return {"property_top_goods": clean_data}

    # ==================== 非品类分析专用节点 ====================

    def _analyze_dimension_node(self, state: AbroadDeepresearchState) -> dict:
        """
        维度分析节点
        使用LLM判断用户查询的维度类型：0其他/1颜色/2面料/3价格带
        """
        req = state.request
        logger.info("[海外探款] 维度分析 - 开始")

        # 获取LLM
        llm = llm_factory.get_llm(
            LlmProvider.OPENROUTER.name,
            LlmModelName.OPENROUTER_GEMINI_3_FLASH_PREVIEW.value
        )
        struct_llm = llm.with_structured_output(schema=AbroadDimensionAnalysis).with_retry(stop_after_attempt=2)

        invoke_params = {
            "user_query": req.user_query if req else "",
        }

        # 从CozePromptHub获取提示词
        messages: list[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.DR_ABROAD_DIMENSION_ANALYZE_PROMPT.value,
            variables=invoke_params,
        )

        try:
            result: AbroadDimensionAnalysis = struct_llm.with_config(run_name="维度分析").invoke(messages)
            logger.info(f"[海外探款] 维度分析完成 - dimension_type={result.dimension_type}")
            return {"property_dimension_type": result.dimension_type}
        except Exception as e:
            logger.warning(f"[海外探款] 维度分析失败: {e}，默认使用属性维度")
            return {"property_dimension_type": AbroadDimensionType.OTHER.value}

    def _query_dimension_data(self, state: AbroadDeepresearchState) -> dict:
        """
        根据维度类型查询对应数据（已废弃，保留向后兼容）
        现已拆分为4个独立的维度查询方法，通过 conditional_edges 路由
        """
        logger.warning("[海外探款] _query_dimension_data 已废弃，请使用具体的维度查询方法")
        return {"property_dimension_data": None}

    def _query_dim_other_data(self, state: AbroadDeepresearchState) -> dict:
        """
        查询属性数据（维度类型=0：其他/默认属性）
        - 调用 property-list 接口获取属性列表
        - 使用 LLM 提取用户查询的属性名
        - 调用 v2/trend 接口获取属性趋势数据
        """
        req = state.request
        params = self._get_api_params(state)
        logger.info("[海外探款] 属性查询 - 开始")

        api = get_abroad_api()

        # 1. 获取可用属性列表
        property_list_request = AbroadPropertyListRequest(
            platformTypeList=params["platform_type_list"],
            categoryIdList=params["category_id_list"],
        )
        property_list_response = api.get_property_list(
            user_id=params["user_id"],
            team_id=params["team_id"],
            request=property_list_request,
        )

        if not property_list_response.success or not property_list_response.result:
            logger.warning("[海外探款] 属性列表获取失败")
            return {"property_dimension_data": AbroadPropertyCleanResponse(success=False, result=[])}

        # 提取属性名称列表
        available_properties = [
            item.property_name for item in property_list_response.result
            if item.property_name
        ]

        # 2. 使用 LLM 提取用户查询的属性名
        llm = llm_factory.get_llm(
            LlmProvider.OPENROUTER.name,
            LlmModelName.OPENROUTER_GEMINI_3_FLASH_PREVIEW.value
        )
        struct_llm = llm.with_structured_output(schema=AbroadExtractedProperty).with_retry(stop_after_attempt=2)

        invoke_params = {
            "user_query": req.user_query if req else "",
            "available_properties": json.dumps(available_properties, ensure_ascii=False),
        }

        messages: list[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.DR_ABROAD_PROPERTY_EXTRACT_PROMPT.value,
            variables=invoke_params,
        )

        try:
            extracted: AbroadExtractedProperty = struct_llm.with_config(run_name="属性提取").invoke(messages)
            property_name = extracted.property_name
            logger.info(f"[海外探款] 属性提取完成 - property_name={property_name}")
        except Exception as e:
            logger.warning(f"[海外探款] 属性提取失败: {e}，使用默认属性")
            property_name = available_properties[0] if available_properties else "袖长"

        # 3. 调用 v2/trend 接口获取属性趋势数据
        property_trend_request = AbroadPropertyTrendRequest(
            platformTypeList=params["platform_type_list"],
            propertyName=property_name,
            categoryIdList=params["category_id_list"],
            startDate=params["start_date"],
            endDate=params["end_date"],
            dateGranularity="MONTH",
        )
        raw_response = api.get_property_trend(
            user_id=params["user_id"],
            team_id=params["team_id"],
            request=property_trend_request,
        )

        # 数据清洗
        clean_data = self._clean_property_trend_data(raw_response)
        logger.info(f"[海外探款] 属性查询完成 - success={clean_data.success}")
        return {
            "property_dimension_data": clean_data,
            "extracted_property": property_name,
        }

    def _query_dim_color_data(self, state: AbroadDeepresearchState) -> dict:
        """
        查询颜色数据（维度类型=1：颜色）
        调用 info 接口，dimension=COLOR
        """
        params = self._get_api_params(state)
        logger.info("[海外探款] 颜色查询(非品类分支) - 开始")

        # 构建 Request 对象
        request = AbroadDimensionInfoRequest(
            platformTypeList=params["platform_type_list"],
            dimension="COLOR",
            categoryIdList=params["category_id_list"],
            startDate=params["start_date"],
            endDate=params["end_date"],
            dateGranularity=params["date_granularity"],
        )

        api = get_abroad_api()
        raw_response = api.get_dimension_info(
            user_id=params["user_id"],
            team_id=params["team_id"],
            request=request,
        )

        # 数据清洗 - 转换为属性格式
        clean_data = self._clean_color_to_property_data(raw_response)
        logger.info(f"[海外探款] 颜色查询(非品类分支)完成 - success={clean_data.success}")
        return {"property_dimension_data": clean_data}

    def _query_dim_fabric_data(self, state: AbroadDeepresearchState) -> dict:
        """
        查询面料数据（维度类型=2：面料）
        调用 v2/trend 接口，property_name="面料"
        """
        params = self._get_api_params(state)
        logger.info("[海外探款] 面料查询 - 开始")

        # 构建 Request 对象
        request = AbroadPropertyTrendRequest(
            platformTypeList=params["platform_type_list"],
            propertyName="面料",  # 固定查询面料
            categoryIdList=params["category_id_list"],
            startDate=params["start_date"],
            endDate=params["end_date"],
            dateGranularity=params["date_granularity"],
        )

        api = get_abroad_api()
        raw_response = api.get_property_trend(
            user_id=params["user_id"],
            team_id=params["team_id"],
            request=request,
        )

        # 数据清洗
        clean_data = self._clean_property_trend_data(raw_response)
        logger.info(f"[海外探款] 面料查询完成 - success={clean_data.success}")
        return {"property_dimension_data": clean_data}

    def _query_dim_price_data(self, state: AbroadDeepresearchState) -> dict:
        """
        查询价格带数据（维度类型=3：价格带）
        调用 info 接口，dimension=PRICE
        """
        params = self._get_api_params(state)
        logger.info("[海外探款] 价格带查询(非品类分支) - 开始")

        # 构建 Request 对象
        request = AbroadDimensionInfoRequest(
            platformTypeList=params["platform_type_list"],
            dimension="PRICE",
            categoryIdList=params["category_id_list"],
            startDate=params["start_date"],
            endDate=params["end_date"],
            dateGranularity="MONTH",
        )

        api = get_abroad_api()
        raw_response = api.get_dimension_info(
            user_id=params["user_id"],
            team_id=params["team_id"],
            request=request,
        )

        # 数据清洗 - 转换为属性格式
        clean_data = self._clean_price_to_property_data(raw_response)
        logger.info(f"[海外探款] 价格带查询(非品类分支)完成 - success={clean_data.success}")
        return {"property_dimension_data": clean_data}

    # ==================== 数据清洗方法 ====================

    def _clean_trend_summary_data(
        self, raw_response: AbroadTrendSummaryRawResponse
    ) -> AbroadTrendCleanResponse:
        """清洗 trend-summary 接口响应（适配数组格式，取第一个 bucket）"""
        if not raw_response.success or not raw_response.result:
            return AbroadTrendCleanResponse(success=False, result=None)

        # result 是数组，取第一个 bucket
        if len(raw_response.result) == 0:
            return AbroadTrendCleanResponse(success=False, result=None)

        bucket_data = raw_response.result[0]

        # 提取销量趋势
        volume_trend = [
            AbroadTrendSlimItem(
                recordDate=item.key or item.record_date,
                saleVolume=item.value or 0,
                saleAmount=0,
            )
            for item in bucket_data.sale_volume_trend
        ]

        # 提取销售额趋势
        amount_trend = [
            AbroadTrendSlimItem(
                recordDate=item.key or item.record_date,
                saleVolume=0,
                saleAmount=item.value or 0,
            )
            for item in bucket_data.sale_amount_trend
        ]

        return AbroadTrendCleanResponse(
            success=True,
            result=AbroadTrendSlimResult(
                sale_volume_trend=volume_trend,
                sale_amount_trend=amount_trend,
                currency=bucket_data.currency,
            )
        )

    def _clean_dimension_info_data(
        self, raw_response: AbroadDimensionInfoRawResponse
    ) -> AbroadDimensionInfoCleanResponse:
        """清洗 info 接口响应（通用：品类/颜色/价格带）"""
        if not raw_response.success or not raw_response.result:
            return AbroadDimensionInfoCleanResponse(success=False, result=None)

        result_data = raw_response.result

        # 使用 .data_list 获取维度数据列表
        items = [
            AbroadDimensionInfoItem(
                dimensionInfo=item.dimension_info,
                saleVolume=item.sale_volume,
                saleAmount=item.sale_amount,
                saleVolumeRatio=item.sale_volume_ratio,
                saleAmountRatio=item.sale_amount_ratio,
                saleVolumeMomRatio=item.sale_volume_mom_ratio,
                saleAmountMomRatio=item.sale_amount_mom_ratio,
                saleVolumeYoyRatio=item.sale_volume_yoy_ratio,
                saleAmountYoyRatio=item.sale_amount_yoy_ratio,
            )
            for item in result_data.data_list
        ]

        return AbroadDimensionInfoCleanResponse(
            success=True,
            result=AbroadDimensionInfoResult(
                resultCount=result_data.result_count or len(items),
                items=items,
            )
        )

    def _clean_color_data(
        self, raw_response: AbroadDimensionInfoRawResponse
    ) -> AbroadColorCleanResponse:
        """清洗颜色数据（从维度分析接口响应）"""
        if not raw_response.success or not raw_response.result:
            return AbroadColorCleanResponse(success=False, result=[])

        # 使用 .data_list 获取维度数据列表
        items = [
            AbroadColorSlimItem(
                property_value=item.dimension_info,
                sales_volume=item.sale_volume,
                sales_amount=item.sale_amount,
                rate=item.sale_volume_ratio,
                other_flag=False,
            )
            for item in raw_response.result.data_list
        ]

        return AbroadColorCleanResponse(success=True, result=items)

    def _clean_price_data(
        self, raw_response: AbroadDimensionInfoRawResponse
    ) -> AbroadPriceCleanResponse:
        """清洗价格带数据（从维度分析接口响应）"""
        if not raw_response.success or not raw_response.result:
            return AbroadPriceCleanResponse(success=False, result=[])

        items = []
        # 使用 .data_list 获取维度数据列表
        for item in raw_response.result.data_list:
            # dimension_info 格式可能是 "0-100" 或具体价格区间
            price_range = item.dimension_info or ""
            if "~" in price_range:
                parts = price_range.split("~")
                left_price = int(float(parts[0])) * 100 if parts[0] else 0
                right_price = int(float(parts[1])) * 100 if len(parts) > 1 and parts[1] else None
            else:
                left_price = int(float(price_range)) * 100 if price_range else 0
                right_price = None

            items.append(AbroadPriceSlimItem(
                left_price=left_price,
                right_price=right_price,
                sales_volume=item.sale_volume,
                rate=item.sale_volume_ratio,
            ))

        return AbroadPriceCleanResponse(success=True, result=items)

    def _clean_top_goods_data(
        self, raw_response: AbroadTopGoodsRawResponse
    ) -> AbroadTopGoodsCleanResponse:
        """清洗 goods-zone-list 接口响应"""
        if not raw_response.success or not raw_response.result:
            return AbroadTopGoodsCleanResponse(success=False, result=None)

        result_data = raw_response.result

        items = [
            AbroadTopGoodsSlimItem(
                rank=idx + 1,
                item_id=item.product_id,
                title=item.product_name,
                pic_url=item.pic_url,
                category_name=item.category_name,
                category_detail=item.category_detail,
                sales_volume=item.sale_volume_30day,
                sales_amount=item.sale_amount_30day,
                min_price=str(item.min_price) if item.min_price else None,
                max_s_price=int(item.max_s_price) if item.max_s_price else None,
            )
            for idx, item in enumerate(result_data.result_list[:10])  # 只取前10
        ]

        return AbroadTopGoodsCleanResponse(
            success=True,
            result=AbroadTopGoodsSlimResult(
                start=result_data.start,
                page_size=result_data.page_size,
                result_count=result_data.result_count,
                top10_slim=items,
            )
        )

    def _clean_property_trend_data(
        self, raw_response: AbroadPropertyRawResponse
    ) -> AbroadPropertyCleanResponse:
        """清洗属性趋势数据（v2/trend接口）"""
        if not raw_response.success:
            return AbroadPropertyCleanResponse(success=False, result=[])

        items = [
            AbroadPropertySlimItem(
                property_value=item.property_value,
                sales_volume=item.sales_volume,
                sales_amount=item.sales_amount,
                rate=item.rate,
                other_flag=item.other_flag,
                trends=[
                    AbroadPropertyTrendItem(
                        date_range=trend.date_range,
                        sales_volume=trend.sales_volume,
                    )
                    for trend in item.trends
                ],
            )
            for item in raw_response.result
        ]

        return AbroadPropertyCleanResponse(success=True, result=items)

    def _clean_color_to_property_data(
        self, raw_response: AbroadDimensionInfoRawResponse
    ) -> AbroadPropertyCleanResponse:
        """将颜色数据转换为属性数据格式（用于非品类分支）"""
        if not raw_response.success or not raw_response.result:
            return AbroadPropertyCleanResponse(success=False, result=[])

        # 使用 .data_list 获取维度数据列表
        items = [
            AbroadPropertySlimItem(
                property_value=item.dimension_info,
                sales_volume=item.sale_volume,
                sales_amount=item.sale_amount,
                rate=item.sale_volume_ratio,
                other_flag=False,
                trends=[],  # 颜色数据无趋势
            )
            for item in raw_response.result.data_list
        ]

        return AbroadPropertyCleanResponse(success=True, result=items)

    def _clean_price_to_property_data(
        self, raw_response: AbroadDimensionInfoRawResponse
    ) -> AbroadPropertyCleanResponse:
        """将价格带数据转换为属性数据格式（用于非品类分支）"""
        if not raw_response.success or not raw_response.result:
            return AbroadPropertyCleanResponse(success=False, result=[])

        # 使用 .data_list 获取维度数据列表
        items = [
            AbroadPropertySlimItem(
                property_value=item.dimension_info or "",
                sales_volume=item.sale_volume,
                sales_amount=item.sale_amount,
                rate=item.sale_volume_ratio,
                other_flag=False,
                trends=[],  # 价格带数据无趋势
            )
            for item in raw_response.result.data_list
        ]

        return AbroadPropertyCleanResponse(success=True, result=items)

    # ==================== 数据聚合节点 ====================

    def _aggregate_category_data(self, state: AbroadDeepresearchState) -> dict:
        """
        聚合品类分析分支数据
        将各查询结果汇总为AbroadAggregatedData供报告生成使用
        """
        aggregated = AbroadAggregatedData(
            table_type=1,
            trend_data=state.category_trend_data,
            category_data=state.category_category_data,
            color_data=state.category_color_data,
            price_data=state.category_price_data,
            top_goods=state.category_top_goods,
        )
        return {"aggregated_data": aggregated}

    def _aggregate_property_data(self, state: AbroadDeepresearchState) -> dict:
        """
        聚合非品类分析分支数据
        将各查询结果汇总为AbroadAggregatedData供报告生成使用
        """
        aggregated = AbroadAggregatedData(
            table_type=2,
            trend_data=state.property_trend_data,
            category_data=state.property_category_data,
            property_data=state.property_dimension_data,
            top_goods=state.property_top_goods,
        )
        return {"aggregated_data": aggregated}

    # ==================== 报告生成节点 ====================
    def _route_if_thinking_report(self, state: AbroadDeepresearchState) -> bool:
        return state.is_thinking

    def _generate_category_report_node(self, state: AbroadDeepresearchState) -> dict:
        """
        生成品类分析报告
        """
        req = state.request
        param = state.main_parse_param
        aggregated_data = state.aggregated_data
        pusher = state.message_pusher

        if not param or not aggregated_data:
            return {"report_text": "数据不足，无法生成报告。"}

        pusher.push_phase("洞察生成中")

        # 获取LLM
        llm = llm_factory.get_llm(
            LlmProvider.OPENROUTER.name,
            LlmModelName.OPENROUTER_GEMINI_3_FLASH_PREVIEW.value
        )
        chain = llm.with_retry(stop_after_attempt=2) | StrOutputParser()

        # 准备报告生成参数
        invoke_params = {
            # 基础数据
            "date_range": f"{param.start_date}至{param.end_date}",
            "category_path": f"{param.root_category_id_name}-{param.category_id_name}",
            # 分析结果数据
            "overview_sale_trend_data": aggregated_data.trend_data.model_dump_json(ensure_ascii=False) if aggregated_data.trend_data else "{}",
            "category_summary": aggregated_data.category_data.model_dump_json(ensure_ascii=False) if aggregated_data.category_data else "{}",
            "price_summary": aggregated_data.price_data.model_dump_json(ensure_ascii=False) if aggregated_data.price_data else "{}",
            "color_summary": aggregated_data.color_data.model_dump_json(ensure_ascii=False) if aggregated_data.color_data else "{}",
            "top_products": aggregated_data.top_goods.model_dump_json(ensure_ascii=False) if aggregated_data.top_goods else "{}",
        }

        # 从CozePromptHub获取提示词
        messages: list[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.DR_ABROAD_CATEGORY_REPORT_GENERATE_PROMPT.value,
            variables=invoke_params,
        )
        report_text = chain.with_config(run_name="生成品类分析报告").invoke(messages)
        logger.info(f"[海外探款数据洞察] 报告生成完成")

        pusher.push_phase("洞察生成完成")

        return {"report_text": report_text}


    def _generate_normal_report_node(self, state: AbroadDeepresearchState) -> dict:
        """
        生成非品类分析报告
        """
        req = state.request
        param = state.main_parse_param
        aggregated_data = state.aggregated_data
        pusher = state.message_pusher

        if not param or not aggregated_data:
            return {"report_text": "数据不足，无法生成报告。"}

        pusher.push_phase("洞察生成中")

        # 获取LLM
        llm = llm_factory.get_llm(
            LlmProvider.OPENROUTER.name,
            LlmModelName.OPENROUTER_GEMINI_3_FLASH_PREVIEW.value
        )
        chain = llm.with_retry(stop_after_attempt=2) | StrOutputParser()

        # 准备报告生成参数
        invoke_params = {
            # 基础数据
            "date_range": f"{param.start_date}至{param.end_date}",
            "category_path": f"{param.root_category_id_name}-{param.category_id_name}",
            # 分析结果数据
            "overview_sale_trend_data": aggregated_data.trend_data.model_dump_json(ensure_ascii=False) if aggregated_data.trend_data else "{}",
            "trend_data": aggregated_data.trend_data.model_dump_json(ensure_ascii=False) if aggregated_data.trend_data else "{}",
            "category_summary": aggregated_data.category_data.model_dump_json(ensure_ascii=False) if aggregated_data.category_data else "{}",
            "property_summary": aggregated_data.property_data.model_dump_json(ensure_ascii=False) if aggregated_data.property_data else "{}",
            "top_products": aggregated_data.top_goods.model_dump_json(ensure_ascii=False) if aggregated_data.top_goods else "{}",
        }

        # 从CozePromptHub获取提示词
        messages: list[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.DR_ABROAD_NORMAL_REPORT_GENERATE_PROMPT.value,
            variables=invoke_params,
        )

        # 执行LLM生成报告
        report_text = chain.with_config(run_name="生成非品类分析报告").invoke(messages)
        logger.info(f"[海外探款] 报告生成完成 - 长度={len(report_text)}")

        pusher.push_phase("洞察生成完成")

        return {"report_text": report_text}

    def _generate_nonthinking_category_report_node(self, state: AbroadDeepresearchState) -> dict:
        param = state.main_parse_param
        aggregated_data = state.aggregated_data
        pusher = state.message_pusher

        pusher.push_phase("正在生成数据洞察报告")

        # 获取LLM
        llm = llm_factory.get_llm(LlmProvider.OPENROUTER.name, LlmModelName.OPENROUTER_GEMINI_3_FLASH_PREVIEW.value)
        chain = llm.with_retry(stop_after_attempt=2) | StrOutputParser()

        # 准备报告生成参数
        invoke_params = {
            # 基础数据m
            "date_range": f"{param.start_date}至{param.end_date}",
            "category_path": f"{param.root_category_id_name}-{param.category_id_name}",
            # 分析结果数据
            "overview_sale_trend_data": aggregated_data.trend_data.model_dump_json(ensure_ascii=False) if aggregated_data.trend_data else "{}",
            "category_summary": aggregated_data.category_data.model_dump_json(ensure_ascii=False) if aggregated_data.category_data else "{}",
            "price_summary": aggregated_data.price_data.model_dump_json(ensure_ascii=False) if aggregated_data.price_data else "{}",
            "color_summary": aggregated_data.color_data.model_dump_json(ensure_ascii=False) if aggregated_data.color_data else "{}",
        }

        # 从CozePromptHub获取提示词
        messages: list[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.DR_ABROAD_CATEGORY_NONTHINKING_REPORT_GENERATE_PROMPT.value,
            variables=invoke_params,
        )

        report_text = chain.with_config(run_name="生成品类非深度思考报告").invoke(messages)
        logger.info("[海外探款数据洞察]非思考品类报告生成完成")

        pusher.push_phase("报告已生成")

        return {
            "report_text": report_text
        }

    def _generate_nonthinking_normal_report_node(self, state: AbroadDeepresearchState) -> dict:
        param = state.main_parse_param
        aggregated_data = state.aggregated_data
        pusher = state.message_pusher

        pusher.push_phase("正在生成数据洞察报告")

        # 获取LLM
        llm = llm_factory.get_llm(LlmProvider.OPENROUTER.name, LlmModelName.OPENROUTER_GEMINI_3_FLASH_PREVIEW.value)
        chain = llm.with_retry(stop_after_attempt=2) | StrOutputParser()

        # 准备报告生成参数
        invoke_params = {
            # 基础数据m
            "date_range": f"{param.start_date}至{param.end_date}",
            "category_path": f"{param.root_category_id_name}-{param.category_id_name}",
            # 分析结果数据
            "overview_sale_trend_data": aggregated_data.trend_data.model_dump_json(ensure_ascii=False) if aggregated_data.trend_data else "{}",
            "trend_data": aggregated_data.trend_data.model_dump_json(ensure_ascii=False) if aggregated_data.trend_data else "{}",
            "category_summary": aggregated_data.category_data.model_dump_json(ensure_ascii=False) if aggregated_data.category_data else "{}",
            "property_summary": aggregated_data.property_data.model_dump_json(ensure_ascii=False) if aggregated_data.property_data else "{}",
        }

        # 从CozePromptHub获取提示词
        messages: list[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.DR_ABROAD_NORMAL_NONTHINKING_REPORT_GENERATE_PROMPT.value,
            variables=invoke_params,
        )

        report_text = chain.with_config(run_name="生成通用非深度思考报告").invoke(messages)

        pusher.push_phase("报告已生成")

        return {
            "report_text": report_text
        }

    def _assemble_output_node(self, state: AbroadDeepresearchState) -> dict:
        """
        组装最终输出
        将报告和相关数据封装为WorkflowResponse，并推送 Excel 导出数据
        """
        from app.schemas.response.workflow_response import WorkflowResponse

        # 获取状态数据
        report_text = state.report_text or "报告生成失败"
        aggregated_data = state.aggregated_data
        param = state.main_parse_param
        pusher = state.message_pusher

        # 构建 Excel 导出数据并推送
        if aggregated_data and param and pusher:
            exporter = AbroadExcelExporter(aggregated_data=aggregated_data, param=param)
            excel_data_list = exporter.build_excel_data_list()
            pusher.push_report_and_excel_data(entity_type=WorkflowEntityType.ABROAD_ITEM.code, report_text=report_text, excel_data_list=excel_data_list)
            logger.info(f"[海外探款] Excel 导出数据已推送, count={len(excel_data_list)}")

        pusher.push_task_status_msg()

        # 构建 WorkflowResponse
        workflow_response = WorkflowResponse(
            select_result=report_text,
            relate_data=aggregated_data.model_dump_json(ensure_ascii=False) if aggregated_data else None
        )

        return {
            "workflow_response": workflow_response
        }
