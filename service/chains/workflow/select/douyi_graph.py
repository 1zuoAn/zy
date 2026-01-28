"""
抖衣(抖音)选品工作流 - LangGraph 版本（重构对齐 Zhiyi 2.0 标准）

节点清单（13 个）：
- 预处理: init_state, pre_think, query_selections, llm_parse, llm_merge, parallel_parse
- 主 API: api_strict
- 兜底 API: api_fallback
- 汇聚: merge_result
- 后处理: post_process, has_result, no_result, package

"""

from __future__ import annotations

import json
import random
import time
from datetime import datetime, timedelta
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import text

from app.core.clients.coze_loop_client import coze_loop_client_provider
from app.core.clients.db_client import pg_session
from app.core.clients.redis_client import redis_client
from app.core.config.constants import (
    CozePromptHubKey,
    DBAlias,
    LlmModelName,
    LlmProvider,
    RedisMessageKeyName,
    VolcKnowledgeServiceId,
    WorkflowEntityType,
    WorkflowMessageContentType,
)
from app.core.tools import llm_factory
from app.schemas.entities.message.redis_message import (
    BaseRedisMessage,
    CustomDataContent,
    ParameterData,
    ParameterDataContent,
    TextMessageContent,
    WithActionContent,
)
from app.schemas.entities.workflow.graph_state import DouyiWorkflowState
from app.schemas.entities.workflow.llm.douyi_output import (
    DouyiCategoryParseResult,
    DouyiMiscParseResult,
    DouyiNumericParseResult,
    DouyiParamMergeResult,
    DouyiParseParam,
    DouyiPropertiesParseResult,
    DouyiPropertyLlmCleanResult,
    DouyiSalesFlagParseResult,
    DouyiSortIntentParseResult,
    DouyiSortTypeParseResult,
    DouyiTimeParseResult,
    DouyiUserTagResult,
)
from app.schemas.response.common import PageResult
from app.schemas.response.workflow_response import WorkflowResponse
from app.service.chains.templates.douyi_progress_template import DOUYI_PROGRESS_TEMPLATE
from app.service.chains.workflow.base_graph import BaseWorkflowGraph
from app.service.chains.workflow.progress_pusher import PhaseProgressPusher
from app.service.rpc.volcengine_kb_api import get_volcengine_kb_api
from app.service.rpc.zhiyi_api import (
    DouyiGoodsEntity,
    DouyiGoodsListRequest,
    DouyiSearchRequest,
    get_zhiyi_api_client,
)
from app.utils import thread_pool
from app.utils.query_reference import QueryReferenceHelper

DEFAULT_SORT_FIELD = "saleVolumeDaily"
DEFAULT_SORT_FIELD_NAME = "本期销量最高"
GOODS_LIST_DEFAULT_SORT_FIELD = 3
GOODS_LIST_DEFAULT_SORT_TYPE = 1


class DouyiGraph(BaseWorkflowGraph):
    """抖衣(抖音)选品工作流 - LangGraph 版本"""

    span_name = "抖衣数据源工作流"
    run_name = "douyi-graph"

    # 使用 templates 文件夹的进度模板
    PROGRESS_TEMPLATE = DOUYI_PROGRESS_TEMPLATE

    # 节点名称中文映射（用于 CozeLoop 追踪显示）
    _NODE_NAME_MAP = {
        "init_state": "初始化",
        "pre_think": "预处理思考",
        "query_selections": "查询筛选项",
        "llm_parse": "LLM参数解析",
        "llm_merge": "LLM参数合并",
        "parallel_parse": "并行解析",
        "api_strict": "主API调用",
        "api_fallback": "兜底API调用",
        "merge_result": "结果汇聚",
        "post_process": "后处理",
        "has_result": "有结果处理",
        "no_result": "无结果处理",
        "package": "封装结果",
    }

    def __init__(self) -> None:
        super().__init__()

    def _get_trace_name_modifier(self):
        """返回节点名称修改函数（用于 CozeLoop 追踪）"""

        def modifier(node_name: str) -> str:
            if node_name == "init_state":
                return self.span_name
            return self._NODE_NAME_MAP.get(node_name, node_name)

        return modifier

    def _build_graph(self) -> CompiledStateGraph:
        """构建工作流图 -"""
        graph = StateGraph(DouyiWorkflowState)

        # ===== 预处理节点 =====
        graph.add_node("init_state", self._init_state_node, metadata={"__display_name__": "初始化"})
        graph.add_node(
            "pre_think", self._pre_think_node, metadata={"__display_name__": "预处理思考"}
        )
        graph.add_node(
            "query_selections",
            self._query_selection_node,
            metadata={"__display_name__": "查询筛选项"},
        )
        graph.add_node(
            "llm_parse", self._llm_param_parse_node, metadata={"__display_name__": "LLM参数解析"}
        )
        graph.add_node(
            "llm_merge", self._llm_param_merge_node, metadata={"__display_name__": "LLM参数合并"}
        )
        graph.add_node(
            "parallel_parse", self._parallel_parse_node, metadata={"__display_name__": "并行解析"}
        )

        # ===== API 节点（显式主/兜底结构）=====
        graph.add_node(
            "api_strict", self._api_strict_node, metadata={"__display_name__": "主API调用"}
        )
        graph.add_node(
            "api_fallback", self._api_fallback_node, metadata={"__display_name__": "兜底API调用"}
        )
        graph.add_node(
            "merge_result", self._merge_result_node, metadata={"__display_name__": "结果汇聚"}
        )

        # ===== 后处理节点 =====
        graph.add_node(
            "post_process", self._post_process_node, metadata={"__display_name__": "后处理"}
        )
        graph.add_node(
            "has_result", self._has_result_node, metadata={"__display_name__": "有结果处理"}
        )
        graph.add_node(
            "no_result", self._no_result_node, metadata={"__display_name__": "无结果处理"}
        )
        graph.add_node(
            "package", self._package_result_node, metadata={"__display_name__": "封装结果"}
        )

        # ===== 预处理边 =====
        graph.set_entry_point("init_state")
        graph.add_edge("init_state", "pre_think")
        graph.add_edge("pre_think", "query_selections")
        graph.add_edge("query_selections", "llm_parse")
        graph.add_edge("llm_parse", "llm_merge")
        graph.add_edge("llm_merge", "parallel_parse")
        graph.add_edge("parallel_parse", "api_strict")

        # ===== 主 API -> 检查 -> 兜底/汇聚 =====
        graph.add_conditional_edges(
            "api_strict",
            self._check_strict_result,
            {"has_result": "merge_result", "no_result": "api_fallback"},
        )
        graph.add_edge("api_fallback", "merge_result")
        graph.add_edge("merge_result", "post_process")

        # ===== 后处理 -> 结果分支 =====
        graph.add_conditional_edges(
            "post_process",
            self._check_final_result,
            {"has_result": "has_result", "no_result": "no_result"},
        )

        graph.add_edge("has_result", "package")
        graph.add_edge("no_result", "package")
        graph.add_edge("package", END)

        return graph.compile()

    def _check_strict_result(self, state: DouyiWorkflowState) -> str:
        """检查主 API 结果"""
        result_count = state.get("result_count", 0)
        api_success = state.get("api_success", True)
        if not api_success or (result_count is not None and result_count < 50):
            return "no_result"
        return "has_result"

    def _check_final_result(self, state: DouyiWorkflowState) -> str:
        """检查最终结果是否有数据"""
        result_count = state.get("final_result_count") or state.get("result_count") or 0
        if result_count > 0:
            return "has_result"
        return "no_result"

    # ==================== 工具方法 ====================

    def _get_pusher(self, req: Any) -> PhaseProgressPusher:
        return PhaseProgressPusher(template=self.PROGRESS_TEMPLATE, request=req)

    def _build_time_data(self) -> str:
        """构建时间数据（对齐 n8n 时间数据节点输出为数组）"""
        now = datetime.now()
        current_time = now - timedelta(days=1)
        last_7_start = now - timedelta(days=8)
        last_7_end = now - timedelta(days=1)
        last_month_start = now - timedelta(days=31)
        last_month_end = now - timedelta(days=1)
        year = now.year
        is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

        time_info = {
            "当前时间": current_time.strftime("%Y-%m-%d"),
            "最近七天": f"{last_7_start.strftime('%Y-%m-%d')} ~ {last_7_end.strftime('%Y-%m-%d')}",
            "最近一个月": f"{last_month_start.strftime('%Y-%m-%d')}~{last_month_end.strftime('%Y-%m-%d')}",
            "当年春季": f"{datetime(year, 3, 1).strftime('%Y-%m-%d')} ~ {datetime(year, 5, 31).strftime('%Y-%m-%d')}",
            "当年冬季": f"{datetime(year, 12, 1).strftime('%Y-%m-%d')} ~ {datetime(year, 12, 31).strftime('%Y-%m-%d')}",
            "是否闰年": "是闰年" if is_leap_year else "不是闰年",
            "年份": f"{year}年",
        }
        return json.dumps([time_info], ensure_ascii=False)

    # ==================== 节点实现 ====================

    def _init_state_node(self, state: DouyiWorkflowState) -> dict[str, Any]:
        """初始化状态节点 - 埋点"""
        req = state["request"]
        logger.debug(f"[抖衣] Step 1: init_state - 用户查询: {req.user_query}")

        def insert_track():
            with pg_session(DBAlias.DB_ABROAD_AI.value) as session:
                session.execute(
                    text(
                        """
                        INSERT INTO holo_zhiyi_aiagent_query_prod(query, session_id, message_id, user_id, team_id)
                        VALUES (:query, :session_id, :message_id, :user_id, :team_id)
                    """
                    ),
                    {
                        "query": req.user_query,
                        "session_id": req.session_id,
                        "message_id": req.message_id,
                        "user_id": req.user_id,
                        "team_id": req.team_id,
                    },
                )

        thread_pool.submit_with_context(insert_track)
        return {}

    def _pre_think_node(self, state: DouyiWorkflowState) -> dict[str, Any]:
        """预处理节点 - 发送开始消息和思维链"""
        req = state["request"]

        # 推送开始消息
        start_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="收到任务",
            status="RUNNING",
            content_type=WorkflowMessageContentType.PRE_TEXT.value,
            content=WithActionContent(
                text="收到，我会为你选出想要的款式",
                actions=["view", "export", "download"],
                agent="search",
                data=ParameterData(entity_type=WorkflowEntityType.DY_ITEM.code),
            ),
            create_ts=int(round(time.time() * 1000)),
        )
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            start_message.model_dump_json(),
        )

        def _generate_thinking():
            ref_helper = QueryReferenceHelper.from_request(req)
            format_query = ref_helper.replace_placeholders(req.user_query)

            invoke_params = {
                "user_query": format_query,
                "preferred_entity": req.preferred_entity,
                "industry": req.industry.split("#")[0] if req.industry else "",
                "user_preferences": req.user_preferences,
                "now_time": datetime.now().strftime("%Y-%m-%d"),
            }

            thinking_text = ""
            try:
                messages: list[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
                    prompt_key=CozePromptHubKey.DOUYI_THINK_PROMPT.value,
                    variables=invoke_params,
                )
                llm: BaseChatModel = llm_factory.get_llm(
                    LlmProvider.OPENROUTER.name, LlmModelName.OPENROUTER_GEMINI_3_FLASH_PREVIEW.value
                )
                retry_llm = llm.with_retry(stop_after_attempt=2)
                thinking_chain = retry_llm | StrOutputParser()
                thinking_text = thinking_chain.with_config(run_name="思维链生成").invoke(messages)
                thinking_text = thinking_text.replace("\n\n", "\n").strip()
            except Exception as e:
                logger.warning(f"[抖衣] 思维链生成失败: {e}")

            # 推送思维链对应的阶段进度
            if thinking_text:
                self._get_pusher(req=req).complete_phase("选品任务规划", content=thinking_text)

        # 输出的比较慢，会晚于后面的输出消息，先改成同步调用解决问题。后续考虑这边用更快的模型
        _generate_thinking()
        return {}

    def _query_selection_node(self, state: DouyiWorkflowState) -> dict[str, Any]:
        """查询品类维表 + 检测品牌/店铺引用"""
        req = state["request"]

        # 先替换占位符，避免类目向量检索与后续 LLM 使用的查询文本不一致
        ref_helper = QueryReferenceHelper.from_request(req)
        display_query = ref_helper.replace_placeholders(req.user_query)

        # 品类向量检索
        category_vector_content = self._fetch_category_vector(
            display_query,
            VolcKnowledgeServiceId.DOUYI_CATEGORY_VECTOR,
            CozePromptHubKey.DOUYI_CATEGORY_VECTOR_CLEAN_PROMPT.value,
        )


        # 检测引用
        # 品牌引用
        brand_refs = ref_helper.get_by_type("DOUYI_BRAND")
        brand_name = None
        if brand_refs:
            brand_name = brand_refs[0].get("display_name")
            logger.debug(f"[抖衣] 检测到品牌引用: {brand_name}")
        
        # 店铺引用（兼容 DOUYIN_SHOP 和 DOUYI_SHOP 两种写法）
        shop_refs = ref_helper.get_by_type("DOUYIN_SHOP")
        if not shop_refs:
            shop_refs = ref_helper.get_by_type("DOUYI_SHOP")
        shop_id = None
        shop_name = None
        if shop_refs:
            shop_id = shop_refs[0].get("entity_id")
            shop_name = shop_refs[0].get("display_name")
            logger.debug(f"[抖衣] 检测到店铺引用: {shop_name} (ID: {shop_id})")

        selection_dict: dict[str, Any] = {
            "category_data": category_vector_content,
            "brand_name": brand_name,
            "shop_id": shop_id,
            "shop_name": shop_name,
        }

        logger.debug(
            f"[抖衣] Step 3: query_selections - 品类数量: {len(category_vector_content)}, "
            f"品牌: {brand_name}, 店铺: {shop_name}"
        )
        return {"selection_dict": selection_dict}

    def _llm_param_parse_node(self, state: DouyiWorkflowState) -> dict[str, Any]:
        """LLM 参数解析节点（拆分多维度子解析）"""
        req = state["request"]
        selection_dict = state["selection_dict"]

        raw_category_list = selection_dict.get("category_data", [])
        category_data = json.dumps([{"content_list": raw_category_list}], ensure_ascii=False)
        brand_name = selection_dict.get("brand_name")
        shop_id = selection_dict.get("shop_id")
        shop_name = selection_dict.get("shop_name")
        
        time_data = self._build_time_data()
        current_time = ""
        last_month = ""
        try:
            parsed_time = json.loads(time_data) if time_data else []
            if isinstance(parsed_time, list) and parsed_time and isinstance(parsed_time[0], dict):
                current_time = str(parsed_time[0].get("当前时间") or "")
                last_month = str(parsed_time[0].get("最近一个月") or "")
        except Exception:
            pass

        # 替换占位符并添加品牌/店铺信息
        ref_helper = QueryReferenceHelper.from_request(req)
        format_query = ref_helper.replace_placeholders(req.user_query)
        
        # 如果有品牌引用，添加到查询文本中
        if brand_name:
            format_query = f"{format_query}（品牌：{brand_name}）"
        
        # 如果有店铺引用，添加到查询文本中
        if shop_name:
            format_query = f"{format_query}（店铺：{shop_name}）"

        common_param = {
            "user_query": format_query,
            "user_preferences": req.user_preferences,
            "industry": req.industry,
        }

        def run_parse(
            prompt_key: str,
            schema: type[BaseModel],
            run_name: str,
            variables: dict[str, Any],
        ) -> BaseModel:
            try:
                messages: list[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
                    prompt_key=prompt_key,
                    variables=variables,
                )
                llm: BaseChatModel = llm_factory.get_llm(
                    LlmProvider.OPENROUTER.name,
                    LlmModelName.OPENROUTER_GEMINI_3_FLASH_PREVIEW.value,
                )
                structured_llm = llm.with_structured_output(schema).with_retry(
                    stop_after_attempt=2
                )
                return structured_llm.with_config(run_name=run_name).invoke(messages)
            except Exception as exc:
                logger.warning(f"[抖衣] {run_name} 解析失败: {exc}")
                return schema()

        category_result = run_parse(
            CozePromptHubKey.DOUYI_CATEGORY_PARSE_PROMPT.value,
            DouyiCategoryParseResult,
            "类目解析",
            {**common_param, "category_data": category_data},
        )

        time_future = thread_pool.submit_with_context(
            lambda: run_parse(
                CozePromptHubKey.DOUYI_TIME_PARSE_PROMPT.value,
                DouyiTimeParseResult,
                "时间解析",
                {
                    **common_param,
                    "time_data": time_data,
                    "current_time": current_time,
                    "last_month": last_month,
                },
            )
        )
        numeric_future = thread_pool.submit_with_context(
            lambda: run_parse(
                CozePromptHubKey.DOUYI_NUMERIC_PARSE_PROMPT.value,
                DouyiNumericParseResult,
                "数值解析",
                common_param,
            )
        )
        sales_future = thread_pool.submit_with_context(
            lambda: run_parse(
                CozePromptHubKey.DOUYI_SALES_FLAG_PARSE_PROMPT.value,
                DouyiSalesFlagParseResult,
                "销售方式解析",
                common_param,
            )
        )
        sort_future = thread_pool.submit_with_context(
            lambda: run_parse(
                CozePromptHubKey.DOUYI_SORT_INTENT_PARSE_PROMPT.value,
                DouyiSortIntentParseResult,
                "排序意图解析",
                common_param,
            )
        )
        properties_future = thread_pool.submit_with_context(
            lambda: run_parse(
                CozePromptHubKey.DOUYI_PROPERTIES_PARSE_PROMPT.value,
                DouyiPropertiesParseResult,
                "属性词解析",
                common_param,
            )
        )
        misc_future = thread_pool.submit_with_context(
            lambda: run_parse(
                CozePromptHubKey.DOUYI_MISC_PARSE_PROMPT.value,
                DouyiMiscParseResult,
                "标题画像解析",
                common_param,
            )
        )

        time_result = time_future.result()
        numeric_result = numeric_future.result()
        sales_result = sales_future.result()
        sort_intent_result = sort_future.result()
        properties_result = properties_future.result()
        misc_result = misc_future.result()

        param_result = self._merge_param_result(
            category_result=category_result,
            time_result=time_result,
            numeric_result=numeric_result,
            sales_result=sales_result,
            sort_intent_result=sort_intent_result,
            properties_result=properties_result,
            misc_result=misc_result,
        )

        # 暂时禁用品牌过滤（抖衣品牌向量库未接入）
        # 原品牌处理逻辑保留（如需恢复可取消注释）
        # if shop_id and param_result.brand:
        #     logger.debug(f"[抖衣] 检测到店铺引用，清除LLM提取的品牌参数: {param_result.brand}")
        #     param_result.brand = None
        #
        # if brand_name and not param_result.brand:
        #     param_result.brand = brand_name
        #     logger.debug(f"[抖衣] 强制设置品牌参数: {brand_name}")
        if param_result.brand:
            logger.debug(f"[抖衣] 暂时忽略品牌参数: {param_result.brand}")
            param_result.brand = None

        # 将店铺ID保存到state（后续API调用时使用）
        logger.debug(
            f"[抖衣] Step 4: llm_parse - 解析结果: category={param_result.category_id_list}, "
            f"price={param_result.min_price}-{param_result.max_price}, brand={param_result.brand}, shop_id={shop_id}"
        )
        return {
            "param_result": param_result,
            "shop_id": shop_id,
        }

    @staticmethod
    def _merge_param_result(
        *,
        category_result: DouyiCategoryParseResult,
        time_result: DouyiTimeParseResult,
        numeric_result: DouyiNumericParseResult,
        sales_result: DouyiSalesFlagParseResult,
        sort_intent_result: DouyiSortIntentParseResult,
        properties_result: DouyiPropertiesParseResult,
        misc_result: DouyiMiscParseResult,
    ) -> DouyiParseParam:
        """合并子解析结果为主参数"""

        def normalize_text(value: Any) -> str | None:
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        def normalize_date_range(start: Optional[str], end: Optional[str]) -> tuple[str | None, str | None]:
            if start and end and start > end:
                return end, start
            return start, end

        min_price = numeric_result.min_price
        max_price = numeric_result.max_price
        if min_price is not None and max_price is not None and min_price > max_price:
            min_price, max_price = max_price, min_price

        limit = numeric_result.limit if numeric_result.limit is not None else 6000
        if limit > 6000:
            limit = 6000
        if limit < 0:
            limit = 0

        stats_start, stats_end = normalize_date_range(
            normalize_text(time_result.start_date),
            normalize_text(time_result.end_date),
        )
        record_start, record_end = normalize_date_range(
            normalize_text(time_result.put_on_sale_start_date),
            normalize_text(time_result.put_on_sale_end_date),
        )

        properties_value = properties_result.properties
        if properties_value is None:
            properties = ""
        else:
            properties = properties_value.strip()
            if not properties:
                properties = ""

        return DouyiParseParam(
            root_category_id=category_result.root_category_id,
            category_id_list=category_result.category_id_list or [],
            min_price=min_price,
            max_price=max_price,
            put_on_sale_start_date=record_start,
            put_on_sale_end_date=record_end,
            start_date=stats_start,
            end_date=stats_end,
            year_season=normalize_text(time_result.year_season),
            is_monitor_shop=sales_result.is_monitor_shop or 0,
            is_monitor_streamer=sales_result.is_monitor_streamer or 0,
            sort_field=normalize_text(sort_intent_result.sort_field) or "默认",
            limit=limit,
            sale_style=normalize_text(sales_result.sale_style),
            properties=properties,
            has_live_sale=sales_result.has_live_sale,
            has_video_sale=sales_result.has_video_sale,
            has_card_sale=sales_result.has_card_sale,
            only_live_sale=sales_result.only_live_sale,
            only_video_sale=sales_result.only_video_sale,
            only_card_sale=sales_result.only_card_sale,
            title=normalize_text(misc_result.title),
            user_data=misc_result.user_data if misc_result.user_data is not None else 0,
            type=normalize_text(sort_intent_result.type),
            brand=normalize_text(misc_result.brand),
        )

    def _llm_param_merge_node(self, state: DouyiWorkflowState) -> dict[str, Any]:
        """LLM 合并节点：修正冲突/过窄条件"""
        req = state["request"]
        param_result: DouyiParseParam = state["param_result"]

        ref_helper = QueryReferenceHelper.from_request(req)
        format_query = ref_helper.replace_placeholders(req.user_query)

        current_time = datetime.now().strftime("%Y-%m-%d")
        min_price_text = (
            str(param_result.min_price) if param_result.min_price is not None else ""
        )
        max_price_text = (
            str(param_result.max_price) if param_result.max_price is not None else ""
        )

        prompt_param = {
            "user_query": format_query,
            "user_preferences": req.user_preferences or "",
            "industry": req.industry or "",
            "current_time": current_time,
            "category_id_list": json.dumps(param_result.category_id_list or [], ensure_ascii=False),
            "properties": param_result.properties or "",
            "brand": param_result.brand or "",
            "title": param_result.title or "",
            "type": param_result.type or "",
            "sort_field": param_result.sort_field or "",
            "min_price": min_price_text,
            "max_price": max_price_text,
            "start_date": param_result.start_date or "",
            "end_date": param_result.end_date or "",
            "put_on_sale_start_date": param_result.put_on_sale_start_date or "",
            "put_on_sale_end_date": param_result.put_on_sale_end_date or "",
        }

        try:
            messages: list[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
                prompt_key=CozePromptHubKey.DOUYI_PARAM_MERGE_PROMPT.value,
                variables=prompt_param,
            )
            llm: BaseChatModel = llm_factory.get_llm(
                LlmProvider.OPENROUTER.name,
                LlmModelName.OPENROUTER_GEMINI_3_FLASH_PREVIEW.value,
            )
            structured_llm = llm.with_structured_output(DouyiParamMergeResult).with_retry(
                stop_after_attempt=2
            )
            merge_result = structured_llm.with_config(run_name="参数合并").invoke(messages)
        except Exception as exc:
            logger.warning(f"[抖衣] 参数合并失败: {exc}")
            return {}

        updated = param_result.model_copy(deep=True)
        for field_name in (
            "properties",
            "year_season",
            "start_date",
            "end_date",
            "put_on_sale_start_date",
            "put_on_sale_end_date",
            "min_price",
            "max_price",
            "has_live_sale",
            "has_video_sale",
            "has_card_sale",
            "only_live_sale",
            "only_video_sale",
            "only_card_sale",
            "is_monitor_shop",
            "is_monitor_streamer",
            "sort_field",
            "type",
            "title",
        ):
            value = getattr(merge_result, field_name)
            if value is not None:
                if isinstance(value, str):
                    value = value.strip()
                    value = value or None
                setattr(updated, field_name, value)
        return {"param_result": updated}

    def _parallel_parse_node(self, state: DouyiWorkflowState) -> dict[str, Any]:
        """并行解析节点 - 属性 + 用户画像 + 排序（全部并行）"""
        req = state["request"]
        param_result: DouyiParseParam = state["param_result"]

        ref_helper = QueryReferenceHelper.from_request(req)
        format_query = ref_helper.replace_placeholders(req.user_query)

        futures = {}

        # 属性解析
        if param_result.properties and param_result.properties.strip():
            futures["properties"] = thread_pool.submit_with_context(
                self._parse_properties, req, format_query, param_result.properties, param_result
            )

        # 用户画像 + 标签
        if param_result.user_data == 1:
            futures["profile"] = thread_pool.submit_with_context(self._fetch_user_profile_tags, req.user_id)

        # 排序解析（并入线程池）
        futures["sort"] = thread_pool.submit_with_context(
            self._parse_sort, format_query, param_result.sort_field, param_result.type
        )

        # 用户 filters 查询（对齐 n8n 拿到用选品户画像1）
        if param_result.root_category_id and param_result.category_id_list:
            futures["filters"] = thread_pool.submit_with_context(
                self._fetch_user_filters,
                req.user_id,
                param_result.root_category_id,
                param_result.category_id_list,
            )

        # 收集结果
        property_list: list[list[str]] = []
        user_tags: list[str] = []
        sort_result: DouyiSortTypeParseResult | None = None
        user_filters: list[Any] = []

        for key, future in futures.items():
            try:
                result = future.result(timeout=20)
                if key == "properties":
                    property_list = result or []
                elif key == "profile":
                    user_tags = result or []
                elif key == "sort":
                    sort_result = result
                elif key == "filters":
                    user_filters = result or []
            except Exception as e:
                logger.warning(f"[并行解析] {key} 失败: {e}")

        # 记录 LLM 参数（对齐 n8n 埋点1）
        self._track_llm_parameters(
            req, param_result, property_list, sort_result, user_tags, user_filters
        )

        logger.debug(
            f"[抖衣] Step 5: parallel_parse - 属性={property_list}, 用户画像标签={user_tags}, 排序={sort_result}"
        )
        return {
            "property_list": property_list,
            "sort_param_result": sort_result,
            "user_filters": user_filters,
            "user_tags": user_tags,
        }

    def _parse_sort(
        self, user_query: str, sort_field: str | None, is_new_flag: str | None
    ) -> DouyiSortTypeParseResult | None:
        """解析排序项（对齐 n8n 封装排序项）"""
        try:
            prompt_param = {
                "user_query": user_query,
                "sort_field": sort_field or "",
                "is_new_flag": is_new_flag or "",
            }
            messages = coze_loop_client_provider.get_langchain_messages(
                prompt_key=CozePromptHubKey.DOUYI_SORT_TYPE_PARSE_PROMPT.value,
                variables=prompt_param,
            )

            llm = llm_factory.get_llm(
                LlmProvider.OPENROUTER.name, LlmModelName.OPENROUTER_GEMINI_2_5_FLASH.value
            )
            structured_llm = llm.with_structured_output(DouyiSortTypeParseResult).with_retry(
                stop_after_attempt=2
            )

            result: DouyiSortTypeParseResult = structured_llm.with_config(
                run_name="排序解析节点"
            ).invoke(messages)
            if not result.sort_type_final:
                return DouyiSortTypeParseResult(
                    sort_type_final=DEFAULT_SORT_FIELD,
                    sort_type_final_name=DEFAULT_SORT_FIELD_NAME,
                )
            return result
        except Exception as e:
            logger.warning(f"[排序解析] 失败: {e}")
            return DouyiSortTypeParseResult(
                sort_type_final=DEFAULT_SORT_FIELD,
                sort_type_final_name=DEFAULT_SORT_FIELD_NAME,
            )

    def _parse_properties(
        self,
        req: Any,
        origin_text: str,
        properties_text: str,
        param_result: DouyiParseParam,
    ) -> list[list[str]]:
        """解析属性（对齐 n8n 封装多属性）"""
        try:
            cleaned_text = properties_text.replace("\n", " ").strip()
            if not cleaned_text:
                return []

            def _recall_property_candidates_from_vector() -> list[str]:
                kb_client = get_volcengine_kb_api()
                response = kb_client.simple_chat(
                    query=properties_text,
                    service_resource_id=VolcKnowledgeServiceId.DOUYI_PROPERTIES_VECTOR.value,
                )
                if not response.data or not response.data.result_list:
                    return []
                content_list: list[str] = []
                for item in response.data.result_list:
                    value = ""
                    for chunk in item.table_chunk_fields:
                        field_name = chunk.get("field_name", "")
                        field_value = chunk.get("field_value", "")
                        if field_name == "key":
                            value = field_value
                    if value:
                        content_list.append(value)
                return content_list

            content_list: list[str] = []
            allowed: dict[str, set[str] | None] = {}
            # TODO: 暂时禁用类目属性候选接口，先回退到向量召回
            # try:
            #     root_category_id = param_result.root_category_id
            #     if root_category_id is None:
            #         industry = getattr(req, "industry", "") or ""
            #         if "#" in industry:
            #             tail = industry.split("#")[-1].strip()
            #             if tail.isdigit():
            #                 root_category_id = int(tail)
            #     category_id_list = param_result.category_id_list or []
            #
            #     if root_category_id or category_id_list:
            #         api_client = get_douyin_common_api_client()
            #         options = api_client.get_property_selector_list(
            #             user_id=req.user_id,
            #             team_id=req.team_id,
            #             root_category_id=root_category_id,
            #             category_id_list=category_id_list,
            #         )
            #         content_list, allowed = self._normalize_property_options_to_candidates(options)
            # except Exception as e:
            #     logger.warning(f"[抖衣属性标签解析] 类目属性候选接口不可用，降级使用向量召回：{e}")

            if not content_list:
                content_list = _recall_property_candidates_from_vector()
            if not content_list:
                return []

            # 2. llm标签清洗
            structured_llm = llm_factory.get_llm(
                LlmProvider.OPENROUTER.name, LlmModelName.OPENROUTER_GEMINI_2_5_FLASH.value
            ).with_structured_output(schema=DouyiPropertyLlmCleanResult).with_retry(stop_after_attempt=2)
            messages = coze_loop_client_provider.get_langchain_messages(
                prompt_key=CozePromptHubKey.DOUYI_PROPERTY_CLEAN_PROMPT.value,
                variables={
                    "query_tag_list": json.dumps(content_list, ensure_ascii=False),
                    "origin_text": origin_text
                },
            )
            result: DouyiPropertyLlmCleanResult = structured_llm.with_config(run_name="召回属性清洗").invoke(messages)
            clean_tags: list[str] = self._filter_property_clean_tags(result.clean_tag_list, allowed)
            if not clean_tags:
                return []

            # 3. 组织二维数组格式
            grouped_property_list: list[list[str]] = self._group_property_tags(clean_tags)
            return grouped_property_list
        except Exception as e:
            logger.warning(f"[抖衣属性标签解析]发生异常：{e}")
            return []


    def _group_property_tags(self, tags: list[str]) -> list[list[str]]:
        """将属性标签按名称分组"""
        grouped: dict[str, list[str]] = {}
        for tag in tags:
            if not isinstance(tag, str) or not tag.strip():
                continue
            if ":" in tag:
                key = tag.split(":", 1)[0].strip()
            else:
                key = "其他"
            grouped.setdefault(key, []).append(tag.strip())
        return list(grouped.values())

    @staticmethod
    def _normalize_property_options_to_candidates(
        options: list[dict[str, Any]],
        *,
        max_tags: int = 2000,
        max_values_per_property: int = 50,
    ) -> tuple[list[str], dict[str, set[str] | None]]:
        """将类目属性候选转换为 tag 列表 + 允许映射"""
        allowed: dict[str, set[str] | None] = {}
        flattened: list[str] = []

        def pick_text(item: dict[str, Any], keys: tuple[str, ...]) -> str:
            for key in keys:
                value = item.get(key)
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    return text
            return ""

        def pick_values(item: dict[str, Any]) -> list[str]:
            for key in ("values", "valueList", "enum", "options", "items", "children"):
                raw = item.get(key)
                if raw is None:
                    continue
                if isinstance(raw, str):
                    parts = [p.strip() for p in raw.replace("，", ",").split(",")]
                    return [p for p in parts if p]
                if isinstance(raw, list):
                    results: list[str] = []
                    for elem in raw:
                        if elem is None:
                            continue
                        if isinstance(elem, str):
                            text = elem.strip()
                            if text:
                                results.append(text)
                            continue
                        if isinstance(elem, dict):
                            text = pick_text(elem, ("value", "name", "label", "text", "key"))
                            if text:
                                results.append(text)
                    return results
            return []

        for item in options:
            if len(flattened) >= max_tags:
                break
            if not isinstance(item, dict):
                continue
            name = pick_text(item, ("name", "propertyName", "propName", "key", "label", "attrName"))
            if not name:
                raw_kv = pick_text(item, ("value", "content", "text"))
                if raw_kv and ":" in raw_kv:
                    key, value = raw_kv.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    if key and value:
                        allowed.setdefault(key, set()).add(value)
                        flattened.append(f"{key}:{value}")
                continue

            values = pick_values(item)
            if not values:
                allowed.setdefault(name, None)
                continue

            value_set = allowed.get(name)
            if value_set is None:
                value_set = set()
                allowed[name] = value_set

            for value in values[:max_values_per_property]:
                if len(flattened) >= max_tags:
                    break
                value_set.add(value)
                flattened.append(f"{name}:{value}")

        return flattened, allowed

    @staticmethod
    def _filter_property_clean_tags(
        clean_tags: list[str],
        allowed: dict[str, set[str] | None],
    ) -> list[str]:
        """按候选属性过滤清洗结果"""
        if not clean_tags:
            return []
        if not allowed:
            return [tag for tag in clean_tags if isinstance(tag, str) and ":" in tag]

        results: list[str] = []
        for tag in clean_tags:
            if not isinstance(tag, str) or ":" not in tag:
                continue
            key, value = tag.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not key or not value:
                continue
            allowed_values = allowed.get(key)
            if not allowed_values:
                continue
            if value in allowed_values:
                results.append(f"{key}:{value}")
        return results

    def _fetch_user_filters(
        self, user_id: int, root_category_id: int, category_id_list: list[int]
    ) -> list[Any]:
        """获取用户画像 filters（对齐 n8n 拿到用选品户画像1）
        
        查询 ads_zhiyi_user_profile_recommend_goods_selection 表的 filters 字段
        """
        try:
            # 解析 category_id_list
            category_ids: list[int] = []
            for item in category_id_list or []:
                if isinstance(item, int):
                    category_ids.append(item)
                    continue
                value_text = str(item).strip()
                if value_text.isdigit():
                    category_ids.append(int(value_text))
            if not category_ids:
                return []

            # 构建 SQL: category_id in (...)
            placeholders = ",".join([f":cat_{i}" for i in range(len(category_ids))])
            query_params = {
                "user_id": user_id,
                "root_category_id": root_category_id,
            }
            for i, cid in enumerate(category_ids):
                query_params[f"cat_{i}"] = cid

            with pg_session(DBAlias.DB_ABROAD_AI.value) as session:
                row = (
                    session.execute(
                        text(
                            f"""
                            select filters
                            from public.ads_zhiyi_user_profile_recommend_goods_selection
                            where user_id = :user_id
                              and root_category_id = :root_category_id
                              and category_id in ({placeholders})
                              and filters is not null
                              and filters <> ''
                              and filters not like '%null%'
                            limit 1
                            """
                        ),
                        query_params,
                    )
                    .mappings()
                    .first()
                )

            if not row:
                return []

            filters_value = row.get("filters")
            if not filters_value:
                return []

            # 解析 JSON 字符串为数组
            if isinstance(filters_value, str):
                try:
                    parsed = json.loads(filters_value)
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    pass

            return []
        except Exception as e:
            logger.warning(f"[用户filters] 获取失败: {e}")
            return []

    def _fetch_user_style(self, user_id: int) -> dict[str, Any]:
        """获取用户画像风格"""
        try:
            with pg_session(DBAlias.DB_ABROAD_AI.value) as session:
                row = (
                    session.execute(
                        text(
                            """
                        select team_id, user_id, style
                        from public.ads_abroad_user_profile_recommend_goods_selection
                        where user_id = :user_id
                        limit 1
                        """
                        ),
                        {"user_id": user_id},
                    )
                    .mappings()
                    .first()
                )
            return dict(row) if row else {}
        except Exception as e:
            logger.warning(f"[用户画像] 获取失败: {e}")
            return {}

    def _fetch_user_profile_tags(self, user_id: int) -> list[str]:
        """获取用户画像并解析标签（对齐 n8n 用户画像标签流程）"""
        profile = self._fetch_user_style(user_id)
        style_text = profile.get("style") if isinstance(profile, dict) else ""
        if not isinstance(style_text, str) or not style_text.strip():
            return []
        payload = json.dumps([profile], ensure_ascii=False)
        return self._parse_user_tags(payload)

    def _parse_user_tags(self, style_text: str) -> list[str]:
        """解析用户画像标签（返回 3 个关键词）"""

        def extract_style(text: str) -> str:
            cleaned = text.strip()
            if not cleaned or cleaned[0] not in "[{":
                return cleaned
            try:
                parsed = json.loads(cleaned)
                items = parsed if isinstance(parsed, list) else [parsed]
                styles = []
                for item in items:
                    if isinstance(item, dict):
                        style_value = item.get("style")
                        if isinstance(style_value, str) and style_value.strip():
                            styles.append(style_value.strip())
                return ",".join(styles) if styles else cleaned
            except Exception:
                return cleaned

        try:
            llm = llm_factory.get_llm(
                LlmProvider.OPENROUTER.name, LlmModelName.OPENROUTER_GEMINI_2_5_FLASH.value
            )
            structured_llm = llm.with_structured_output(DouyiUserTagResult).with_retry(
                stop_after_attempt=2
            )
            result: DouyiUserTagResult = structured_llm.with_config(
                run_name="用户画像标签解析节点"
            ).invoke(
                coze_loop_client_provider.get_langchain_messages(
                    prompt_key=CozePromptHubKey.DOUYI_USER_TAG_PARSE_PROMPT.value,
                    variables={"user_select_message": style_text},
                )
            )
            values = (result.values or "").strip()
            if not values:
                return []
            return [item.strip() for item in values.split(",") if item.strip()][:3]
        except Exception as e:
            logger.warning(f"[用户画像标签] 解析失败: {e}")
            fallback_source = extract_style(style_text)
            fallback = [item.strip() for item in fallback_source.split(",") if item.strip()]
            return fallback[:3]

    def _track_llm_parameters(
        self,
        req: Any,
        param_result: DouyiParseParam,
        property_list: list[list[str]],
        sort_result: DouyiSortTypeParseResult | None,
        user_tags: list[str],
        user_filters: list[Any],
    ) -> None:
        """记录 LLM 参数解析结果（对齐 n8n 埋点1）"""
        param_payload = param_result.model_dump(by_alias=True, exclude_none=True)
        if "minPrice" in param_payload:
            param_payload["minSprice"] = param_payload.pop("minPrice")
        if "maxPrice" in param_payload:
            param_payload["maxSprice"] = param_payload.pop("maxPrice")
        if "yearSeason" in param_payload:
            param_payload["year_season"] = param_payload.pop("yearSeason")
        if "saleStyle" in param_payload:
            param_payload["sale_style"] = param_payload.pop("saleStyle")
        if "userData" in param_payload:
            param_payload["user_data"] = param_payload.pop("userData")
        param_payload["userid"] = req.user_id
        param_payload["teamid"] = req.team_id

        sort_field = DEFAULT_SORT_FIELD
        sort_name = DEFAULT_SORT_FIELD_NAME
        if sort_result and sort_result.sort_type_final:
            sort_field = sort_result.sort_type_final
            sort_name = sort_result.sort_type_final_name or DEFAULT_SORT_FIELD_NAME

        sort_payload = {"output": {"sortField_new": sort_field, "sortField_new_name": sort_name}}
        values = ",".join(user_tags) if user_tags else ""
        user_tag_payload = {"output": {"values": values}}

        filters_payload = {"output": {"user_tags": user_filters if user_filters is not None else []}}
        llm_parameters = json.dumps(
            [
                {"propertyList": property_list},
                {"output": param_payload},
                sort_payload,
                user_tag_payload,
                filters_payload,
            ],
            ensure_ascii=False,
        )

        def insert_track():
            with pg_session(DBAlias.DB_ABROAD_AI.value) as session:
                session.execute(
                    text(
                        """
                        INSERT INTO holo_zhiyi_aiagent_query_prod(query, session_id, message_id, user_id, team_id, llm_parameters)
                        VALUES (:query, :session_id, :message_id, :user_id, :team_id, :llm_parameters)
                    """
                    ),
                    {
                        "query": req.user_query,
                        "session_id": req.session_id,
                        "message_id": req.message_id,
                        "user_id": req.user_id,
                        "team_id": req.team_id,
                        "llm_parameters": llm_parameters,
                    },
                )

        thread_pool.submit_with_context(insert_track)

    def _extract_keyword_from_properties(self, property_list: list[list[str]]) -> str:
        """从属性列表提取关键词"""
        try:
            first = property_list[0][0] if property_list and property_list[0] else ""
            if not isinstance(first, str) or not first.strip() or ":" not in first:
                return ""
            _, value = first.split(":", 1)
            return value.strip()
        except Exception:
            return ""

    def _apply_sort_field(
        self, api_request: DouyiSearchRequest, sort_result: DouyiSortTypeParseResult | None
    ) -> None:
        """设置排序字段（对齐 n8n 默认值）"""
        if sort_result and sort_result.sort_type_final:
            api_request.sort_field = sort_result.sort_type_final
        else:
            api_request.sort_field = DEFAULT_SORT_FIELD

    # ==================== goods-list API 辅助方法（店铺查询专用）====================

    def _normalize_goods_list_category_id_list(
        self, param_result: DouyiParseParam
    ) -> list[int]:
        """将类目ID列表转换为整数列表"""
        raw_list = param_result.category_id_list or []
        if not raw_list:
            return []
        ids: list[int] = []
        for item in raw_list:
            if isinstance(item, int):
                ids.append(item)
                continue
            text = str(item).strip()
            if text.isdigit():
                ids.append(int(text))
        return ids

    def _resolve_goods_list_time_range(
        self, param_result: DouyiParseParam
    ) -> tuple[str | None, str | None, str | None, str | None]:
        """解析 goods-list API 的时间范围，只有 LLM 明确解析出时间时才设置"""
        stats_start = param_result.start_date
        stats_end = param_result.end_date
        record_start = param_result.put_on_sale_start_date
        record_end = param_result.put_on_sale_end_date
        return stats_start, stats_end, record_start, record_end

    def _resolve_goods_list_keyword(
        self, param_result: DouyiParseParam, property_list: list[list[str]]
    ) -> str:
        """从属性列表中提取关键词用于 goods-list 搜索"""
        keyword = self._extract_keyword_from_properties(property_list)
        return keyword or ""

    def _apply_goods_list_sort(
        self, api_request: DouyiGoodsListRequest, sort_result: DouyiSortTypeParseResult | None
    ) -> None:
        """设置 goods-list API 的排序参数

        排序字段映射:
        - 1: 本期销量 (saleVolumeDaily)
        - 3: 收录时间 (firstRecordTime)
        - 8: 全网销量
        排序类型: 0=升序, 1=降序
        """
        sort_field = GOODS_LIST_DEFAULT_SORT_FIELD
        sort_type = GOODS_LIST_DEFAULT_SORT_TYPE

        if sort_result:
            sort_key = (sort_result.sort_type_final or "").strip()
            sort_name = (sort_result.sort_type_final_name or "").strip()
            if sort_key == "saleVolumeDaily":
                sort_field, sort_type = 1, 1
            elif sort_key:
                lower_key = sort_key.lower()
                if "firstrecord" in lower_key:
                    sort_field = 3
                    sort_type = 0 if "asc" in lower_key else 1

            if sort_field == GOODS_LIST_DEFAULT_SORT_FIELD and sort_type == GOODS_LIST_DEFAULT_SORT_TYPE:
                if "全网销量" in sort_name or "总销量" in sort_name:
                    sort_field, sort_type = 8, 1
                elif "本期销量" in sort_name or "30天销量" in sort_name or "近30天销量" in sort_name:
                    sort_field, sort_type = 1, 1
                elif "收录时间" in sort_name or "上架时间" in sort_name or "最新" in sort_name:
                    if "从旧" in sort_name or "最早" in sort_name or "旧到新" in sort_name:
                        sort_field, sort_type = 3, 0
                    else:
                        sort_field, sort_type = 3, 1

        api_request.sort_field = sort_field
        api_request.sort_type = sort_type

    def _build_goods_list_request(
        self,
        param_result: DouyiParseParam,
        shop_id: str,
        sort_result: DouyiSortTypeParseResult | None,
        property_list: list[list[str]],
        keyword_override: str | None = None,
    ) -> DouyiGoodsListRequest:
        """构建 goods-list API 请求，用于店铺商品查询

        与 common-search API 的区别:
        - 专用于店铺查询，通过 shopId 定位
        - 时间参数只在 LLM 明确解析出时才设置，避免过度收窄结果
        - 价格单位为分（需要 *100 转换）
        """
        stats_start, stats_end, record_start, record_end = self._resolve_goods_list_time_range(param_result)
        category_id_list = self._normalize_goods_list_category_id_list(param_result)
        keyword = keyword_override if keyword_override is not None else self._resolve_goods_list_keyword(
            param_result, property_list
        )
        min_price = param_result.min_price
        max_price = param_result.max_price
        if min_price == 0 and max_price and max_price >= 999999:
            min_price = None
            max_price = None
        # goods-list price unit is cents
        if min_price is not None:
            min_price = int(min_price) * 100
        if max_price is not None:
            max_price = int(max_price) * 100

        api_request = DouyiGoodsListRequest(
            shopId=str(shop_id),
            startDate=stats_start,
            endDate=stats_end,
            sortStartTime=record_start,
            sortEndTime=record_end,
            categoryIdList=category_id_list,
            minPrice=min_price,
            maxPrice=max_price,
            keyword=keyword,
        )
        self._apply_goods_list_sort(api_request, sort_result)
        return api_request

    def _maybe_clear_record_time_for_shop(
        self, api_request: DouyiSearchRequest, shop_id: str | None
    ) -> None:
        """店铺查询对齐商品库口径，避免收录时间过度收窄结果"""
        if not shop_id:
            return
        if api_request.min_first_record_date or api_request.max_first_record_date:
            api_request.min_first_record_date = None
            api_request.max_first_record_date = None
            api_request.first_record_date_type = None

    # ==================== API 节点（显式主/兜底结构）====================

    def _api_strict_node(self, state: DouyiWorkflowState) -> dict[str, Any]:
        """主 API 节点 - 使用完整 propertyList 的精确查询"""
        req = state["request"]
        param_result: DouyiParseParam = state["param_result"]
        sort_result: DouyiSortTypeParseResult | None = state.get("sort_param_result")
        property_list = state.get("property_list") or []
        user_filters = state.get("user_filters", [])
        shop_id = state.get("shop_id")

        zhiyi_client = get_zhiyi_api_client()
        api_request = None
        try:
            if shop_id:
                api_request = self._build_goods_list_request(
                    param_result, shop_id, sort_result, property_list
                )
                page_result: PageResult[DouyiGoodsEntity] = zhiyi_client.goods_list_douyi(
                    user_id=req.user_id, team_id=req.team_id, params=api_request
                )
            else:
                # 构建主请求（精确匹配）
                api_request = DouyiSearchRequest.from_parse_param(param_result)
                api_request.sale_style = None
                api_request.hot_properties = property_list if property_list else []
                api_request.keyword = ""

                self._maybe_clear_record_time_for_shop(api_request, shop_id)
                self._apply_sort_field(api_request, sort_result)
                page_result = zhiyi_client.common_search_douyi(
                    user_id=req.user_id, team_id=req.team_id, params=api_request
                )
            result_count = page_result.result_count
            if result_count is None:
                result_count = len(page_result.result_list)

            logger.debug(f"[抖衣] Step 6: api_strict - 成功, 结果数量: {result_count}")
            return {
                "api_request": api_request,
                "api_resp": page_result,
                "result_count": result_count,
                "api_success": True,
                "user_filters": user_filters,
                "browsed_count": result_count,
            }
        except Exception as e:
            error_reason = f"主API调用失败: {str(e)}"
            logger.warning(f"[抖衣] {error_reason}")
            logger.debug(f"[抖衣] Step 6: api_strict - 失败: {error_reason}")
            return {
                "api_request": api_request,
                "api_resp": None,
                "result_count": 0,
                "api_success": False,
                "user_filters": user_filters,
                "browsed_count": 0,
                "api_error_reason": error_reason,
            }

    def _api_fallback_node(self, state: DouyiWorkflowState) -> dict[str, Any]:
        """兜底 API 节点 - 移除 propertyList，改用 keyword 查询"""
        req = state["request"]
        param_result: DouyiParseParam = state["param_result"]
        sort_result: DouyiSortTypeParseResult | None = state.get("sort_param_result")
        property_list = state.get("property_list") or []
        user_filters = state.get("user_filters", [])
        prev_browsed = state.get("browsed_count", 0)
        shop_id = state.get("shop_id")

        self._get_pusher(req=req).complete_phase(
            phase_name="商品筛选中",
            variables={
                "datasource": "抖音商品库",
                "platform": "抖音",
                "browsed_count": str(random.randint(100000, 1000000)),
                "filter_result_text": "当前筛选出的商品数量不足，不满足列表需求，我将再次搜索确保无遗漏",
            },
        )

        zhiyi_client = get_zhiyi_api_client()
        api_request = None
        try:
            if shop_id:
                api_request = self._build_goods_list_request(
                    param_result, shop_id, sort_result, property_list
                )
                page_result: PageResult[DouyiGoodsEntity] = zhiyi_client.goods_list_douyi(
                    user_id=req.user_id, team_id=req.team_id, params=api_request
                )
            else:
                # 构建兜底请求（放宽条件）
                api_request = DouyiSearchRequest.from_parse_param(param_result)
                api_request.sale_style = None
                api_request.hot_properties = []  # 移除属性词
                api_request.keyword = self._extract_keyword_from_properties(property_list) or ""

                self._maybe_clear_record_time_for_shop(api_request, shop_id)
                self._apply_sort_field(api_request, sort_result)
                page_result = zhiyi_client.common_search_douyi(
                    user_id=req.user_id, team_id=req.team_id, params=api_request
                )
            result_count = page_result.result_count
            if result_count is None:
                result_count = len(page_result.result_list)

            # 更新浏览数量
            total_browsed = prev_browsed + result_count

            return {
                "fallback_api_request": api_request,
                "fallback_api_resp": page_result,
                "fallback_result_count": result_count,
                "fallback_api_success": True,
                "user_filters": user_filters,
                "browsed_count": total_browsed,
            }
        except Exception as e:
            error_reason = f"兜底API调用失败: {str(e)}"
            logger.warning(f"[抖衣] {error_reason}")
            return {
                "fallback_api_request": api_request,
                "fallback_api_resp": None,
                "fallback_result_count": 0,
                "fallback_api_success": False,
                "user_filters": user_filters,
                "browsed_count": prev_browsed,
                "fallback_error_reason": error_reason,
            }

    def _merge_result_node(self, state: DouyiWorkflowState) -> dict[str, Any]:
        """汇聚节点 - 合并主调用和兜底调用结果"""
        req = state["request"]
        # 主调用结果
        api_resp = state.get("api_resp")
        result_count = state.get("result_count", 0)
        api_success = state.get("api_success", False)
        api_request = state.get("api_request")

        # 兜底调用结果
        fallback_resp = state.get("fallback_api_resp")
        fallback_count = state.get("fallback_result_count", 0)
        fallback_success = state.get("fallback_api_success", False)
        fallback_request = state.get("fallback_api_request")

        # 是否走了兜底路径
        fallback_attempted = fallback_request is not None

        # 选择最终结果：优先使用主调用；兜底有结果则用兜底；兜底无结果时保留主调用结果
        if api_success and result_count >= 50:
            final_resp = api_resp
            final_count = result_count
            final_request = api_request
        elif fallback_success and fallback_count > 0:
            final_resp = fallback_resp
            final_count = fallback_count
            final_request = fallback_request
        elif api_success and result_count > 0:
            final_resp = api_resp
            final_count = result_count
            final_request = api_request
        else:
            # 都失败或都无结果，按无结果处理
            final_resp = None
            final_count = 0
            final_request = fallback_request or api_request

        browsed_count = state.get("browsed_count", 0) or max(result_count or 0, fallback_count or 0)

        # 是否最终使用了兜底结果
        used_fallback = fallback_resp is not None and final_resp is fallback_resp

        # 推送进度 - 对齐 INS 工作流格式
        pusher = self._get_pusher(req=req)
        if fallback_attempted:
            fallback_has_results = fallback_count > 0
            if fallback_has_results and final_count > 0:
                retry_text = (
                    "商品筛选完成，当前筛选出的商品数量符合报表要求，接下来我将根据数据绘制商品列表"
                )
                browsed_count_display = str(random.randint(100000, 1000000))
            else:
                retry_text = f"当前筛选出的商品数量不满足需求，我可能需要提醒用户调整筛选维度"
                browsed_count_display = str(random.randint(100000, 1000000))
            pusher.complete_phase(
                phase_name="二次筛选中",
                variables={
                    "datasource": "抖音商品库",
                    "platform": "抖音",
                    "browsed_count": browsed_count_display,
                    "retry_result_text": retry_text,
                },
            )
        else:
            browsed_count_display = str(random.randint(100000, 1000000))
            pusher.complete_phase(
                phase_name="商品筛选中",
                variables={
                    "datasource": "抖音商品库",
                    "platform": "抖音",
                    "browsed_count": browsed_count_display,
                    "filter_result_text": "商品筛选完成，当前筛选出的商品数量满足需求，无需二次筛选，接下来我将根据数据完成商品列表",
                },
            )

        # 转移error_reason字段，用于最终的错误提示
        result = {
            "api_request": final_request,
            "api_resp": final_resp,
            "result_count": final_count,
            "browsed_count": browsed_count,
            "used_fallback": used_fallback,
        }

        # 错误原因透传：无结果时保留兜底错误
        if state.get("api_error_reason") and not used_fallback:
            result["api_error_reason"] = state.get("api_error_reason")
        if state.get("fallback_error_reason") and (used_fallback or final_count == 0):
            result["fallback_error_reason"] = state.get("fallback_error_reason")

        logger.debug(
            f"[抖衣] Step 8: merge_result - 最终结果数量: {final_count}, 走兜底: {used_fallback}"
        )
        return result

    def _post_process_node(self, state: DouyiWorkflowState) -> dict[str, Any]:
        """后处理节点"""
        api_resp: PageResult[DouyiGoodsEntity] | None = state.get("api_resp")

        if not api_resp or not api_resp.result_list:
            logger.debug("[抖衣] Step 9: post_process - 无结果")
            return {
                "final_result_count": 0,
            }

        final_count = len(api_resp.result_list)
        logger.debug(f"[抖衣] Step 9: post_process - 结果数量: {final_count}")
        return {
            "final_result_count": final_count,
        }

    def _has_result_node(self, state: DouyiWorkflowState) -> dict[str, Any]:
        """有结果分支"""
        logger.debug("[抖衣] Step 10: has_result - 进入有结果分支")
        req = state["request"]
        api_request: DouyiSearchRequest | DouyiGoodsListRequest = state["api_request"]
        param_result: DouyiParseParam = state["param_result"]
        page_result: PageResult[DouyiGoodsEntity] = state["api_resp"]
        user_filters = state.get("user_filters", [])

        pusher = self._get_pusher(req=req)
        pusher.complete_phase("生成列表中")
        pusher.complete_phase("选品完成")

        # 推送任务状态
        task_status_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="任务状态",
            status="RUNNING",
            content_type=WorkflowMessageContentType.TASK_STATUS.value,
            content=ParameterDataContent(data=ParameterData(task_status=1)),
            create_ts=int(round(time.time() * 1000)),
        )
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            task_status_message.model_dump_json(),
        )

        # 推送数据源消息
        path_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="输出参数",
            status="RUNNING",
            content_type=8,
            content=CustomDataContent(data={"entity_type": 1, "content": "dy-item-all"}),
            create_ts=int(round(time.time() * 1000)),
        )
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            path_message.model_dump_json(),
        )

        # 推送输出参数消息（对齐 INS 工作流格式）
        parameter_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="输出参数",
            status="END",
            content_type=5,
            content=ParameterDataContent(
                data=ParameterData(
                    request_path="goods-center/goods-zone-list",
                    request_body=api_request.model_dump_json(by_alias=True, exclude_none=True),
                    actions=["view", "export", "download"],
                    title=param_result.title,
                    entity_type=2,
                    filters=user_filters if user_filters is not None else [],
                )
            ),
            create_ts=int(round(time.time() * 1000)),
        )
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            parameter_message.model_dump_json(),
        )

        # 构建商品简化列表（包含展示所需的关键字段）
        entity_list = [
            {
                "商品id": e.goods_id,
                "title": e.title,
                "shopName": e.shop_name,
                "price": e.price,
                "saleVolume": e.sale_volume_30day,
                "imageUrl": e.pic_url,
            }
            for e in page_result.result_list
        ]

        return {
            "has_query_result": True,
            "entity_simple_data": entity_list,
        }

    def _no_result_node(self, state: DouyiWorkflowState) -> dict[str, Any]:
        """无结果分支"""
        req = state["request"]

        pusher = self._get_pusher(req=req)
        pusher.fail_phase(phase_name="生成列表失败", error_message="我未能完成列表绘制，原因是没有数据\n需要提醒用户调整筛选维度，才能更好的获取数据",)
        pusher.fail_phase(phase_name="选品未完成", error_message=None)

        # 推送任务状态
        task_status_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="任务状态",
            status="RUNNING",
            content_type=WorkflowMessageContentType.TASK_STATUS.value,
            content=ParameterDataContent(data=ParameterData(task_status=0)),
            create_ts=int(round(time.time() * 1000)),
        )
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            task_status_message.model_dump_json(),
        )

        # 优先检查是否有API错误原因
        api_error = state.get("api_error_reason")
        fallback_error = state.get("fallback_error_reason")

        # 构建失败消息（优先显示API错误原因）
        if api_error:
            fail_text = f"未找到符合需求的商品。原因: {api_error}"
        elif fallback_error:
            fail_text = f"未找到符合需求的商品。原因: {fallback_error}"
        else:
            fail_text = "未找到符合需求的商品，请尝试调整筛选条件。"

        # 推送无结果消息
        # no_result_message = BaseRedisMessage(
        #     session_id=req.session_id,
        #     reply_message_id=req.message_id,
        #     reply_id=f"reply_{req.message_id}",
        #     reply_seq=0,
        #     operate_id="结果",
        #     status="END",
        #     content_type=1,
        #     content=TextMessageContent(text=fail_text),
        #     create_ts=int(round(time.time() * 1000)),
        # )
        # redis_client.list_left_push(
        #     RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
        #     no_result_message.model_dump_json(),
        # )

        return {"has_query_result": False}

    def _package_result_node(self, state: DouyiWorkflowState) -> dict[str, Any]:
        """封装返回结果节点"""
        req = state.get("request")
        has_result = state.get("has_query_result", False)
        entity_data = state.get("entity_simple_data", [])
        if has_result:
            response = WorkflowResponse(
                select_result="基于以上条件，您的选品任务已经完成，还有其他需要帮助的地方吗？",
                relate_data=json.dumps(entity_data, ensure_ascii=False),
            )
        else:
            response = WorkflowResponse(
                select_result="无结果，可能与价格、销量、参考历史选品经验有关",
                relate_data=None,
            )
        logger.debug(
            f"[抖衣] Step 12: package - 最终响应: has_result={has_result}, entity_count={len(entity_data)}"
        )
        return {"workflow_response": response}


__all__ = ["DouyiGraph"]
