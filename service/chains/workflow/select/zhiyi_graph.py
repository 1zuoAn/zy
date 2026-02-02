"""
知衣选品工作流 - LangGraph 版本
"""

from __future__ import annotations

import json
import re
import random
import time
from datetime import datetime, timedelta
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from pydantic import BaseModel, TypeAdapter
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.config import settings
from app.core.clients.coze_loop_client import coze_loop_client_provider
from app.core.clients.db_client import mysql_session_readonly, pg_session
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
    ZhiyiDataSourceKey,
)
from app.core.errors import AppException, ErrorCode
from app.core.tools import llm_factory
from app.schemas.entities.message.redis_message import (
    BaseRedisMessage,
    CustomDataContent,
    ParameterData,
    ParameterDataContent,
    TextMessageContent,
    UserTagFilterItem,
    WithActionContent,
)
from app.schemas.entities.workflow.graph_state import ZhiyiWorkflowState
from app.schemas.entities.workflow.llm.zhiyi_output import (
    BranchRequestSpec,
    RequestContext,
    ZhiyiApiBranch,
    ZhiyiBrandPhraseParseResult,
    ZhiyiCategoryParseResult,
    ZhiyiExtendedRequest,
    ZhiyiLlmParametersItem,
    ZhiyiNumericParseResult,
    ZhiyiParamMergeResult,
    ZhiyiParseParam,
    ZhiyiPropertyLlmCleanResult,
    ZhiyiRouteParseResult,
    ZhiyiShopLlmCleanResult,
    ZhiyiShopRequest,
    ZhiyiSortResult,
    ZhiyiStandardRequest,
    ZhiyiTimeParseResult,
    ZhiyiUserTagResult,
)
from app.schemas.request.workflow_request import QueryReferenceEntityType, WorkflowRequest
from app.schemas.response.workflow_response import WorkflowResponse
from app.service.chains.templates.zhiyi_progress_template import ZHIYI_PROGRESS_TEMPLATE
from app.service.chains.workflow.base_graph import BaseWorkflowGraph
from app.service.chains.workflow.progress_pusher import PhaseProgressPusher
from app.service.rpc.volcengine_kb_api import get_volcengine_kb_api
from app.service.rpc.zhiyi.client import get_zhiyi_api_client
from app.utils import thread_pool
from app.utils.query_reference import QueryReferenceHelper


class ZhiyiGraph(BaseWorkflowGraph):
    """知衣选品工作流"""

    span_name = "知衣选品工作流"
    run_name = "zhiyi-graph"

    # 使用 templates 文件夹的进度模板
    PROGRESS_TEMPLATE = ZHIYI_PROGRESS_TEMPLATE
    _QUERY_TITLE_TERM_LIMIT = 2
    _BRANCH_REQUEST_SPECS: dict[tuple[ZhiyiApiBranch, bool], BranchRequestSpec] = {
        ("shop", False): BranchRequestSpec(
            kind="shop",
            include_property_list=False,
            query_title_mode="main",
            page_size=100,
        ),
        ("shop", True): BranchRequestSpec(
            kind="shop",
            include_property_list=False,
            query_title_mode="main",
            page_size=100,
        ),
        ("monitor_hot", False): BranchRequestSpec(
            kind="standard",
            group_id_list=[-3, -4],
            include_property_list=True,
            query_title_mode="main",
        ),
        ("monitor_hot", True): BranchRequestSpec(
            kind="standard",
            group_id_list=[-3, -4],
            include_property_list=False,
            query_title_mode="main",
        ),
        ("all_hot", False): BranchRequestSpec(
            kind="standard",
            group_id_list=[],
            include_property_list=True,
            query_title_mode="main",
        ),
        ("all_hot", True): BranchRequestSpec(
            kind="standard",
            group_id_list=[],
            include_property_list=False,
            query_title_mode="main",
        ),
        ("all_new", False): BranchRequestSpec(
            kind="standard",
            group_id_list=[],
            include_property_list=True,
            query_title_mode="main",
        ),
        ("all_new", True): BranchRequestSpec(
            kind="extended",
            group_id_list=[],
            page_no=10,
            page_size=1,
            include_property_list=False,
            include_start_date=False,
            query_title_mode="main",
            use_param_shop_filter=False,
        ),
        ("monitor_new", False): BranchRequestSpec(
            kind="extended",
            group_id_list=[-3, -4],
            page_no=10,
            page_size=1,
            include_property_list=True,
            include_start_date=False,
            query_title_mode="main",
            use_param_shop_filter=False,
        ),
        ("monitor_new", True): BranchRequestSpec(
            kind="extended",
            group_id_list=[-3, -4],
            page_no=10,
            page_size=1,
            include_property_list=False,
            include_start_date=False,
            query_title_mode="main",
            use_param_shop_filter=False,
        ),
    }

    # 节点名称中文映射（用于 CozeLoop 追踪显示）
    _NODE_NAME_MAP = {
        "init_state": "初始化",
        "pre_think": "预处理思考",
        "query_selections": "查询筛选项",
        "llm_parse": "LLM参数解析",
        "llm_merge": "LLM参数合并",
        "parallel_parse": "并行解析",
        "api_shop": "店铺API",
        "api_monitor_hot": "监控热销API",
        "api_monitor_new": "监控新品API",
        "api_all_hot": "全网热销API",
        "api_all_new": "全网新品API",
        "fallback_shop": "店铺兜底",
        "fallback_monitor_hot": "监控热销兜底",
        "fallback_monitor_new": "监控新品兜底",
        "fallback_all_hot": "全网热销兜底",
        "fallback_all_new": "全网新品兜底",
        "merge_shop": "店铺结果汇聚",
        "merge_monitor_hot": "监控热销汇聚",
        "merge_monitor_new": "监控新品汇聚",
        "merge_all_hot": "全网热销汇聚",
        "merge_all_new": "全网新品汇聚",
        "post_process": "后处理",
        "has_result": "结果输出",
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
        """构建工作流图"""
        graph = StateGraph(ZhiyiWorkflowState)

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

        # ===== 主 API 节点 =====
        graph.add_node(
            "api_shop",
            self._api_shop_node,
            metadata={"__display_name__": "店铺API"},
        )
        graph.add_node(
            "api_monitor_hot",
            self._api_monitor_hot_node,
            metadata={"__display_name__": "监控热销API"},
        )
        graph.add_node(
            "api_monitor_new",
            self._api_monitor_new_node,
            metadata={"__display_name__": "监控新品API"},
        )
        graph.add_node(
            "api_all_hot", self._api_all_hot_node, metadata={"__display_name__": "全网热销API"}
        )
        graph.add_node(
            "api_all_new", self._api_all_new_node, metadata={"__display_name__": "全网新品API"}
        )

        # ===== 兜底 API 节点 =====
        graph.add_node(
            "fallback_shop",
            self._fallback_shop_node,
            metadata={"__display_name__": "店铺兜底"},
        )
        graph.add_node(
            "fallback_monitor_hot",
            self._fallback_monitor_hot_node,
            metadata={"__display_name__": "监控热销兜底"},
        )
        graph.add_node(
            "fallback_monitor_new",
            self._fallback_monitor_new_node,
            metadata={"__display_name__": "监控新品兜底"},
        )
        graph.add_node(
            "fallback_all_hot",
            self._fallback_all_hot_node,
            metadata={"__display_name__": "全网热销兜底"},
        )
        graph.add_node(
            "fallback_all_new",
            self._fallback_all_new_node,
            metadata={"__display_name__": "全网新品兜底"},
        )

        # ===== 汇聚节点 =====
        graph.add_node(
            "merge_shop", self._merge_node, metadata={"__display_name__": "店铺结果汇聚"}
        )
        graph.add_node(
            "merge_monitor_hot", self._merge_node, metadata={"__display_name__": "监控热销汇聚"}
        )
        graph.add_node(
            "merge_monitor_new", self._merge_node, metadata={"__display_name__": "监控新品汇聚"}
        )
        graph.add_node(
            "merge_all_hot", self._merge_node, metadata={"__display_name__": "全网热销汇聚"}
        )
        graph.add_node(
            "merge_all_new", self._merge_node, metadata={"__display_name__": "全网新品汇聚"}
        )

        # ===== 后处理节点 =====
        graph.add_node(
            "post_process", self._post_process_node, metadata={"__display_name__": "后处理"}
        )
        graph.add_node(
            "has_result", self._has_result_node, metadata={"__display_name__": "结果输出"}
        )
        graph.add_node(
            "package", self._package_result_node, metadata={"__display_name__": "封装结果"}
        )

        # ===== 入口和预处理边 =====
        graph.set_entry_point("init_state")
        graph.add_edge("init_state", "pre_think")
        graph.add_edge("pre_think", "query_selections")
        graph.add_edge("query_selections", "llm_parse")
        graph.add_edge("llm_parse", "parallel_parse")
        graph.add_edge("parallel_parse", "llm_merge")

        # ===== API 路由 (5 分支) =====
        graph.add_conditional_edges(
            "llm_merge",
            self._route_api_call,
            {
                "shop": "api_shop",
                "monitor_hot": "api_monitor_hot",
                "monitor_new": "api_monitor_new",
                "all_hot": "api_all_hot",
                "all_new": "api_all_new",
            },
        )

        # ===== 店铺分支 =====
        graph.add_conditional_edges(
            "api_shop",
            self._check_result_count,
            {"has_result": "merge_shop", "no_result": "fallback_shop"},
        )
        graph.add_edge("fallback_shop", "merge_shop")
        graph.add_edge("merge_shop", "post_process")

        # ===== 监控热销分支 =====
        graph.add_conditional_edges(
            "api_monitor_hot",
            self._check_result_count,
            {"has_result": "merge_monitor_hot", "no_result": "fallback_monitor_hot"},
        )
        graph.add_edge("fallback_monitor_hot", "merge_monitor_hot")
        graph.add_edge("merge_monitor_hot", "post_process")

        # ===== 监控新品分支 =====
        graph.add_conditional_edges(
            "api_monitor_new",
            self._check_result_count,
            {"has_result": "merge_monitor_new", "no_result": "fallback_monitor_new"},
        )
        graph.add_edge("fallback_monitor_new", "merge_monitor_new")
        graph.add_edge("merge_monitor_new", "post_process")

        # ===== 全网热销分支 =====
        graph.add_conditional_edges(
            "api_all_hot",
            self._check_result_count,
            {"has_result": "merge_all_hot", "no_result": "fallback_all_hot"},
        )
        graph.add_edge("fallback_all_hot", "merge_all_hot")
        graph.add_edge("merge_all_hot", "post_process")

        # ===== 全网新品分支 =====
        graph.add_conditional_edges(
            "api_all_new",
            self._check_result_count,
            {"has_result": "merge_all_new", "no_result": "fallback_all_new"},
        )
        graph.add_edge("fallback_all_new", "merge_all_new")
        graph.add_edge("merge_all_new", "post_process")

        # ===== 后处理和输出 =====
        graph.add_edge("post_process", "has_result")
        graph.add_edge("has_result", "package")
        graph.add_edge("package", END)

        return graph.compile()

    # ==================== 路由函数 ====================

    def _route_api_call(self, state: ZhiyiWorkflowState) -> str:
        """API 路由：根据品牌/监控/全网 + 热销/新品分支"""
        if state.get("is_brand_item") == 1:
            return "shop"

        # flag + sale_type 组合路由（监控/全网 × 热销/新品）
        flag = state.get("flag", 2)
        sale_type = state.get("sale_type", "热销")

        routing_table = {
            (1, "热销"): "monitor_hot",  # 监控热销
            (1, "新品"): "monitor_new",  # 监控新品
            (2, "热销"): "all_hot",  # 全网热销
            (2, "新品"): "all_new",  # 全网新品
        }

        return routing_table.get((flag, sale_type), "all_new")

    def _check_result_count(self, state: ZhiyiWorkflowState) -> str:
        """检查结果数量：< 10 或失败则走兜底"""
        result_count = state.get("result_count") or 0
        api_success = state.get("api_success", True)
        if not api_success or (result_count is not None and result_count < 10):
            return "no_result"
        return "has_result"

    # ==================== 工具方法 ====================

    @staticmethod
    def _normalize_property_options_to_candidates(
        options: list[dict[str, Any]],
        *,
        max_tags: int = 2000,
        max_values_per_property: int = 50,
    ) -> tuple[list[str], dict[str, set[str] | None]]:
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
            for key in (
                "values",
                "valueList",
                "propertyValue",
                "propertyValueList",
                "property_value_list",
                "enum",
                "options",
                "items",
                "children",
            ):
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

    def _get_pusher(self, req: WorkflowRequest) -> PhaseProgressPusher:
        return PhaseProgressPusher(template=self.PROGRESS_TEMPLATE, request=req)

    def _get_datasource_key(self, state: ZhiyiWorkflowState) -> dict[str, Any]:
        """根据查询模式获取数据源 Key"""
        api_branch = state.get("api_branch", "all_hot")

        if api_branch == "shop":
            return {"content": ZhiyiDataSourceKey.SHOP.value}
        if api_branch == "monitor_hot":
            return {"content": ZhiyiDataSourceKey.MONITOR_HOT.value}
        elif api_branch == "monitor_new":
            return {"content": ZhiyiDataSourceKey.MONITOR_NEW.value}
        elif api_branch == "all_hot":
            return {"content": ZhiyiDataSourceKey.HOT.value}
        else:  # all_new
            return {"content": ZhiyiDataSourceKey.ALL.value}

    @staticmethod
    def _resolve_request_path(api_branch: str | None) -> str:
        if api_branch == "shop":
            return "/v2-0-x/item/shop/all-item-list"
        if api_branch in ("monitor_new", "all_new"):
            return "/v2-0-x/item/simple-item-list"
        return "/v2-0-x/item/list"

    # ==================== 预处理节点 ====================

    def _init_state_node(self, state: ZhiyiWorkflowState) -> dict[str, Any]:
        """初始化状态节点"""
        req = state["request"]

        def insert_track():
            with pg_session(DBAlias.DB_ABROAD_AI.value) as session:
                params = {
                    "query": req.user_query,
                    "session_id": req.session_id,
                    "message_id": req.message_id,
                    "user_id": req.user_id,
                    "team_id": req.team_id,
                }
                result = session.execute(
                    text(
                        """
                        UPDATE holo_zhiyi_aiagent_query_prod
                        SET query = :query,
                            session_id = :session_id,
                            user_id = :user_id,
                            team_id = :team_id
                        WHERE message_id = :message_id
                    """
                    ),
                    params,
                )
                if result.rowcount == 0:
                    session.execute(
                        text(
                            """
                            INSERT INTO holo_zhiyi_aiagent_query_prod(
                                query, session_id, message_id, user_id, team_id
                            )
                            VALUES (
                                :query, :session_id, :message_id, :user_id, :team_id
                            )
                        """
                        ),
                        params,
                    )

        thread_pool.submit_with_context(insert_track)
        return {}

    def _pre_think_node(self, state: ZhiyiWorkflowState) -> dict[str, Any]:
        """预处理节点：发送开始消息和思维链"""
        req = state["request"]
        pusher = self._get_pusher(req)

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
                data=ParameterData(entity_type=WorkflowEntityType.TAOBAO_ITEM.code),
            ),
            create_ts=int(round(time.time() * 1000)),
        )
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            start_message.model_dump_json(),
        )

        def _generate_thinking_and_report_task() -> None:
            ref_helper = QueryReferenceHelper.from_request(req)
            format_query = ref_helper.replace_placeholders(req.user_query)
            invoke_params = {
                "user_query": format_query,
                "preferred_entity": req.preferred_entity,
                "industry": req.industry.split("#")[0] if req.industry else "",
                "user_preferences": req.user_preferences,
                "now_time": datetime.now().strftime("%Y-%m-%d"),
            }

            try:
                messages: list[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
                    prompt_key=CozePromptHubKey.ZHIYI_THINK_PROMPT.value,
                    variables=invoke_params,
                )
                llm: BaseChatModel = llm_factory.get_llm(
                    LlmProvider.DASHSCOPE.name, LlmModelName.DASHSCOPE_QWEN3_MAX.value
                )
                retry_llm = llm.with_retry(stop_after_attempt=2)

                thinking_chain = retry_llm | StrOutputParser()
                thinking_text = thinking_chain.with_config(run_name="思维链生成").invoke(messages)
                thinking_text = thinking_text.replace("\n\n", "\n").strip()

                if pusher and thinking_text:
                    pusher.complete_phase("选品任务规划", content=thinking_text)
            except Exception as e:
                logger.warning(f"[知衣] 思维链生成失败: {e}")

        # 同步调用确保消息推送顺序一致。后续可考虑使用更快的模型
        _generate_thinking_and_report_task()
        return {}

    def _query_selection_node(self, state: ZhiyiWorkflowState) -> dict[str, Any]:
        """查询品类维表"""
        req = state["request"]
        user_query = req.user_query
        industry = req.industry


        def query_activity_data() -> list[dict[str, Any]]:
            """查询活动维表数据（对齐 n8n: tb_activity_map）"""
            activity_sql = """
                SELECT activity_name, start_time, end_time
                FROM tb_activity_map
                WHERE status = 1
            """
            try:
                with mysql_session_readonly(DBAlias.B) as session:
                    db_result = session.execute(text(activity_sql))
                    return [dict(row) for row in db_result.mappings().all()]
            except (SQLAlchemyError, KeyError) as e:
                logger.warning(f"[知衣] 活动维表 tb_activity_map 不可用，跳过活动解析: {e}")
            except Exception as e:
                logger.warning(f"[知衣] 活动维表查询失败: {e}")
            return []

        # 并行执行: 活动维表 + 知识库检索 + 类目向量检索（保留 CozeTrace 上下文）
        activity_future = thread_pool.submit_with_context(query_activity_data)
        # 拼接上行业强化向量召回准确度
        kb_query_text = f"{user_query}, {industry}"
        kb_future = thread_pool.submit_with_context(self._fetch_knowledge_base, kb_query_text)
        category_vector_future = thread_pool.submit_with_context(
            self._fetch_category_vector,
            user_query,
            VolcKnowledgeServiceId.ZHIYI_CATEGORY_VECTOR,
            CozePromptHubKey.ZHIYI_CATEGORY_VECTOR_CLEAN_PROMPT.value,
        )

        category_vector_content = category_vector_future.result()
        category_data = (
            [{"content_list": category_vector_content}] if category_vector_content else []
        )
        selection_dict = {
            "category_data": category_data,
            "activity_data": activity_future.result(),
            "style_data": [],
            "kb_content": kb_future.result(),
            "category_vector_content": category_vector_content,
        }

        # 不在此处推送进度，等待 parallel_parse 完成后推送 PLANNING_FINISH
        return {"selection_dict": selection_dict}

    def _fetch_knowledge_base(self, query: str) -> str:
        """检索知识库内容"""
        service_id = VolcKnowledgeServiceId.ZHIYI_KNOWLEDGE.value
        if not service_id:
            return ""  # 服务ID未配置，跳过知识库检索

        try:
            from app.service.rpc.volcengine_kb_api import KBMessage, get_volcengine_kb_api

            kb_client = get_volcengine_kb_api()
            messages = [KBMessage(role="user", content=query)]
            response = kb_client.chat(messages=messages, service_resource_id=service_id)

            if not response.data or not response.data.result_list:
                return ""

            # 提取检索结果内容列表
            content_list = []
            for item in response.data.result_list:
                if hasattr(item, "content") and item.content:
                    content_list.append(item.content)
                elif hasattr(item, "table_chunk_fields") and len(item.table_chunk_fields) > 1:
                    content_list.append(item.table_chunk_fields[1].get("field_value", ""))

            if not content_list:
                return ""

            # 使用 LLM 过滤无关信息
            kb_content = content_list[0] if content_list else ""
            filtered_content = self._filter_kb_content_with_llm(kb_content, query)

            return filtered_content
        except Exception as e:
            logger.warning(f"[知识库检索] 失败: {e}")
            return ""

    def _filter_kb_content_with_llm(self, kb_content: str, user_query: str) -> str:
        """使用 LLM 过滤无关的知识库内容"""
        if not kb_content:
            return ""

        try:
            llm: BaseChatModel = llm_factory.get_llm(
                LlmProvider.OPENROUTER.name, LlmModelName.OPENROUTER_GEMINI_2_5_FLASH.value
            )
            result = llm.with_config(run_name="补充知识库清洗").invoke(
                coze_loop_client_provider.get_langchain_messages(
                    prompt_key=CozePromptHubKey.ZHIYI_KB_FILTER_PROMPT.value,
                    variables={"kb_content": kb_content, "user_query": user_query},
                ),
            )

            content = result.content if hasattr(result, "content") else str(result)
            # 如果返回 null 或空，说明内容不相关
            if not content or content.strip().lower() == "null":
                return ""
            return content.strip()
        except Exception as e:
            logger.warning(f"[知识库过滤] LLM 调用失败: {e}，返回原始内容")
            return kb_content  # 降级：返回原始内容


    def _extract_user_style_tags(self, style_payload: Any) -> str:
        """从用户画像风格中提取 3 个关键词（对齐 n8n 封装用户画像标签）"""
        if not style_payload:
            return ""

        style_json = json.dumps(style_payload, ensure_ascii=False)
        try:
            llm: BaseChatModel = llm_factory.get_llm(
                LlmProvider.OPENROUTER.name, LlmModelName.OPENROUTER_GEMINI_2_5_FLASH.value
            )
            structured_llm = llm.with_structured_output(ZhiyiUserTagResult).with_retry(
                stop_after_attempt=2
            )
            result = structured_llm.with_config(run_name="用户画像解析").invoke(
                coze_loop_client_provider.get_langchain_messages(
                    prompt_key=CozePromptHubKey.ZHIYI_USER_TAG_PARSE_PROMPT.value,
                    variables={"user_select_message": style_json},
                )
            )
            return (result.values or "").strip()
        except Exception as e:
            logger.warning(f"[知衣] 用户画像标签解析失败: {e}")

        if isinstance(style_payload, list):
            return ",".join([str(v) for v in style_payload[:3] if str(v).strip()])
        return str(style_payload)

    def _build_llm_parameters_payload(
        self,
        *,
        req: WorkflowRequest,
        param_result: ZhiyiParseParam,
        property_list: list[dict[str, Any]],
        sort_result: ZhiyiSortResult | None,
        user_style_values: str,
        user_filter_tags: Any,
        brand_list: list[str],
        is_brand_item: int,
        shop_id: str | None,
    ) -> list[ZhiyiLlmParametersItem]:
        """构建对齐 n8n 的 llm_parameters 数据结构"""
        # Ensure user filters are JSON-serializable for DB tracking.
        user_filter_payload: list[dict[str, Any]] = []
        for item in user_filter_tags or []:
            if isinstance(item, UserTagFilterItem):
                payload = item.model_dump(exclude_none=True)
            elif isinstance(item, dict):
                payload = {k: v for k, v in item.items() if v is not None}
            else:
                value = str(item).strip() if item is not None else ""
                payload = {"value": value} if value else {}
            if payload:
                user_filter_payload.append(payload)

        param_output = param_result.model_dump(by_alias=True, exclude_none=False)
        param_output["userid"] = req.user_id
        param_output["teamid"] = req.team_id

        sort_output = {
            "sortField_new": sort_result.sort_type_final if sort_result else "",
            "sortField_new_name": sort_result.sort_type_final_name if sort_result else "",
        }

        brand_output: dict[str, Any] = {"brand_list": brand_list, "is_brand_item": is_brand_item}
        if shop_id is not None:
            try:
                brand_output["shopid"] = int(shop_id)
            except Exception:
                brand_output["shopid"] = shop_id

        return [
            ZhiyiLlmParametersItem(property_list=property_list or []),
            ZhiyiLlmParametersItem(output=param_output),
            ZhiyiLlmParametersItem(output=sort_output),
            ZhiyiLlmParametersItem(output={"values": user_style_values or ""}),
            ZhiyiLlmParametersItem(output={"user_tags": user_filter_payload}),
            ZhiyiLlmParametersItem(output=brand_output),
        ]

    def _insert_llm_parameters(
        self, req: WorkflowRequest, payload: list[ZhiyiLlmParametersItem]
    ) -> None:
        """插入 llm_parameters 埋点（对齐 n8n 埋点1）"""

        def _insert():
            adapter = TypeAdapter(list[ZhiyiLlmParametersItem])
            llm_parameters = adapter.dump_json(
                payload, by_alias=True, exclude_none=True
            ).decode("utf-8")
            with pg_session(DBAlias.DB_ABROAD_AI.value) as session:
                params = {
                    "query": req.user_query,
                    "session_id": req.session_id,
                    "message_id": req.message_id,
                    "user_id": req.user_id,
                    "team_id": req.team_id,
                    "llm_parameters": llm_parameters,
                }
                result = session.execute(
                    text(
                        """
                        UPDATE holo_zhiyi_aiagent_query_prod
                        SET query = :query,
                            session_id = :session_id,
                            user_id = :user_id,
                            team_id = :team_id,
                            llm_parameters = :llm_parameters
                        WHERE message_id = :message_id
                    """
                    ),
                    params,
                )
                if result.rowcount == 0:
                    session.execute(
                        text(
                            """
                            INSERT INTO holo_zhiyi_aiagent_query_prod(
                                query, session_id, message_id, user_id, team_id, llm_parameters
                            )
                            VALUES (
                                :query, :session_id, :message_id, :user_id, :team_id, :llm_parameters
                            )
                        """
                        ),
                        params,
                    )

        thread_pool.submit_with_context(_insert)

    # ==================== 解析节点 ====================

    def _llm_param_parse_node(self, state: ZhiyiWorkflowState) -> dict[str, Any]:
        """LLM 参数解析节点（拆分多维度子解析）"""
        req = state["request"]
        selection_dict = state["selection_dict"]

        # 替换用户问题中的占位符
        ref_helper = QueryReferenceHelper.from_request(req)
        format_query = ref_helper.replace_placeholders(req.user_query)

        # 构建引用信息，供 LLM 识别和替换引用名称
        # 过滤无效引用（reference_name 或 display_name 为空）并处理特殊字符
        query_references_text = "\n".join(
            f"{ref.reference_name} -> {ref.display_name.replace(chr(10), ' ').replace(chr(13), ' ')}"
            for ref in (req.query_references or [])
            if ref.reference_name and ref.display_name
        )

        current_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        common_param = {
            "user_query": format_query,
            "industry": req.industry.split("#")[0] if req.industry else "",
            "user_preferences": req.user_preferences,
            "current_date": current_date,
            "query_references": query_references_text,  # 引用名称映射，格式：reference_name -> display_name
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
                    LlmProvider.DASHSCOPE.name,
                    LlmModelName.DASHSCOPE_QWEN3_MAX.value,
                )
                structured_llm = llm.with_structured_output(schema).with_retry(
                    stop_after_attempt=2
                )
                return structured_llm.with_config(run_name=run_name).invoke(messages)
            except Exception as exc:
                logger.warning(f"[知衣] {run_name} 解析失败: {exc}")
                return schema()

        category_future = thread_pool.submit_with_context(
            lambda: run_parse(
                CozePromptHubKey.ZHIYI_CATEGORY_PARSE_PROMPT.value,
                ZhiyiCategoryParseResult,
                "类目解析",
                {
                    **common_param,
                    "category_data": json.dumps(
                        selection_dict.get("category_data", []), ensure_ascii=False
                    ),
                    "kb_content": selection_dict.get("kb_content", ""),
                },
            )
        )
        time_future = thread_pool.submit_with_context(
            lambda: run_parse(
                CozePromptHubKey.ZHIYI_TIME_PARSE_PROMPT.value,
                ZhiyiTimeParseResult,
                "时间解析",
                {
                    **common_param,
                    "activity_data": json.dumps(
                        selection_dict.get("activity_data", []), ensure_ascii=False
                    ),
                },
            )
        )
        numeric_future = thread_pool.submit_with_context(
            lambda: run_parse(
                CozePromptHubKey.ZHIYI_NUMERIC_PARSE_PROMPT.value,
                ZhiyiNumericParseResult,
                "数值解析",
                common_param,
            )
        )
        route_future = thread_pool.submit_with_context(
            lambda: run_parse(
                CozePromptHubKey.ZHIYI_ROUTE_PARSE_PROMPT.value,
                ZhiyiRouteParseResult,
                "路由意图解析",
                common_param,
            )
        )
        brand_future = thread_pool.submit_with_context(
            lambda: run_parse(
                CozePromptHubKey.ZHIYI_BRAND_PHRASE_PARSE_PROMPT.value,
                ZhiyiBrandPhraseParseResult,
                "品牌短语解析",
                common_param,
            )
        )

        category_result = category_future.result()

        time_result = time_future.result()
        numeric_result = numeric_future.result()
        route_result = route_future.result()
        brand_result = brand_future.result()

        param_result = self._merge_param_result(
            category_result=category_result,
            time_result=time_result,
            numeric_result=numeric_result,
            route_result=route_result,
            brand_result=brand_result,
        )

        # 不在此处推送进度，等待 parallel_parse 完成后推送 PLANNING_FINISH
        return {
            "param_result": param_result,
            "category_parse_result": category_result,
            "time_parse_result": time_result,
            "numeric_parse_result": numeric_result,
            "route_parse_result": route_result,
            "brand_parse_result": brand_result,
        }

    @staticmethod
    def _merge_param_result(
        *,
        category_result: ZhiyiCategoryParseResult,
        time_result: ZhiyiTimeParseResult,
        numeric_result: ZhiyiNumericParseResult,
        route_result: ZhiyiRouteParseResult,
        brand_result: ZhiyiBrandPhraseParseResult,
    ) -> ZhiyiParseParam:
        """合并子解析结果为主参数"""

        def normalize_text(value: Any) -> str | None:
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        limit = numeric_result.limit if numeric_result.limit is not None else 6000
        if limit > 6000:
            limit = 6000

        return ZhiyiParseParam(
            low_volume=numeric_result.low_volume if numeric_result.low_volume is not None else 0,
            high_volume=(
                numeric_result.high_volume if numeric_result.high_volume is not None else 99999999
            ),
            low_price=numeric_result.low_price if numeric_result.low_price is not None else 0,
            high_price=(
                numeric_result.high_price if numeric_result.high_price is not None else 999999
            ),
            start_date=time_result.start_date,
            end_date=time_result.end_date,
            sale_start_date=time_result.sale_start_date,
            sale_end_date=time_result.sale_end_date,
            category_id=category_result.category_id or [],
            root_category_id=category_result.root_category_id,
            category_name=category_result.category_name or [],
            root_category_name=category_result.root_category_name,
            properties=None,
            query_title=None,
            brand=normalize_text(brand_result.brand),
            type=route_result.type or "热销",
            shop_type=route_result.shop_type or None,
            flag=route_result.flag if route_result.flag is not None else 2,
            user_data=route_result.user_data if route_result.user_data is not None else 0,
            shop_switch=route_result.shop_switch or [],
            sort_field=route_result.sort_field or "默认",
            limit=limit,
            title=normalize_text(category_result.title),
        )

    def _llm_param_merge_node(self, state: ZhiyiWorkflowState) -> dict[str, Any]:
        """LLM 合并节点：仅允许修改少量字段以提高一致性"""
        req = state["request"]
        param_result: ZhiyiParseParam = state["param_result"]

        ref_helper = QueryReferenceHelper.from_request(req)
        format_query = ref_helper.replace_placeholders(req.user_query)

        def dump_prompt_value(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, BaseModel):
                payload = value.model_dump()
            else:
                payload = value
            try:
                return json.dumps(payload, ensure_ascii=False)
            except TypeError:
                return str(payload)

        # 从 state 提取属性清洗结果（parallel_parse 产出）
        property_list: list[dict] = state.get("property_list") or []
        matched_properties = ",".join(
            item.get("name", "") for item in property_list if item.get("name")
        )
        unmatched_keywords = param_result.query_title or ""

        prompt_param = {
            "user_query": format_query,
            "current_query_title": unmatched_keywords,
            "category_name": json.dumps(param_result.category_name or [], ensure_ascii=False),
            "root_category_name": param_result.root_category_name or "",
            "matched_properties": matched_properties,
            "unmatched_keywords": unmatched_keywords,
            "brand": param_result.brand or "",
            "user_preferences": req.user_preferences or "",
            "category_parse_result": dump_prompt_value(state.get("category_parse_result")),
            "time_parse_result": dump_prompt_value(state.get("time_parse_result")),
            "numeric_parse_result": dump_prompt_value(state.get("numeric_parse_result")),
            "route_parse_result": dump_prompt_value(state.get("route_parse_result")),
            "brand_parse_result": dump_prompt_value(state.get("brand_parse_result")),
        }

        try:
            messages: list[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
                prompt_key=CozePromptHubKey.ZHIYI_PARAM_MERGE_PROMPT.value,
                variables=prompt_param,
            )
            llm: BaseChatModel = llm_factory.get_llm(
                LlmProvider.DASHSCOPE.name,
                LlmModelName.DASHSCOPE_QWEN3_MAX.value,
            )
            structured_llm = llm.with_structured_output(ZhiyiParamMergeResult).with_retry(
                stop_after_attempt=2
            )
            merge_result = structured_llm.with_config(run_name="参数合并").invoke(messages)
        except Exception as exc:
            logger.warning(f"[知衣] 参数合并失败: {exc}")
            return {}

        updated_param = self._apply_param_merge_result(
            param_result=param_result,
            merge_result=merge_result,
        )

        return {"param_result": updated_param}

    @staticmethod
    def _apply_param_merge_result(
        *,
        param_result: ZhiyiParseParam,
        merge_result: ZhiyiParamMergeResult,
    ) -> ZhiyiParseParam:
        def normalize_text(value: Any) -> str | None:
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        updated = param_result.model_copy(deep=True)

        raw_merge_title = normalize_text(merge_result.query_title)
        if raw_merge_title is not None or param_result.query_title:
            base_title = raw_merge_title or param_result.query_title or ""
            cleaned_title = ZhiyiGraph._clean_query_title(base_title)
            if cleaned_title:
                updated.query_title = cleaned_title
            else:
                fallback_title = ZhiyiGraph._fallback_query_title(param_result)
                if fallback_title:
                    updated.query_title = fallback_title
                else:
                    updated.query_title = ""

        return updated

    @staticmethod
    def _clean_query_title(text: str) -> str:
        if not text:
            return ""
        tokens = re.split(r"[，,、;；\\s]+", str(text))
        keep: list[str] = []
        generic_phrases = (
            "热销",
            "热门",
            "爆款",
            "畅销",
            "卖得好",
            "卖得比较好",
            "比较好",
            "好卖",
            "高销量",
            "销量高",
            "销量好",
        )

        for token in tokens:
            term = token.strip()
            if not term:
                continue
            if any(phrase in term for phrase in generic_phrases):
                continue
            if re.search(r"\\d{4}-\\d{2}-\\d{2}", term):
                continue
            if re.search(r"\\d+\\s*(天|日|周|月|年|季度)", term):
                continue
            if re.search(r"\\d{4}年\\d{1,2}月\\d{1,2}日", term):
                continue
            if re.search(r"(近|最近|过去|近一段时间|近段时间).*(天|周|月|年|季度)", term):
                continue
            keep.append(term)

        return ",".join(keep)

    @staticmethod
    def _fallback_query_title(param_result: ZhiyiParseParam) -> str:
        if param_result.properties:
            return param_result.properties
        if param_result.category_name:
            return ",".join([name for name in param_result.category_name if name])
        if param_result.root_category_name:
            return param_result.root_category_name
        return ""

    def _parallel_parse_node(self, state: ZhiyiWorkflowState) -> dict[str, Any]:
        """并行解析节点：属性、排序、品牌、用户画像"""
        req = state["request"]
        param_result: ZhiyiParseParam = state["param_result"]

        ref_helper = QueryReferenceHelper.from_request(req)
        format_query = ref_helper.replace_placeholders(req.user_query)

        def parse_sort():
            """解析排序项"""
            prompt_param = {
                "user_query": format_query,
                "sort_field": param_result.sort_field or "默认",
                "flag": str(param_result.flag),
                "type": param_result.type,
            }
            messages = coze_loop_client_provider.get_langchain_messages(
                prompt_key=CozePromptHubKey.ZHIYI_SORT_TYPE_PARSE_PROMPT.value,
                variables=prompt_param,
            )
            llm = llm_factory.get_llm(
                LlmProvider.OPENROUTER.name, LlmModelName.OPENROUTER_GEMINI_2_5_FLASH.value
            )
            structured_llm = llm.with_structured_output(ZhiyiSortResult).with_retry(
                stop_after_attempt=2
            )
            return structured_llm.with_config(run_name="排序解析").invoke(messages)

        def split_brand_text(brand_text: str | None) -> list[str]:
            """品牌拆分（对齐 n8n，不走 LLM）"""
            if not brand_text:
                return []
            text = brand_text if isinstance(brand_text, str) else str(brand_text)
            parts = re.split(r"[，,、;；|#]+", text)
            brand_list: list[str] = []
            seen: set[str] = set()
            for part in parts:
                item = part.strip()
                if not item or item in seen:
                    continue
                seen.add(item)
                brand_list.append(item)
            return brand_list

        def merge_unique(items: list[str]) -> list[str]:
            merged: list[str] = []
            seen: set[str] = set()
            for item in items:
                if item in seen:
                    continue
                seen.add(item)
                merged.append(item)
            return merged

        def extract_brand_refs() -> list[str]:
            brand_refs = ref_helper.get_by_type(QueryReferenceEntityType.TAOBAO_BRAND.value)
            names: list[str] = []
            for ref in brand_refs:
                raw = ref.get("display_name") or ref.get("reference_name")
                if not raw:
                    continue
                names.extend(split_brand_text(raw))
            return names

        def split_shop_candidates(raw: str | None) -> list[str]:
            if not raw:
                return []
            text = str(raw)
            for sep in ("，", ",", "、", ";", "；", "|", "#"):
                text = text.replace(sep, ",")
            return [part.strip() for part in text.split(",") if part.strip()]

        def extract_shop_name_from_llm() -> str | None:
            """优先使用 LLM 解析的店铺/品牌字段作为店铺名候选。"""
            shop_hints = ("官方旗舰店", "旗舰店", "官方店", "专卖店", "店铺")
            candidates = split_shop_candidates(param_result.brand)
            if not candidates:
                return None
            for cand in candidates:
                if any(hint in cand for hint in shop_hints):
                    return cand
            return candidates[0]

        def recall_shop_id_by_name(shop_name: str) -> str | None:
            """根据店铺名召回店铺 ID"""
            try:
                kb_client = get_volcengine_kb_api()
                response = kb_client.simple_chat(
                    query=shop_name,
                    service_resource_id=VolcKnowledgeServiceId.ZHIYI_SHOP_KNOWLEDGE.value,
                )
                if not response.data or not response.data.result_list:
                    return None
                content_list: list[str] = []
                for item in response.data.result_list:
                    key = ""
                    value = ""
                    for chunk in item.table_chunk_fields:
                        field_name = chunk.get("field_name", "")
                        field_value = chunk.get("field_value", "")
                        if field_name == "key":
                            key = field_value
                        elif field_name == "value":
                            value = field_value
                    if key or value:
                        content_list.append(f"{key},{value}")
                if not content_list:
                    return None

                structured_llm = (
                    llm_factory.get_llm(
                        LlmProvider.OPENROUTER.name,
                        LlmModelName.OPENROUTER_GEMINI_2_5_FLASH.value,
                    )
                    .with_structured_output(schema=ZhiyiShopLlmCleanResult)
                    .with_retry(stop_after_attempt=2)
                )
                messages = coze_loop_client_provider.get_langchain_messages(
                    prompt_key=CozePromptHubKey.ZHIYI_SHOP_CLEAN_PROMPT.value,
                    variables={
                        "query_tag_list": json.dumps(content_list, ensure_ascii=False),
                        "origin_text": shop_name,
                    },
                )
                result: ZhiyiShopLlmCleanResult = structured_llm.with_config(
                    run_name="召回店铺清洗"
                ).invoke(messages)
                clean_tags: list[str] = result.clean_tag_list
                if not clean_tags:
                    return None
                parts = clean_tags[0].split(",", 1)
                if len(parts) < 2:
                    return None
                return parts[1]
            except Exception as exc:
                logger.warning(f"[知衣] 店铺名称召回失败: {exc}")
                return None

        def query_shop_id() -> str | None:
            """根据店铺引用或店铺名查询店铺 ID"""
            try:
                shop_ref = ref_helper.get_first_entity_by_type(
                    QueryReferenceEntityType.TAOBAO_SHOP.value
                )
                if not shop_ref:
                    shop_name = extract_shop_name_from_llm()
                    if not shop_name:
                        return None
                    return recall_shop_id_by_name(shop_name)
                shop_id = shop_ref.get("entity_id")
                if shop_id:
                    return shop_id
                shop_name = shop_ref.get("display_name") or shop_ref.get("reference_name")
                if not shop_name:
                    return None
                return recall_shop_id_by_name(shop_name)
            except Exception as e:
                logger.warning(f"[知衣] 店铺引用解析失败: {e}")
                return None

        def get_user_style_values() -> str:
            """获取用户画像风格标签（对齐 n8n 用户画像逻辑）"""
            if param_result.user_data != 1:
                return ""

            root_category_id = param_result.root_category_id
            category_ids = param_result.category_id or []
            if not root_category_id or not category_ids:
                return ""

            try:
                with pg_session(DBAlias.DB_ABROAD_AI.value) as session:
                    # 使用参数化查询避免 SQL 注入
                    style_result = session.execute(
                        text(
                            """
                            SELECT style FROM public.ads_zhiyi_user_profile_recommend_goods_selection
                            WHERE user_id = :user_id AND root_category_id = :root_category_id
                              AND category_id = ANY(:category_ids)
                            LIMIT 1
                        """
                        ),
                        {
                            "user_id": req.user_id,
                            "root_category_id": root_category_id,
                            "category_ids": category_ids,
                        },
                    )
                    style_row = style_result.fetchone()
                    style_payload = style_row[0] if style_row and style_row[0] else None
                    if not style_payload:
                        return ""

                    return self._extract_user_style_tags(style_payload)
            except Exception as e:
                logger.warning(f"[知衣] 用户画像风格查询失败: {e}")
                return ""

        def get_user_filter_tags() -> list[UserTagFilterItem]:
            """获取用户画像 filters（仅用于输出）"""
            root_category_id = param_result.root_category_id
            category_ids = param_result.category_id or []
            if not root_category_id or not category_ids:
                return []

            try:
                with pg_session(DBAlias.DB_ABROAD_AI.value) as session:
                    # 使用参数化查询避免 SQL 注入
                    filters_result = session.execute(
                        text(
                            """
                            SELECT filters FROM public.ads_zhiyi_user_profile_recommend_goods_selection
                            WHERE user_id = :user_id AND root_category_id = :root_category_id
                              AND category_id = ANY(:category_ids)
                              AND filters IS NOT NULL AND filters <> '' AND filters NOT LIKE '%null%'
                            LIMIT 1
                        """
                        ),
                        {
                            "user_id": req.user_id,
                            "root_category_id": root_category_id,
                            "category_ids": category_ids,
                        },
                    )
                    filters_row = filters_result.fetchone()
                    if filters_row and filters_row[0]:
                        raw = filters_row[0]
                        if not raw:
                            return []
                        raw = json.loads(raw)
                        # 转换为 UserTagFilterItem 列表
                        return [UserTagFilterItem(**item) for item in raw]
            except Exception as e:
                logger.warning(f"[知衣] 用户画像 filters 查询失败: {e}")
            return []

        def parse_properties() -> tuple[list[dict[str, str]], str]:
            """解析属性列表（对齐抖衣逻辑：类目属性候选 + LLM 清洗），返回 (property_list, queryTitle)"""
            try:
                category_id_list = param_result.category_id or []
                if not category_id_list:
                    return [], ""

                content_list: list[str] = []
                allowed: dict[str, set[str] | None] = {}
                try:
                    api_client = get_zhiyi_api_client()
                    options = api_client.get_category_property_options(
                        user_id=req.user_id,
                        team_id=req.team_id,
                        category_id_list=category_id_list,
                    )
                    content_list, allowed = self._normalize_property_options_to_candidates(options)
                except Exception as e:
                    logger.warning(f"[知衣工作流属性解析] 类目属性候选接口不可用: {e}")

                if not content_list:
                    return [], ""

                # 2. llm清洗标签
                structured_llm = (
                    llm_factory.get_llm(
                        LlmProvider.DASHSCOPE.name, LlmModelName.DASHSCOPE_QWEN3_MAX.value
                    )
                    # DashScope OpenAI-compatible mode: enable thinking via extra_body.
                    .bind(extra_body={"enable_thinking": True})
                    .with_structured_output(schema=ZhiyiPropertyLlmCleanResult)
                    .with_retry(stop_after_attempt=2)
                )
                messages = coze_loop_client_provider.get_langchain_messages(
                    prompt_key=CozePromptHubKey.ZHIYI_PROPERTY_CLEAN_PROMPT.value,
                    variables={
                        "query_tag_list": json.dumps(content_list, ensure_ascii=False),
                        "origin_text": format_query,
                    },
                )
                result: ZhiyiPropertyLlmCleanResult = structured_llm.with_config(run_name="召回属性清洗").invoke(messages)

                # 提取 unmatched_keywords 作为 queryTitle
                query_title_from_props = ",".join(result.unmatched_keywords) if result.unmatched_keywords else ""

                filtered_tags = self._filter_property_clean_tags(result.clean_tag_list, allowed)
                if not filtered_tags:
                    return [], query_title_from_props

                # 3. 组织为知衣api的name value格式 example: [{"name": "key", "values":"value1,value2"}]
                property_dict: dict[str, list[str]] = {}
                for tag in filtered_tags:
                    key, value = tag.split(":", 1)
                    property_dict.setdefault(key, []).append(value)
                group_property_list = [{"name": name, "values": ",".join(values)} for name, values in property_dict.items()]
                return group_property_list, query_title_from_props
            except Exception as e:
                logger.warning(f"[知衣工作流属性解析]发生异常：{e}")
                return [], ""

        brand_list = merge_unique(split_brand_text(param_result.brand) + extract_brand_refs())
        shop_id_future = thread_pool.submit_with_context(query_shop_id)

        # 并行执行排序与用户画像解析任务（保留 CozeTrace 上下文）
        sort_future = thread_pool.submit_with_context(parse_sort)
        user_style_values_future = thread_pool.submit_with_context(get_user_style_values)
        user_filter_tags_future = thread_pool.submit_with_context(get_user_filter_tags)

        sort_result = sort_future.result()
        user_style_values = user_style_values_future.result()
        user_filter_tags = user_filter_tags_future.result()

        shop_id = shop_id_future.result()
        property_future = thread_pool.submit_with_context(parse_properties)
        property_list, query_title_from_props = property_future.result()
        # 将属性清洗产出的 queryTitle（unmatched_keywords）回写到 param_result
        param_result.query_title = query_title_from_props or None
        is_brand_item = 1 if shop_id else 0
        flag = param_result.flag  # 1=监控店铺, 2=全网
        sale_type = param_result.type  # 热销 / 新品

        llm_parameters_payload = self._build_llm_parameters_payload(
            req=req,
            param_result=param_result,
            property_list=property_list,
            sort_result=sort_result,
            user_style_values=user_style_values,
            user_filter_tags=user_filter_tags,
            brand_list=brand_list,
            is_brand_item=is_brand_item,
            shop_id=shop_id,
        )
        self._insert_llm_parameters(req, llm_parameters_payload)

        return {
            "param_result": param_result,
            "sort_result": sort_result,
            "brand_list": brand_list,
            "shop_id": shop_id,
            "user_style_values": user_style_values,
            "user_filter_tags": user_filter_tags,
            "property_list": property_list,
            "is_brand_item": is_brand_item,
            "flag": flag,
            "sale_type": sale_type,
            "style_list": [],
        }

    # ==================== 主 API 节点（通用实现）====================

    def _build_standard_request(
        self,
        context: RequestContext,
        spec: BranchRequestSpec,
        shop_label_list: list[str],
        shop_type: str | None,
        query_title: str,
    ) -> dict[str, Any]:
        """构建标准请求体（对齐 n8n params/params1/params3）"""
        request = ZhiyiStandardRequest(
            start_date=context.start_date,
            end_date=context.end_date,
            sale_start_date=context.sale_start_date,
            sale_end_date=context.sale_end_date,
            category_id_list=context.category_id_list,
            group_id_list=spec.group_id_list,
            shop_label_list=shop_label_list,
            min_volume=context.min_volume,
            max_volume=context.max_volume,
            shop_type=shop_type,
            root_category_id_list=context.root_category_id_list,
            page_size=spec.page_size,
            page_no=spec.page_no,
            sort_field=context.sort_field,
            min_coupon_cprice=context.min_coupon_cprice,
            max_coupon_cprice=context.max_coupon_cprice,
            query_title=query_title,
            brand_list=context.brand_list or None,
            limit=context.limit,
            property_list=context.property_list if spec.include_property_list else None,
        )
        return request.model_dump(by_alias=True, exclude_none=True)

    def _build_extended_request(
        self,
        context: RequestContext,
        spec: BranchRequestSpec,
        shop_label_list: list[str],
        shop_type: str | None,
        query_title: str,
    ) -> dict[str, Any]:
        """构建扩展请求体（对齐 n8n params2/params 的 params2）"""
        request = ZhiyiExtendedRequest(
            root_category_id_list=context.root_category_id_list,
            category_id_list=context.category_id_list,
            sale_start_date=context.sale_start_date,
            sale_end_date=context.sale_end_date,
            page_no=spec.page_no,
            page_size=spec.page_size,
            limit=context.limit,
            min_coupon_cprice=context.min_coupon_cprice,
            max_coupon_cprice=context.max_coupon_cprice,
            shop_type=shop_type,
            min_volume=context.min_volume,
            max_volume=context.max_volume,
            shop_label_list=shop_label_list,
            query_title=query_title,
            group_id_list=spec.group_id_list,
            sort_field=context.sort_field,
            brand_list=context.brand_list or [],
            property_list=context.property_list if spec.include_property_list else None,
            start_date=context.start_date if spec.include_start_date else None,
            end_date=context.end_date if spec.include_start_date else None,
        )
        return request.model_dump(by_alias=True, exclude_none=True)

    def _build_shop_request(
        self,
        context: RequestContext,
        spec: BranchRequestSpec,
        shop_id: int,
        query_title: str | None,
    ) -> dict[str, Any]:
        """构建店铺请求体（对齐 n8n params4）"""
        # 店铺接口默认 pageSize=10，会导致“前100名”只返回10条。
        # 显式传入 pageSize/pageNo/limit，并用 limit 约束 pageSize。
        page_size = spec.page_size
        if context.limit:
            try:
                page_size = min(page_size, max(1, int(context.limit)))
            except (TypeError, ValueError):
                page_size = spec.page_size
        request = ZhiyiShopRequest(
            page_size=page_size,
            page_no=spec.page_no,
            shop_id=shop_id,
            sort_field=context.sort_field,
            category_id_list=context.category_id_list,
            start_date=context.start_date,
            end_date=context.end_date,
            sale_start_date=context.sale_start_date,
            sale_end_date=context.sale_end_date,
            indeterminate_root_category_id_list=context.root_category_id_list,
            min_volume=context.min_volume,
            max_volume=context.max_volume,
            min_coupon_cprice=context.min_coupon_cprice,
            max_coupon_cprice=context.max_coupon_cprice,
            limit=context.limit,
            property_list=context.property_list if spec.include_property_list else None,
            query_title=query_title,
        )
        return request.model_dump(by_alias=True, exclude_none=True)

    def _build_request_for_branch(
        self, state: ZhiyiWorkflowState, api_branch: ZhiyiApiBranch, is_fallback: bool
    ) -> dict[str, Any]:
        """根据分支构建请求体（严格对齐 n8n）"""
        context = self._build_request_context(state)
        spec = self._get_request_spec(api_branch, is_fallback)
        if not spec:
            return {}
        fallback_root_only = is_fallback and bool(context.root_category_id_list)
        if fallback_root_only:
            # 兜底时放宽类目：去掉叶子类目，仅保留根类目筛选。
            context.category_id_list = []

        if spec.kind == "shop":
            shop_id = state.get("shop_id")
            if not shop_id:
                return {}
            shop_query_title = self._resolve_shop_query_title(
                context, state, spec.query_title_mode
            )
            if spec.query_title_mode != "none" and not shop_query_title:
                shop_query_title = ""
            return self._build_shop_request(
                context=context,
                spec=spec,
                shop_id=int(shop_id),
                query_title=shop_query_title,
            )

        shop_label_list, shop_type = self._resolve_shop_filters(
            context, spec.use_param_shop_filter
        )
        query_title = self._resolve_query_title(context, spec.query_title_mode)
        query_title_value = query_title or ""

        if spec.kind == "standard":
            request = self._build_standard_request(
                context=context,
                spec=spec,
                shop_label_list=shop_label_list,
                shop_type=shop_type,
                query_title=query_title_value,
            )
            if spec.query_title_mode == "none":
                request.pop("queryTitle", None)
            if fallback_root_only:
                request.pop("categoryIdList", None)
            return request

        if spec.kind == "extended":
            request = self._build_extended_request(
                context=context,
                spec=spec,
                shop_label_list=shop_label_list,
                shop_type=shop_type,
                query_title=query_title_value,
            )
            if spec.query_title_mode == "none":
                request.pop("queryTitle", None)
            if fallback_root_only:
                request.pop("categoryIdList", None)
            return request

        return {}

    def _get_request_spec(self, api_branch: ZhiyiApiBranch, is_fallback: bool) -> BranchRequestSpec | None:
        return self._BRANCH_REQUEST_SPECS.get((api_branch, is_fallback))

    @staticmethod
    def _resolve_query_title(context: RequestContext, mode: str) -> str | None:
        if mode == "none":
            return None
        if mode == "fallback":
            value = context.query_title_fallback
        else:
            value = context.query_title_main
        value = (value or "").strip()
        return value or None

    @staticmethod
    def _split_query_terms(raw: str | None) -> list[str]:
        if not raw:
            return []
        parts = re.split(r"[，,、;；/|#\s]+", str(raw))
        terms: list[str] = []
        for part in parts:
            term = str(part).strip()
            if term:
                terms.append(term)
        return terms

    @staticmethod
    def _normalize_query_term(term: str) -> str:
        return re.sub(r"\s+", "", str(term)).strip().lower()

    @staticmethod
    def _split_property_terms(raw: str | None) -> list[str]:
        if not raw:
            return []
        parts = re.split(r"[，,、;；/|#]+", str(raw))
        terms: list[str] = []
        for part in parts:
            term = str(part).strip()
            if term:
                terms.append(term)
        return terms

    @classmethod
    def _extract_property_terms(cls, property_list: list[dict[str, Any]]) -> list[str]:
        terms: list[str] = []
        for item in property_list:
            if not isinstance(item, dict):
                continue
            raw = item.get("values") or item.get("value") or ""
            if isinstance(raw, list):
                parts = raw
            else:
                parts = re.split(r"[，,、;；/|#]+", str(raw))
            for part in parts:
                term = str(part).strip()
                if term:
                    terms.append(term)
        if not terms:
            return []
        seen: set[str] = set()
        unique_terms: list[str] = []
        for term in terms:
            if term in seen:
                continue
            seen.add(term)
            unique_terms.append(term)
        return unique_terms

    @classmethod
    def _dedupe_query_terms(cls, terms: list[str]) -> list[str]:
        seen: set[str] = set()
        unique_terms: list[str] = []
        for term in terms:
            norm = cls._normalize_query_term(term)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            unique_terms.append(term)
        return unique_terms

    @classmethod
    def _build_query_title_main(
        cls,
        *,
        param_result: ZhiyiParseParam,
        property_list: list[dict[str, Any]],
        brand_list: list[str],
        user_style_values: str,
    ) -> str:
        candidate_texts: list[str] = []
        if param_result.properties:
            candidate_texts.append(param_result.properties)
        if param_result.query_title:
            candidate_texts.append(param_result.query_title)
        if user_style_values:
            candidate_texts.append(user_style_values)
        if not brand_list and param_result.brand:
            candidate_texts.append(param_result.brand)

        if not candidate_texts:
            return ""

        cleaned = cls._clean_query_title(",".join(candidate_texts))
        candidates = cls._split_query_terms(cleaned)
        if not candidates:
            return ""

        supplements: list[str] = []
        seen: set[str] = set()
        for term in candidates:
            norm = cls._normalize_query_term(term)
            if not norm or norm in seen:
                continue
            supplements.append(term)
            seen.add(norm)
            if len(supplements) >= cls._QUERY_TITLE_TERM_LIMIT:
                break

        return " ".join(supplements)

    def _resolve_shop_query_title(
        self, context: RequestContext, state: ZhiyiWorkflowState, mode: str
    ) -> str | None:
        if mode == "none":
            return None
        property_list = state.get("property_list") or []
        terms = self._extract_property_terms(property_list)
        if not terms:
            param_result: ZhiyiParseParam = state["param_result"]
            terms = self._split_property_terms(param_result.properties)
        if terms:
            limited = self._dedupe_query_terms(terms)[: self._QUERY_TITLE_TERM_LIMIT]
            return " ".join(limited)
        return self._resolve_query_title(context, mode)

    @staticmethod
    def _resolve_shop_filters(
        context: RequestContext, use_param: bool
    ) -> tuple[list[str], str | None]:
        if use_param:
            return context.shop_label_list, context.shop_type
        return [], None

    def _build_request_context(self, state: ZhiyiWorkflowState) -> RequestContext:
        param_result: ZhiyiParseParam = state["param_result"]
        sort_result: ZhiyiSortResult = state.get("sort_result")
        property_list = state.get("property_list") or []
        user_style_values = state.get("user_style_values") or ""

        sort_field = sort_result.sort_type_final if sort_result else ""
        normalized_sort = str(sort_field).strip().lower()
        if not normalized_sort or normalized_sort in ("默认", "default"):
            sort_field = "newItemScore" if param_result.type == "新品" else "aggSaleVolume"

        start_date = param_result.start_date or ""
        end_date = param_result.end_date or ""
        sale_start_date = param_result.sale_start_date or ""
        sale_end_date = param_result.sale_end_date or ""

        if param_result.type == "新品" and not sale_start_date and start_date:
            sale_start_date = start_date
            sale_end_date = end_date

        brand_list = state.get("brand_list") or []

        return RequestContext(
            start_date=start_date,
            end_date=end_date,
            sale_start_date=sale_start_date,
            sale_end_date=sale_end_date,
            category_id_list=param_result.category_id or [],
            root_category_id_list=(
                [param_result.root_category_id] if param_result.root_category_id else []
            ),
            property_list=property_list,
            brand_list=brand_list,
            min_volume=param_result.low_volume or 0,
            max_volume=(
                param_result.high_volume if param_result.high_volume is not None else 99999999
            ),
            min_coupon_cprice=param_result.low_price or 0,
            max_coupon_cprice=param_result.high_price or 999999,
            sort_field=sort_field,
            query_title_main=self._build_query_title_main(
                param_result=param_result,
                property_list=property_list,
                brand_list=brand_list,
                user_style_values=user_style_values,
            ),
            query_title_fallback=param_result.query_title or "",
            limit=param_result.limit or 6000,
            shop_label_list=param_result.shop_switch or [],
            shop_type=param_result.shop_type,
        )

    # API 分支配置：api_branch -> (api_method, log_name)
    _API_BRANCH_CONFIG = {
        "shop": ("search_shop_goods", "店铺"),
        "monitor_hot": ("search_hot_sale", "监控热销"),
        "monitor_new": ("search_simple_goods", "监控新品"),
        "all_hot": ("search_hot_sale", "全网热销"),
        "all_new": ("search_simple_goods", "全网新品"),
    }

    def _call_main_api(self, state: ZhiyiWorkflowState, api_branch: str) -> dict[str, Any]:
        """通用主 API 调用方法 - 合并 5 个主 API 节点的公共逻辑"""
        req = state["request"]
        api_request = self._build_request_for_branch(state, api_branch, is_fallback=False)

        api_method_name, log_name = self._API_BRANCH_CONFIG[api_branch]

        try:
            api_client = get_zhiyi_api_client()
            api_method = getattr(api_client, api_method_name)
            page_result = api_method(
                user_id=req.user_id,
                team_id=req.team_id,
                params=api_request,
            )
            result_count = page_result.result_count or 0
            goods_list = [g.model_dump(by_alias=True) for g in page_result.result_list]
            return {
                "api_branch": api_branch,
                "api_request": api_request,
                "goods_list": goods_list,
                "result_count": result_count,
                "api_success": True,
                "browsed_count": result_count,
            }
        except Exception as e:
            logger.error(f"[知衣] {log_name} API 调用失败: {e}")
            return {
                "api_branch": api_branch,
                "api_request": api_request,
                "goods_list": [],
                "result_count": 0,
                "api_success": False,
                "browsed_count": 0,
            }

    def _api_monitor_hot_node(self, state: ZhiyiWorkflowState) -> dict[str, Any]:
        """监控店铺热销 API"""
        return self._call_main_api(state, "monitor_hot")

    def _api_shop_node(self, state: ZhiyiWorkflowState) -> dict[str, Any]:
        """店铺 API"""
        return self._call_main_api(state, "shop")

    def _api_monitor_new_node(self, state: ZhiyiWorkflowState) -> dict[str, Any]:
        """监控店铺新品 API"""
        return self._call_main_api(state, "monitor_new")

    def _api_all_hot_node(self, state: ZhiyiWorkflowState) -> dict[str, Any]:
        """全网热销 API"""
        return self._call_main_api(state, "all_hot")

    def _api_all_new_node(self, state: ZhiyiWorkflowState) -> dict[str, Any]:
        """全网新品 API"""
        return self._call_main_api(state, "all_new")

    # ==================== 兜底 API 节点（通用实现）====================

    def _build_fallback_request(self, state: ZhiyiWorkflowState, api_branch: str) -> dict[str, Any]:
        """构建兜底请求（严格对齐 n8n）"""
        return self._build_request_for_branch(state, api_branch, is_fallback=True)

    def _call_fallback_api(self, state: ZhiyiWorkflowState, api_branch: str) -> dict[str, Any]:
        """通用兜底 API 调用方法 - 合并 5 个兜底 API 节点的公共逻辑"""
        req = state["request"]
        self._get_pusher(req).complete_phase(
            phase_name="商品筛选中",
            variables={
                "datasource": "知衣商品库",
                "platform": "淘宝",
                "browsed_count": str(random.randint(100000, 1000000)),
                "filter_result_text": "当前筛选出的商品数量不足，不满足列表需求，我将再次搜索确保无遗漏",
            },
        )
        fallback_request = self._build_fallback_request(state, api_branch)

        api_method_name, log_name = self._API_BRANCH_CONFIG[api_branch]

        try:
            api_client = get_zhiyi_api_client()
            api_method = getattr(api_client, api_method_name)
            page_result = api_method(
                user_id=req.user_id,
                team_id=req.team_id,
                params=fallback_request,
            )
            result_count = page_result.result_count or 0
            goods_list = [g.model_dump(by_alias=True) for g in page_result.result_list]
            return {
                "fallback_api_request": fallback_request,
                "fallback_goods_list": goods_list,
                "fallback_result_count": result_count,
                "fallback_api_success": True,
            }
        except Exception as e:
            logger.error(f"[知衣] {log_name}兜底 API 调用失败: {e}")
            return {
                "fallback_api_request": fallback_request,
                "fallback_goods_list": [],
                "fallback_result_count": 0,
                "fallback_api_success": False,
            }

    def _fallback_shop_node(self, state: ZhiyiWorkflowState) -> dict[str, Any]:
        """店铺兜底 API"""
        return self._call_fallback_api(state, "shop")

    def _fallback_monitor_hot_node(self, state: ZhiyiWorkflowState) -> dict[str, Any]:
        """监控热销兜底 API"""
        return self._call_fallback_api(state, "monitor_hot")

    def _fallback_monitor_new_node(self, state: ZhiyiWorkflowState) -> dict[str, Any]:
        """监控新品兜底 API"""
        return self._call_fallback_api(state, "monitor_new")

    def _fallback_all_hot_node(self, state: ZhiyiWorkflowState) -> dict[str, Any]:
        """全网热销兜底 API"""
        return self._call_fallback_api(state, "all_hot")

    def _fallback_all_new_node(self, state: ZhiyiWorkflowState) -> dict[str, Any]:
        """全网新品兜底 API"""
        return self._call_fallback_api(state, "all_new")

    # ==================== 汇聚节点 ====================

    def _merge_results(self, state: ZhiyiWorkflowState) -> list[dict[str, Any]]:
        """汇聚主调用和兜底调用结果 - 通用逻辑"""
        result_count = state.get("result_count")
        goods_list = state.get("goods_list") or []
        fallback_goods_list = state.get("fallback_goods_list") or []

        # 优先使用主调用结果，如果不足则用兜底结果补充
        if result_count >= 50:
            return goods_list
        if not fallback_goods_list:
            return goods_list

        merged: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        # 去重补充兜底结果，避免重复商品
        def _append(items: list[dict[str, Any]]) -> None:
            for item in items:
                if not isinstance(item, dict):
                    merged.append(item)
                    continue
                goods_id = (
                    item.get("goodsId")
                    or item.get("goods_id")
                    or item.get("itemId")
                    or item.get("spuId")
                )
                if goods_id is not None:
                    key = str(goods_id)
                    if key in seen_ids:
                        continue
                    seen_ids.add(key)
                merged.append(item)

        _append(goods_list)
        _append(fallback_goods_list)
        return merged

    def _push_merge_progress(
        self, state: ZhiyiWorkflowState, final_count: int
    ) -> None:
        """推送 merge 节点的进度 - 根据是否发生兜底请求决定步骤"""
        pusher = self._get_pusher(state["request"])
        datasource = "知衣商品库"
        platform = "淘宝"
        fallback_attempted = state.get("fallback_api_request") is not None
        fallback_count = state.get("fallback_result_count") or 0
        if fallback_attempted:
            fallback_has_results = fallback_count > 0
            if fallback_has_results and final_count > 0:
                retry_result_text = (
                    "商品筛选完成，当前筛选出的商品数量符合报表要求，接下来我将根据数据绘制商品列表"
                )
                browsed_count_display = str(random.randint(100000, 1000000))
            else:
                retry_result_text = (
                    f"当前筛选出的商品数量为{final_count}，我可能需要提醒用户调整筛选维度"
                )
                browsed_count_display = str(random.randint(100000, 1000000))
            pusher.complete_phase(
                phase_name="二次筛选中",
                variables={
                    "datasource": datasource,
                    "platform": platform,
                    "browsed_count": browsed_count_display,
                    "retry_result_text": retry_result_text,
                },
            )
        else:
            if final_count >= 50:
                filter_result_text = "商品筛选完成，当前筛选出的商品数量满足需求，无需二次筛选，接下来我将根据数据完成商品列表"
            elif final_count > 0:
                filter_result_text = (
                    "商品筛选完成，我将根据筛选结果完成商品列表"
                )
            else:
                filter_result_text = "当前筛选出的商品数量不满足需求，我可能需要提醒用户调整筛选维度"
            pusher.complete_phase(
                phase_name="商品筛选中",
                variables={
                    "datasource": datasource,
                    "platform": platform,
                    "browsed_count": str(random.randint(100000, 1000000)),
                    "filter_result_text": filter_result_text,
                },
            )

    def _merge_node(self, state: ZhiyiWorkflowState) -> dict[str, Any]:
        """汇聚节点：合并主调用和兜底调用结果"""
        merged_goods_list = self._merge_results(state)
        goods_list = state.get("goods_list") or []
        used_fallback_for_results = len(merged_goods_list) > len(goods_list)
        final_count = len(merged_goods_list)
        if used_fallback_for_results:
            api_request = state.get("fallback_api_request")
        else:
            api_request = state.get("api_request")

        # 推送筛选完成进度
        self._push_merge_progress(state, final_count)

        return {
            "merged_goods_list": merged_goods_list,
            "browsed_count": final_count,
            "api_request": api_request,
            "used_fallback": used_fallback_for_results,
        }

    # ==================== 后处理节点 ====================

    def _post_process_node(self, state: ZhiyiWorkflowState) -> dict[str, Any]:
        """后处理节点"""
        merged_goods_list = state.get("merged_goods_list") or []

        # 提取商品 ID 列表
        goods_id_list = []
        processed_goods_list = []

        for goods in merged_goods_list:
            goods_id = (
                goods.get("goodsId")
                or goods.get("goods_id")
                or goods.get("itemId")
                or goods.get("spuId")
            )
            if goods_id:
                goods_id_list.append(str(goods_id))
                item = dict(goods)
                item["商品id"] = str(goods_id)
                processed_goods_list.append(item)

        return {
            "goods_id_list": goods_id_list,
            "processed_goods_list": processed_goods_list,
        }

    def _has_result_node(self, state: ZhiyiWorkflowState) -> dict[str, Any]:
        """结果输出节点"""
        req = state["request"]
        param_result: ZhiyiParseParam = state["param_result"]
        processed_goods_list = state.get("processed_goods_list") or []
        api_branch = state.get("api_branch")

        has_query_result = len(processed_goods_list) > 0

        pusher = self._get_pusher(state["request"])
        if has_query_result:
            pusher.complete_phase("生成列表中")
            pusher.complete_phase("选品完成")

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

            # 推送数据源路径
            query_params: list[Any] = []
            datasource_key = self._get_datasource_key(state)
            path_message = BaseRedisMessage(
                session_id=req.session_id,
                reply_message_id=req.message_id,
                reply_id=f"reply_{req.message_id}",
                reply_seq=0,
                operate_id="输出参数",
                status="RUNNING",
                content_type=8,
                content=CustomDataContent(
                    data={"entity_type": 1, "content": datasource_key["content"], "query_params": query_params}
                ),
                create_ts=int(round(time.time() * 1000)),
            )
            redis_client.list_left_push(
                RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
                path_message.model_dump_json(),
            )

            # 推送 API 参数（对齐 INS 工作流格式）
            api_request = state.get("api_request")
            request_path = self._resolve_request_path(state.get("api_branch"))
            # 排除 None 值，避免后端接收大量 null
            filtered_request = {k: v for k, v in api_request.items() if v is not None}
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
                        request_path=request_path,
                        request_body=json.dumps(filtered_request, ensure_ascii=False),
                        actions=["view", "export", "download"],
                        title=param_result.title or "",
                        entity_type=1,
                        filters=state.get("user_filter_tags") or [],
                    )
                ),
                create_ts=int(round(time.time() * 1000)),
            )
            redis_client.list_left_push(
                RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
                parameter_message.model_dump_json(exclude_none=True),
            )
            logger.debug(f"[知衣工作流]推送选品参数输出消息：{parameter_message.model_dump_json(ensure_ascii=False, exclude_none=True)}")
        else:
            # 无结果消息
            pusher.fail_phase(phase_name="生成列表失败", error_message="我未能完成列表绘制，原因是没有数据\n需要提醒用户调整筛选维度，才能更好的获取数据",)
            pusher.fail_phase(phase_name="选品未完成", error_message=None)

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

            # 记录选品失败的参数
            api_request = state.get("api_request")
            filtered_request = {k: v for k, v in api_request.items() if v is not None}
            with pg_session(DBAlias.DB_ABROAD_AI.value) as session:
                session.execute(
                    text("""
                         INSERT INTO zxy_workflow_select_fail_trace(session_id, message_id, api_request, insert_time, env)
                         VALUES (:session_id, :message_id, :api_request, :insert_time, :env)
                         """),
                    {
                        "session_id": req.session_id,
                        "message_id": req.message_id,
                        "api_request": json.dumps(filtered_request, ensure_ascii=False),
                        "insert_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "env": settings.environment,
                    },
                )

            # no_result_message = BaseRedisMessage(
            #     session_id=req.session_id,
            #     reply_message_id=req.message_id,
            #     reply_id=f"reply_{req.message_id}",
            #     reply_seq=0,
            #     operate_id="结果",
            #     status="END",
            #     content_type=1,
            #     content=TextMessageContent(text="未找到符合需求的商品，请尝试调整筛选条件。"),
            #     create_ts=int(round(time.time() * 1000)),
            # )
            # redis_client.list_left_push(
            #     RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            #     no_result_message.model_dump_json(),
            # )


            # 临时部分：失败情况下也推送选品参数，方便文搜图测试
            # 推送 API 参数（对齐 INS 工作流格式）
            api_request = state.get("api_request")
            request_path = self._resolve_request_path(state.get("api_branch"))
            # 排除 None 值，避免后端接收大量 null
            filtered_request = {k: v for k, v in api_request.items() if v is not None}
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
                        request_path=request_path,
                        request_body=json.dumps(filtered_request, ensure_ascii=False),
                        actions=["view", "export", "download"],
                        title=param_result.title or "",
                        entity_type=1,
                        filters=state.get("user_filter_tags") or [],
                    )
                ),
                create_ts=int(round(time.time() * 1000)),
            )
            redis_client.list_left_push(
                RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
                parameter_message.model_dump_json(exclude_none=True),
            )
            logger.debug(f"[知衣工作流]推送选品参数输出消息：{parameter_message.model_dump_json(ensure_ascii=False, exclude_none=True)}")
            # 临时部分结束

        entity_simple_data = processed_goods_list

        return {
            "has_query_result": has_query_result,
            "entity_simple_data": entity_simple_data,
        }

    def _package_result_node(self, state: ZhiyiWorkflowState) -> dict[str, Any]:
        """封装返回结果节点"""
        has_query_result = state.get("has_query_result", False)
        entity_data = state.get("entity_simple_data", [])
        if has_query_result:
            entity_dicts = [e if isinstance(e, dict) else e.model_dump() for e in entity_data]
            response = WorkflowResponse(
                select_result="基于以上条件，您的选品任务已经完成，还有其他需要帮助的地方吗？",
                relate_data=json.dumps(entity_dicts, ensure_ascii=False),
            )
        else:
            response = WorkflowResponse(
                select_result="无结果，可能与价格、销量、参考历史选品经验有关",
                relate_data=None,
            )

        return {"workflow_response": response}


__all__ = ["ZhiyiGraph"]
