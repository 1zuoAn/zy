from __future__ import annotations

import json
import random
import re
import time
from datetime import datetime, timedelta
from typing import Any

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
    WorkflowMessageContentType, WorkflowEntityType,
)
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
from app.schemas.entities.workflow.graph_state import AbroadGoodsWorkflowState
from app.schemas.entities.workflow.llm.douyi_output import DouyiUserTagResult
from app.schemas.entities.workflow.llm.abroad_output import (
    AbroadGoodsAttributeParseResult,
    AbroadGoodsBrandParseResult,
    AbroadGoodsBrandResult,
    AbroadGoodsCategoryParseResult,
    AbroadGoodsNumericParseResult,
    AbroadGoodsParamMergeResult,
    AbroadGoodsParseParam,
    AbroadGoodsPlatformRouteParseResult,
    AbroadGoodsRegionParseResult,
    AbroadGoodsSortResult,
    AbroadGoodsTextTitleParseResult,
    AbroadGoodsTimeParseResult,
)
from app.schemas.request.workflow_request import WorkflowRequest
from app.schemas.response.workflow_response import WorkflowResponse
from app.service.chains.templates.abroad_goods_progress_template import (
    ABROAD_GOODS_PROGRESS_TEMPLATE,
)
from app.service.chains.workflow.base_graph import BaseWorkflowGraph
from app.utils import thread_pool
from app.service.chains.workflow.progress_pusher import PhaseProgressPusher
from app.service.chains.workflow.select.amazon_subgraph import get_amazon_subgraph
from app.service.chains.workflow.select.independent_sites_subgraph import (
    get_independent_sites_subgraph,
)
from app.service.chains.workflow.select.temu_subgraph import get_temu_subgraph
from app.service.rpc.abroad_api import (
    AbroadGoodsSearchRequest,
    get_abroad_api,
)
from app.service.rpc.volcengine_kb_api import get_volcengine_kb_api
from app.utils import thread_pool
from app.utils.query_reference import QueryReferenceHelper


class AbroadGoodsGraph(BaseWorkflowGraph):
    """海外探款商品选品工作流 - LangGraph 版本"""

    span_name = "海外探款商品数据源工作流"
    run_name = "abroad-goods-graph"

    # 使用 templates 文件夹的进度模板
    PROGRESS_TEMPLATE = ABROAD_GOODS_PROGRESS_TEMPLATE

    _NODE_NAME_MAP = {
        "init_state": "初始化",
        "pre_think": "预处理思考",
        "query_selections": "查询筛选项",
        "llm_parse": "LLM参数解析",
        "llm_merge": "LLM参数合并",
        "parallel_parse": "并行解析",
        "amazon_search_goods": "Amazon专区查询",
        "temu_search_goods": "Temu专区查询",
        "match_sites": "站点匹配",
        "site_detail_search": "站点详情搜索",
        "goods_center_search": "全网数据搜索",
        "monitor_route": "监控站点路由",
        "monitor_site_new": "监控站点新品",
        "monitor_site_hot": "监控站点热销",
        "fallback_goods_center": "全网兜底搜索",
        "fallback_monitor_new": "监控新品兜底",
        "fallback_monitor_hot": "监控热销兜底",
        "merge_search_result": "结果汇聚",
        "post_process": "后处理",
        "has_result": "有结果处理",
        "no_result": "无结果兜底",
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

    # ==================== 工具方法 ====================
    def _get_pusher(self, req: Any) -> PhaseProgressPusher:
        """获取进度推送器"""
        return PhaseProgressPusher(template=self.PROGRESS_TEMPLATE, request=req)

    def _build_graph(self) -> CompiledStateGraph:
        """构建工作流图"""
        graph = StateGraph(AbroadGoodsWorkflowState)  # type: ignore[arg-type]

        # ===== 预处理节点 =====
        graph.add_node("init_state", self._init_state_node, metadata={"__display_name__": "初始化"})  # type: ignore[arg-type]
        graph.add_node("pre_think", self._pre_think_node, metadata={"__display_name__": "预处理思考"})  # type: ignore[arg-type]
        graph.add_node("query_selections", self._query_common_selection_node, metadata={"__display_name__": "查询筛选项"})  # type: ignore[arg-type]
        graph.add_node("llm_parse", self._llm_param_parse_node, metadata={"__display_name__": "LLM参数解析"})  # type: ignore[arg-type]
        graph.add_node("llm_merge", self._llm_param_merge_node, metadata={"__display_name__": "LLM参数合并"})  # type: ignore[arg-type]
        graph.add_node("parallel_parse", self._parallel_parse_node, metadata={"__display_name__": "并行解析"})  # type: ignore[arg-type]

        # ===== 专区路由节点 (根据 zone_type 分支) =====
        graph.add_node("amazon_search_goods", self._amazon_search_goods_node, metadata={"__display_name__": "Amazon专区查询"})  # type: ignore[arg-type]
        graph.add_node("temu_search_goods", self._temu_search_goods_node, metadata={"__display_name__": "Temu专区查询"})  # type: ignore[arg-type]
        graph.add_node("match_sites", self._match_sites_node, metadata={"__display_name__": "站点匹配"})  # type: ignore[arg-type]
        graph.add_node("site_detail_search", self._site_detail_search_node, metadata={"__display_name__": "站点详情搜索"})  # type: ignore[arg-type]
        graph.add_node("goods_center_search", self._goods_center_search_node, metadata={"__display_name__": "全网数据搜索"})  # type: ignore[arg-type]
        graph.add_node("monitor_route", lambda s: s, metadata={"__display_name__": "监控站点路由"})  # type: ignore[arg-type]
        graph.add_node("monitor_site_new", self._monitor_site_new_node, metadata={"__display_name__": "监控站点新品"})  # type: ignore[arg-type]
        graph.add_node("monitor_site_hot", self._monitor_site_hot_node, metadata={"__display_name__": "监控站点热销"})  # type: ignore[arg-type]
        graph.add_node("fallback_goods_center", self._fallback_goods_center_node, metadata={"__display_name__": "全网兜底搜索"})  # type: ignore[arg-type]
        graph.add_node("fallback_monitor_new", self._fallback_monitor_new_node, metadata={"__display_name__": "监控新品兜底"})  # type: ignore[arg-type]
        graph.add_node("fallback_monitor_hot", self._fallback_monitor_hot_node, metadata={"__display_name__": "监控热销兜底"})  # type: ignore[arg-type]

        # ===== 汇聚节点 (统一处理) =====
        graph.add_node( "merge_search_result", self._merge_search_result_node, metadata={"__display_name__": "结果汇聚"})  # type: ignore[arg-type]

        # ===== 后处理节点 =====
        graph.add_node("post_process", self._post_process_node, metadata={"__display_name__": "后处理"})  # type: ignore[arg-type]
        graph.add_node("has_result", self._has_result_node, metadata={"__display_name__": "有结果处理"})  # type: ignore[arg-type]
        graph.add_node("no_result", self._no_result_node, metadata={"__display_name__": "无结果兜底"})  # type: ignore[arg-type]
        graph.add_node("package", self._package_result_node, metadata={"__display_name__": "封装结果"})  # type: ignore[arg-type]

        # ===== 入口和预处理边 =====
        graph.set_entry_point("init_state")
        graph.add_edge("init_state", "pre_think")
        graph.add_edge("pre_think", "query_selections")
        graph.add_edge("query_selections", "llm_parse")
        graph.add_edge("llm_parse", "llm_merge")
        graph.add_edge("llm_merge", "parallel_parse")

        # ===== 专区路由 (根据 zone_type) =====
        graph.add_conditional_edges(
            "parallel_parse",
            self._route_by_zone_type,
            {
                "amazon": "amazon_search_goods",
                "temu": "temu_search_goods",
                "default": "match_sites",
            },
        )

        # ===== Amazon/Temu 分支直达汇聚 =====
        graph.add_edge("amazon_search_goods", "merge_search_result")
        graph.add_edge("temu_search_goods", "merge_search_result")

        # ===== 默认分支：站点匹配 + 路由 =====
        graph.add_conditional_edges(
            "match_sites",
            self._route_default_by_site_and_flag,
            {
                "site_detail": "site_detail_search",
                "monitor": "monitor_route",
                "goods_center": "goods_center_search",
            },
        )

        # ===== 监控站点分支 =====
        graph.add_conditional_edges(
            "monitor_route",
            self._route_by_new_type,
            {"new": "monitor_site_new", "hot": "monitor_site_hot"},
        )
        graph.add_conditional_edges(
            "monitor_site_new",
            self._check_search_result,
            {"has_result": "merge_search_result", "no_result": "fallback_monitor_new"},
        )
        graph.add_conditional_edges(
            "monitor_site_hot",
            self._check_search_result,
            {"has_result": "merge_search_result", "no_result": "fallback_monitor_hot"},
        )
        graph.add_edge("fallback_monitor_new", "merge_search_result")
        graph.add_edge("fallback_monitor_hot", "merge_search_result")

        # ===== 站点详情分支 =====
        graph.add_edge("site_detail_search", "merge_search_result")

        # ===== 全网数据分支 =====
        graph.add_conditional_edges(
            "goods_center_search",
            self._check_search_result,
            {"has_result": "merge_search_result", "no_result": "fallback_goods_center"},
        )
        graph.add_edge("fallback_goods_center", "merge_search_result")
        graph.add_edge("merge_search_result", "post_process")

        # ===== 后处理和输出 =====
        graph.add_conditional_edges(
            "post_process",
            self._check_final_result,
            {"has_result": "has_result", "no_result": "no_result"},
        )
        graph.add_edge("has_result", "package")
        graph.add_edge("no_result", "package")
        graph.add_edge("package", END)

        return graph.compile()

    def _route_by_zone_type(self, state: AbroadGoodsWorkflowState) -> str:
        """根据 zone_type 路由到不同的搜索节点"""
        param_result: AbroadGoodsParseParam = state.get("param_result")

        # 防御性检查：param_result为空时使用默认路由
        if not param_result:
            logger.warning("[海外探款商品] LLM参数解析失败，使用默认搜索路由(default)")
            return "default"

        zone_type = param_result.zone_type
        shop_id = state.get("shop_id")
        if shop_id:
            req: WorkflowRequest | None = state.get("request")
            shop_zone_type = (
                state.get("shop_zone_type")
                or zone_type
                or self._infer_zone_type(getattr(param_result, "type", None))
                or self._infer_zone_type(getattr(req, "abroad_type", None) if req else None)
            )
            if shop_zone_type == "amazon":
                return "amazon"
            if shop_zone_type == "temu":
                return "temu"

        if zone_type == "amazon":
            return "amazon"
        elif zone_type == "temu":
            return "temu"
        else:
            logger.debug(f"[海外探款商品] 不支持的zone_type: {zone_type}，使用默认路由")
            return "default"

    def _check_search_result(self, state: AbroadGoodsWorkflowState) -> str:
        """检查搜索 API 结果"""
        result_count = state.get("result_count", 0)
        api_success = state.get("api_success", True)
        if not api_success or (result_count is not None and result_count < 50):
            return "no_result"
        return "has_result"

    def _route_default_by_site_and_flag(self, state: AbroadGoodsWorkflowState) -> str:
        """默认分支路由 - 对齐 n8n 的"判断查询数据类型"节点"""
        param_result: AbroadGoodsParseParam = state.get("param_result")
        is_single_platform = state.get("is_single_platform", False)

        # n8n: 单站点直接走站点详情
        if is_single_platform:
            return "site_detail"

        # 其他情况根据 flag 判断走监控还是全网
        flag = param_result.flag if param_result else 2
        return "monitor" if flag == 1 else "goods_center"

    def _route_by_new_type(self, state: AbroadGoodsWorkflowState) -> str:
        """监控站点新品/热销路由"""
        param_result: AbroadGoodsParseParam = state.get("param_result")
        new_type = param_result.new_type if param_result else ""
        return "new" if new_type == "新品" else "hot"

    def _build_goods_request(
        self,
        param_result: AbroadGoodsParseParam,
        sort_type: int,
        platform_types: list[str] | None,
        style_list: list[str] | None = None,
        brand_tags: list[str] | None = None,
    ) -> AbroadGoodsSearchRequest:
        """构建商品搜索请求 - 对齐 n8n 的参数构建逻辑

        功能:
        - 将解析后的参数转换为 AbroadGoodsSearchRequest 对象
        - 支持平台类型、风格、品牌等参数的覆盖

        参数:
            param_result: LLM 解析的参数结果
            sort_type: 排序类型
            platform_types: 平台类型列表（可选覆盖）
            style_list: 风格列表（可选）
            brand_tags: 品牌标签列表（可选）

        返回:
            AbroadGoodsSearchRequest: 构建好的搜索请求对象
        """
        return AbroadGoodsSearchRequest.from_parse_param(
            param_result,
            style_list=style_list,
            sort_type=sort_type,
            platform_type_list_override=platform_types,
            brand_tags=brand_tags,
        )

    def _check_final_result(self, state: AbroadGoodsWorkflowState) -> str:
        """检查最终是否有结果 - 路由函数

        功能:
        - 检查处理后是否有商品结果
        - 用于路由到"有结果"或"无结果"分支

        返回:
            "has_result": 有结果，进入有结果处理分支
            "no_result": 无结果，进入无结果处理分支
        """
        processed_goods_list = state.get("processed_goods_list", [])
        if processed_goods_list and len(processed_goods_list) > 0:
            return "has_result"
        return "no_result"

    # ==================== 节点实现 ====================

    def _init_state_node(self, state: AbroadGoodsWorkflowState) -> dict[str, Any]:
        """初始化状态节点 - 对齐 n8n 的初始化逻辑

        对应 n8n 节点: 工作流初始化

        功能:
        - 记录用户查询到数据库（异步执行）
        - 初始化工作流状态

        注意: 数据库插入操作在后台线程池中异步执行，不阻塞主流程。
        """
        req = state["request"]

        def insert_start_track() -> None:
            with pg_session(DBAlias.DB_ABROAD_AI.value) as session:
                session.execute(
                    text("""
                        INSERT INTO holo_zhiyi_aiagent_query_prod(query, session_id, message_id, user_id, team_id)
                        VALUES (:query, :session_id, :message_id, :user_id, :team_id)
                    """),
                    {
                        "query": req.user_query,
                        "session_id": req.session_id,
                        "message_id": req.message_id,
                        "user_id": req.user_id,
                        "team_id": req.team_id,
                    },
                )

        thread_pool.submit_with_context(insert_start_track)
        return {}

    def _pre_think_node(self, state: AbroadGoodsWorkflowState) -> dict[str, Any]:
        """预处理节点 - 对齐 n8n 的预处理逻辑

        对应 n8n 节点: 预处理相关节点

        功能:
        - 发送开始消息给用户
        - 执行思维链推理（CozeLoop prompt）
        - 推送任务规划进度

        返回:
            dict: 空字典（仅触发异步任务）
        """
        req = state["request"]

        # 发送开始消息
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
                data=ParameterData(entity_type=WorkflowEntityType.ABROAD_ITEM.code),
            ),
            create_ts=int(round(time.time() * 1000)),
        )
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            start_message.model_dump_json(),
        )

        # 异步生成思维链（完成后会推送进度）
        def _generate_thinking_and_report_task() -> None:
            ref_helper = QueryReferenceHelper.from_request(req)
            format_query = ref_helper.replace_placeholders(req.user_query)
            industry_value = self._override_industry_by_query(req.industry, format_query)

            invoke_params = {
                "user_query": format_query,
                "preferred_entity": req.preferred_entity,
                "industry": industry_value,
                "user_preferences": req.user_preferences,
                "now_time": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
            }

            messages: list[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
                prompt_key=CozePromptHubKey.ABROAD_GOODS_THINK_PROMPT.value,
                variables=invoke_params,
            )
            llm: BaseChatModel = llm_factory.get_llm(
                LlmProvider.OPENROUTER.name, LlmModelName.OPENROUTER_GEMINI_3_FLASH_PREVIEW.value
            )
            retry_llm = llm.with_retry(stop_after_attempt=2)

            thinking_chain = retry_llm | StrOutputParser()
            thinking_result_text = thinking_chain.with_config(run_name="思维链生成").invoke(
                messages
            )
            thinking_result_text = thinking_result_text.replace("\n\n", "\n")

            # 推送思维链
            self._get_pusher(req=req).complete_phase("选品任务规划", content=thinking_result_text)

        _generate_thinking_and_report_task()
        return {}

    def _override_industry_by_query(self, industry: str | None, user_query: str) -> str | None:
        """当 industry 与 user_query 冲突时，优先使用 user_query 的性别意图。"""
        if not industry or not user_query:
            return industry

        query_text = str(user_query)
        industry_text = str(industry)
        gender_hint = None

        if "男装" in query_text:
            gender_hint = "男装"
        elif "女装" in query_text:
            gender_hint = "女装"

        if gender_hint and gender_hint not in industry_text:
            logger.debug(
                f"[海外探款] industry与user_query冲突，使用user_query: {gender_hint}"
            )
            return gender_hint

        return industry

    def _parse_path_tokens(self, raw: str) -> list[str]:
        return [t.strip() for t in raw.split(",") if t.strip()]

    def _extract_category_root_id(
        self, industry: str | None, category_paths: list[dict[str, Any]]
    ) -> str | None:
        """从 industry 或类目表中解析根类目 ID（如 女装#1 -> 1）。"""
        if not industry:
            return None

        raw = str(industry).strip()
        if not raw:
            return None

        name_part = raw.split("#", 1)[0].strip()
        if "#" in raw:
            _, id_part = raw.split("#", 1)
            ids = self._parse_path_tokens(id_part)
            if ids:
                return ids[0]

        if not name_part:
            return None

        for item in category_paths:
            name_path = self._parse_path_tokens(str(item.get("name_path") or ""))
            id_path = self._parse_path_tokens(str(item.get("id_path") or ""))
            if name_path and id_path and name_path[0] == name_part:
                return id_path[0]

        return None

    def _expand_category_paths(
        self,
        category_paths: list[dict[str, Any]],
        root_id: str | None = None,
    ) -> list[str]:
        """将类目路径展开为可选前缀路径，供 LLM 选择父类目。"""
        candidates: list[str] = []
        seen: set[str] = set()
        for item in category_paths:
            name_path = self._parse_path_tokens(str(item.get("name_path") or ""))
            id_path = self._parse_path_tokens(str(item.get("id_path") or ""))
            if not name_path or not id_path:
                continue
            if root_id and id_path[0] != root_id:
                continue
            max_len = min(len(name_path), len(id_path))
            for i in range(1, max_len + 1):
                prefix_ids = id_path[:i]
                key = ",".join(prefix_ids)
                if key in seen:
                    continue
                seen.add(key)
                prefix_names = name_path[:i]
                candidates.append(f"{','.join(prefix_names)}#{key}")
        return candidates

    def _expand_category_vector(
        self,
        category_vector: list[str],
        root_id: str | None = None,
    ) -> list[str]:
        """将类目向量结果展开为可选前缀路径。"""
        candidates: list[str] = []
        seen: set[str] = set()
        for item in category_vector:
            if not item:
                continue
            value = str(item).strip()
            if not value:
                continue
            if "#" not in value:
                if value not in seen:
                    seen.add(value)
                    candidates.append(value)
                continue
            name_part, id_part = value.split("#", 1)
            name_path = self._parse_path_tokens(name_part)
            id_path = self._parse_path_tokens(id_part)
            if not name_path or not id_path:
                continue
            if root_id and id_path[0] != root_id:
                continue
            max_len = min(len(name_path), len(id_path))
            for i in range(1, max_len + 1):
                prefix_ids = id_path[:i]
                key = ",".join(prefix_ids)
                if key in seen:
                    continue
                seen.add(key)
                prefix_names = name_path[:i]
                candidates.append(f"{','.join(prefix_names)}#{key}")
        return candidates

    def _build_category_candidates(
        self,
        category_vector: list[str],
        category_paths: list[dict[str, Any]],
        industry: str | None,
    ) -> tuple[list[str], str | None]:
        """构建类目候选列表，允许选择父类目。"""
        root_id = self._extract_category_root_id(industry, category_paths)
        candidates = self._expand_category_vector(category_vector, root_id) if category_vector else []
        if not candidates:
            candidates = self._expand_category_paths(category_paths, root_id)
        if not candidates and root_id:
            candidates = (
                self._expand_category_vector(category_vector, None)
                if category_vector
                else self._expand_category_paths(category_paths, None)
            )
            root_id = None
        return candidates, root_id

    def _build_category_prefix_set(
        self, category_paths: list[dict[str, Any]], root_id: str | None = None
    ) -> set[tuple[str, ...]]:
        prefixes: set[tuple[str, ...]] = set()
        for item in category_paths:
            id_path = self._parse_path_tokens(str(item.get("id_path") or ""))
            if not id_path:
                continue
            if root_id and id_path[0] != root_id:
                continue
            for i in range(1, len(id_path) + 1):
                prefixes.add(tuple(id_path[:i]))
        return prefixes

    def _normalize_category_id_list(
        self, category_id_list: list[list[str]], prefix_set: set[tuple[str, ...]]
    ) -> list[list[str]]:
        if not category_id_list or not prefix_set:
            return category_id_list

        normalized: list[list[str]] = []
        for path in category_id_list:
            tokens = [str(v).strip() for v in path if str(v).strip()]
            if not tokens:
                continue
            if tuple(tokens) in prefix_set:
                normalized.append(tokens)
                continue
            for i in range(len(tokens) - 1, 0, -1):
                if tuple(tokens[:i]) in prefix_set:
                    normalized.append(tokens[:i])
                    break
        return normalized

    def _query_common_selection_node(self, state: AbroadGoodsWorkflowState) -> dict[str, Any]:
        """查询筛选项数据节点 - 对齐 n8n 的筛选项查询逻辑

        对应 n8n 节点:
        - "筛选项 API 查询"
        - "Call '海外探款通用类目检索'" → "Edit Fields27" → "行业品类及时间-API参数解析Agent"

        功能:
        - 并行执行筛选项API查询和类目向量检索
        - 获取筛选项数据（用于后续参数解析）
        - 获取类目向量检索结果（用于LLM参数解析）

        返回:
            selection_dict: 包含筛选项、类目向量与引用信息的字典
        """
        req = state["request"]
        
        # 检测品牌和站点引用
        ref_helper = QueryReferenceHelper.from_request(req)
        
        # 品牌引用
        brand_refs = ref_helper.get_by_type("ABROAD_BRAND") + ref_helper.get_by_type("ABROAD_GOODS_BRAND")
        brand_name = None
        if brand_refs:
            brand_name = brand_refs[0].get("display_name")
            logger.debug(f"[海外探款] 检测到品牌引用: {brand_name}")
        
        # 站点引用
        site_refs = ref_helper.get_by_type("ABROAD_GOODS_PLATFORM")
        site_name = None
        site_id = None
        if site_refs:
            site_name = site_refs[0].get("display_name")
            site_id = site_refs[0].get("entity_id")
            logger.debug(f"[海外探款] 检测到站点引用: {site_name} (ID: {site_id})")

        # 店铺引用（Amazon/Temu/通用）
        amazon_shop_refs = ref_helper.get_by_type("ABROAD_GOODS_AMAZON_SHOP")
        temu_shop_refs = ref_helper.get_by_type("ABROAD_GOODS_TEMU_SHOP")
        generic_shop_refs = ref_helper.get_by_type("ABROAD_GOODS_SHOP")
        shop_id = None
        shop_zone_type = None
        shop_name = None
        shop_ref = None
        if amazon_shop_refs and temu_shop_refs:
            logger.warning("[海外探款] 同时检测到Amazon和Temu店铺引用，优先使用Amazon")
        if amazon_shop_refs:
            shop_ref = amazon_shop_refs[0]
            shop_name = shop_ref.get("display_name")
            shop_id = shop_ref.get("entity_id")
            shop_zone_type = "amazon"
            logger.debug(f"[海外探款] 检测到Amazon店铺引用: {shop_name} (ID: {shop_id})")
        elif temu_shop_refs:
            shop_ref = temu_shop_refs[0]
            shop_name = shop_ref.get("display_name")
            shop_id = shop_ref.get("entity_id")
            shop_zone_type = "temu"
            logger.debug(f"[海外探款] 检测到Temu店铺引用: {shop_name} (ID: {shop_id})")
        elif generic_shop_refs:
            shop_ref = generic_shop_refs[0]
            shop_name = shop_ref.get("display_name")
            shop_id = shop_ref.get("entity_id")
            # 根据 abroad_type 推断平台类型
            abroad_type = req.abroad_type or ""
            if abroad_type and "temu" in abroad_type.lower():
                shop_zone_type = "temu"
            elif abroad_type and ("amazon" in abroad_type.lower() or "亚马逊" in abroad_type):
                shop_zone_type = "amazon"
            logger.debug(f"[海外探款] 检测到通用店铺引用: {shop_name} (ID: {shop_id}), 平台: {shop_zone_type}")
        if shop_id is not None and not str(shop_id).strip():
            shop_id = None
            shop_zone_type = None
        if shop_ref and shop_zone_type is None:
            inferred_zone = self._infer_zone_type(shop_ref.get("platform_name"))
            if inferred_zone:
                shop_zone_type = inferred_zone
                logger.debug(f"[海外探款] 店铺引用平台名推断平台: {shop_zone_type}")

        # 平台引用（Amazon/Temu 优先使用店铺引用中的 platformType）
        platform_type_override = None
        platform_zone_type = None
        platform_name = None
        if shop_ref:
            platform_type_override = shop_ref.get("platform_type")
            platform_name = shop_ref.get("platform_name")
            platform_zone_type = shop_zone_type
        if platform_type_override is None:
            platform_refs = ref_helper.get_by_type("ABROAD_GOODS_PLATFORM")
            if platform_refs:
                platform_ref = platform_refs[0]
                platform_type_override = platform_ref.get("platform_type") or platform_ref.get("entity_id")
                platform_name = platform_ref.get("platform_name") or platform_ref.get("display_name")
                if platform_name:
                    platform_zone_type = self._infer_zone_type(platform_name)
        if platform_zone_type is None:
            inferred_zone = self._infer_zone_type(req.abroad_type)
            if inferred_zone:
                platform_zone_type = inferred_zone
                if not platform_name:
                    abroad_type = str(req.abroad_type or "")
                    if "站" in abroad_type:
                        platform_name = abroad_type
        if platform_type_override is not None:
            platform_type_str = str(platform_type_override).strip()
            if platform_type_str:
                platform_type_override = (
                    int(platform_type_str) if platform_type_str.isdigit() else platform_type_str
                )
            else:
                platform_type_override = None
        if platform_name is not None and not str(platform_name).strip():
            platform_name = None
        if platform_type_override is not None:
            logger.debug(
                f"[海外探款] 检测到平台引用platformType: {platform_type_override} ({platform_name})"
            )

        # 并行执行：筛选项 API + 类目向量检索
        selection_future = thread_pool.submit_with_context(get_abroad_api().get_common_selections)
        category_vector_future = thread_pool.submit_with_context(
            self._fetch_category_vector,
            req.user_query,
            VolcKnowledgeServiceId.ABROAD_CATEGORY_VECTOR,
        )

        selection_data = selection_future.result()
        category_vector_content = category_vector_future.result()

        # 将类目向量检索结果作为 category_data（对齐 n8n 的 品类候选）
        selection_data["category_vector"] = category_vector_content
        
        # 添加品牌和站点引用信息
        selection_data["brand_name"] = brand_name
        selection_data["site_name"] = site_name
        selection_data["site_id"] = site_id
        selection_data["shop_id"] = shop_id
        selection_data["shop_name"] = shop_name
        selection_data["shop_zone_type"] = shop_zone_type
        selection_data["platform_type_override"] = platform_type_override
        selection_data["platform_name_override"] = platform_name
        selection_data["platform_zone_type"] = platform_zone_type

        # 不在此处推送进度，等待 parallel_parse 完成后推送 PLANNING_FINISH
        return {"selection_dict": selection_data}

    def _normalize_style_list(self, style_value: Any) -> list[str]:
        """将 LLM 输出的 style 字段规范为列表"""
        if not style_value:
            return []
        if isinstance(style_value, list):
            return [str(v).strip() for v in style_value if str(v).strip()]
        if isinstance(style_value, str):
            value = style_value.strip()
            if not value:
                return []
            if value.startswith("["):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return [str(v).strip() for v in parsed if str(v).strip()]
                except Exception:
                    pass
            return [value]
        value = str(style_value).strip()
        return [value] if value else []

    def _infer_zone_type(self, value: Any) -> str | None:
        """根据文本推断专区类型（amazon/temu）"""
        text = str(value or "").strip()
        if not text:
            return None
        lower = text.lower()
        if "temu" in lower:
            return "temu"
        if "amazon" in lower or "亚马逊" in text:
            return "amazon"
        return None

    def _map_design_detail_label(
        self, label_value: Any, common_selections: dict[str, Any]
    ) -> Any:
        """将设计细节标签映射为路径值"""
        if not label_value or not isinstance(label_value, str):
            return label_value

        value = label_value.strip()
        if not value:
            return label_value

        if any(sep in value for sep in [",", "，", ";", "；", "|", "\n"]):
            return label_value

        design_items = common_selections.get("设计细节") or []
        if not isinstance(design_items, list):
            return label_value

        for item in design_items:
            if str(item.get("label", "")).strip() == value:
                mapped = str(item.get("value", "")).strip()
                if mapped:
                    return mapped

        candidates = []
        for item in design_items:
            raw_value = str(item.get("value", "")).strip()
            if not raw_value:
                continue
            first_token = raw_value.split(",")[0].strip()
            if first_token == value:
                candidates.append(item)

        if not candidates:
            return label_value

        for item in candidates:
            if str(item.get("label", "")).strip() == "其他":
                mapped = str(item.get("value", "")).strip()
                if mapped:
                    return mapped

        mapped = str(candidates[0].get("value", "")).strip()
        return mapped or label_value

    def _recall_brand_tags(self, brand: str) -> list[str]:
        """使用火山知识库进行品牌召回"""
        kb_client = get_volcengine_kb_api()
        response = kb_client.simple_chat(query=brand, service_resource_id=VolcKnowledgeServiceId.ABROAD_SITE_BRAND_VECTOR.value)
        if not response.data or not response.data.result_list:
            return []

        content_list = []
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


    def _clean_brand_tags(self, brand_tags: list[str], user_query: str) -> list[str] | None:
        """清洗品牌召回结果 (对应 n8n 的 '封装多品牌' 节点)"""
        if not brand_tags:
            return []

        # 调用llm清洗
        structured_llm = llm_factory.get_llm(
            LlmProvider.OPENROUTER.name, LlmModelName.OPENROUTER_GEMINI_2_5_FLASH.value
        ).with_structured_output(schema=AbroadGoodsBrandResult).with_retry(stop_after_attempt=2)
        messages = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.ABROAD_GOODS_BRAND_CLEAN_PROMPT.value,
            variables={
                "query_tag_list": json.dumps(brand_tags, ensure_ascii=False),
                "origin_text": user_query
            },
        )
        result: AbroadGoodsBrandResult = structured_llm.with_config(run_name="召回品牌清洗").invoke(messages)

        # 解析llm输出
        clean_brand_list = result.clean_tag_list
        return clean_brand_list

    def _recall_and_clean_brand_tags(self, brand: str, req: WorkflowRequest) -> list[str]:
        """品牌召回 + LLM 清洗"""
        try:
            recalled_brand_list = self._recall_brand_tags(brand)
            if not recalled_brand_list:
                return []
            ref_helper = QueryReferenceHelper.from_request(req)
            format_query = ref_helper.replace_placeholders(req.user_query)
            cleaned = self._clean_brand_tags(recalled_brand_list, format_query)
            if cleaned is None:
                return []
            return cleaned
        except Exception as e:
            logger.warning(f"[海外探款商品品牌召回清洗]发生错误：{e}")
            return []

    def _derive_root_and_category_ids(
        self, category_id_list: list[list[str]]
    ) -> tuple[int | None, list[int]]:
        """从 categoryIdList 推导 root_category_id 与 category_id 列表（用于 filters 查询）"""
        if not category_id_list:
            return None, []

        root_category_id = None
        category_ids: list[int] = []

        for item in category_id_list:
            if not item:
                continue
            # item 是 list[str]，如 ["1", "344", "377"]
            ids = [int(x) for x in item if str(x).isdigit()]
            if ids:
                root_category_id = root_category_id or ids[0]
                category_ids.append(ids[-1])  # 使用最后一个作为叶子节点

        # 去重
        category_ids = list(set(category_ids))

        return root_category_id, category_ids

    def _fetch_user_style(self, user_id: int) -> dict[str, Any]:
        """获取用户画像风格（对齐 n8n 拿到用选品户画像）"""
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
            logger.warning(f"[海外探款商品] 用户画像获取失败: {e}")
            return {}

    def _fetch_user_profile_tags(self, user_id: int) -> list[str]:
        """获取用户画像并解析标签（对齐 n8n 封装用户画像标签）"""
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
                    prompt_key=CozePromptHubKey.ABROAD_GOODS_USER_TAG_PARSE_PROMPT.value,
                    variables={"user_select_message": style_text},
                )
            )
            values = (result.values or "").strip()
            if not values:
                return []
            return [item.strip() for item in values.split(",") if item.strip()][:3]
        except Exception as e:
            logger.warning(f"[海外探款商品] 用户画像标签解析失败: {e}")
            fallback_source = extract_style(style_text)
            fallback = [item.strip() for item in fallback_source.split(",") if item.strip()]
            return fallback[:3]

    def _llm_param_parse_node(self, state: AbroadGoodsWorkflowState) -> dict[str, Any]:
        """LLM 参数解析节点 - 对齐 n8n 的 '行业品类及时间-API参数解析Agent' 节点

        对应 n8n 节点: "行业品类及时间-API参数解析Agent"

        功能:
        - 使用 LLM 解析用户查询，提取选品参数（价格、时间、类目等）
        - 优先使用类目向量检索结果（category_vector），回退到 category_paths
        - 映射设计元素标签（label）到筛选项

        返回:
            param_result: 解析后的选品参数
            site_name: 引用站点名称（可选）
            site_id: 引用站点 ID（可选）
        """
        req: WorkflowRequest = state["request"]
        selection_dict = state["selection_dict"]

        ref_helper = QueryReferenceHelper.from_request(req)
        format_query = ref_helper.replace_placeholders(text=req.user_query)
        industry_value = self._override_industry_by_query(req.industry, format_query)

        category_vector = selection_dict.get("category_vector", [])
        category_paths = selection_dict.get("category_paths", [])
        category_candidates, category_root_id = self._build_category_candidates(
            category_vector, category_paths, industry_value
        )
        category_data = json.dumps(category_candidates, ensure_ascii=False)

        common_param = {
            "user_query": format_query,
            "preferred_entity": req.preferred_entity,
            "industry": industry_value,
            "user_preferences": req.user_preferences,
            "abroad_type": req.abroad_type,
            "common_selections": json.dumps(
                selection_dict.get("common_selections", {}), ensure_ascii=False
            ),
            "current_date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
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
                logger.warning(f"[海外探款商品] {run_name} 解析失败: {exc}")
                return schema()

        category_result = run_parse(
            CozePromptHubKey.ABROAD_GOODS_CATEGORY_PARSE_PROMPT.value,
            AbroadGoodsCategoryParseResult,
            "类目解析",
            {
                "user_query": format_query,
                "category_data": category_data,
                "industry": industry_value,
            },
        )
        category_hint = json.dumps(
            {"category_id_list": category_result.category_id_list}, ensure_ascii=False
        )

        attribute_future = thread_pool.submit_with_context(
            lambda: run_parse(
                CozePromptHubKey.ABROAD_GOODS_ATTRIBUTE_PARSE_PROMPT.value,
                AbroadGoodsAttributeParseResult,
                "属性解析",
                {
                    **common_param,
                    "category_hint": category_hint,
                },
            )
        )
        brand_future = thread_pool.submit_with_context(
            lambda: run_parse(
                CozePromptHubKey.ABROAD_GOODS_BRAND_PARSE_PROMPT.value,
                AbroadGoodsBrandParseResult,
                "品牌解析",
                {"user_query": format_query},
            )
        )
        time_future = thread_pool.submit_with_context(
            lambda: run_parse(
                CozePromptHubKey.ABROAD_GOODS_TIME_PARSE_PROMPT.value,
                AbroadGoodsTimeParseResult,
                "时间解析",
                {"user_query": format_query, "current_date": common_param["current_date"]},
            )
        )
        numeric_future = thread_pool.submit_with_context(
            lambda: run_parse(
                CozePromptHubKey.ABROAD_GOODS_NUMERIC_PARSE_PROMPT.value,
                AbroadGoodsNumericParseResult,
                "数值解析",
                {"user_query": format_query},
            )
        )
        platform_route_future = thread_pool.submit_with_context(
            lambda: run_parse(
                CozePromptHubKey.ABROAD_GOODS_PLATFORM_ROUTE_PARSE_PROMPT.value,
                AbroadGoodsPlatformRouteParseResult,
                "平台路由解析",
                {
                    "user_query": format_query,
                    "user_preferences": req.user_preferences,
                    "preferred_entity": req.preferred_entity,
                    "abroad_type": req.abroad_type,
                },
            )
        )
        region_future = thread_pool.submit_with_context(
            lambda: run_parse(
                CozePromptHubKey.ABROAD_GOODS_REGION_PARSE_PROMPT.value,
                AbroadGoodsRegionParseResult,
                "地域解析",
                {
                    "user_query": format_query,
                    "common_selections": common_param["common_selections"],
                },
            )
        )

        attribute_result = attribute_future.result()
        brand_result = brand_future.result()
        time_result = time_future.result()
        numeric_result = numeric_future.result()
        platform_route_result = platform_route_future.result()
        region_result = region_future.result()

        covered_fields = {
            "categoryIdList": category_result.category_id_list,
            "label": getattr(attribute_result, "label", None),
            "style": getattr(attribute_result, "style", None),
            "color": getattr(attribute_result, "color", None),
            "brand": getattr(brand_result, "brand", None),
            "feature": getattr(attribute_result, "feature", None),
        }
        covered_fields = {k: v for k, v in covered_fields.items() if v not in (None, "", [], {})}
        text_title_result = run_parse(
            CozePromptHubKey.ABROAD_GOODS_TEXT_TITLE_PARSE_PROMPT.value,
            AbroadGoodsTextTitleParseResult,
            "文本标题解析",
            {
                "user_query": format_query,
                "covered_fields": json.dumps(covered_fields, ensure_ascii=False),
            },
        )

        param_result = self._merge_param_result(
            category_result=category_result,
            attribute_result=attribute_result,
            brand_result=brand_result,
            time_result=time_result,
            numeric_result=numeric_result,
            platform_route_result=platform_route_result,
            region_result=region_result,
            text_title_result=text_title_result,
            selection_dict=selection_dict,
            req=req,
            industry_value=industry_value,
            category_root_id=category_root_id,
        )

        logger.debug(f"[海外探款商品] 参数解析结果: {param_result}")

        # 将站点信息传递到 state（后续 API 调用时使用）
        return {
            "param_result": param_result,
            "site_name": selection_dict.get("site_name"),
            "site_id": selection_dict.get("site_id"),
            "shop_id": selection_dict.get("shop_id"),
            "shop_zone_type": selection_dict.get("shop_zone_type"),
            "platform_zone_type": selection_dict.get("platform_zone_type"),
            "platform_type_override": selection_dict.get("platform_type_override"),
            "platform_name_override": selection_dict.get("platform_name_override"),
        }

    def _merge_param_result(
        self,
        *,
        category_result: AbroadGoodsCategoryParseResult,
        attribute_result: AbroadGoodsAttributeParseResult,
        brand_result: AbroadGoodsBrandParseResult,
        time_result: AbroadGoodsTimeParseResult,
        numeric_result: AbroadGoodsNumericParseResult,
        platform_route_result: AbroadGoodsPlatformRouteParseResult,
        region_result: AbroadGoodsRegionParseResult,
        text_title_result: AbroadGoodsTextTitleParseResult,
        selection_dict: dict[str, Any],
        req: WorkflowRequest,
        industry_value: str,
        category_root_id: int | None,
    ) -> AbroadGoodsParseParam:
        """合并子解析结果为主参数"""

        def normalize_text(value: Any) -> str | None:
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        def clamp_value(value: int | None, min_value: int, max_value: int) -> int | None:
            if value is None:
                return None
            if value < min_value:
                return min_value
            if value > max_value:
                return max_value
            return value

        limit = numeric_result.limit if numeric_result.limit is not None else 6000
        if limit > 6000:
            limit = 6000

        min_sprice = clamp_value(numeric_result.min_sprice, 0, 999999)
        max_sprice = clamp_value(numeric_result.max_sprice, 0, 999999)
        min_sale_volume = clamp_value(numeric_result.min_sale_volume_total, 0, 999999)
        max_sale_volume = clamp_value(numeric_result.max_sale_volume_total, 0, 999999)

        param_result = AbroadGoodsParseParam(
            min_sale_volume_total=min_sale_volume,
            max_sale_volume_total=max_sale_volume,
            min_sprice=min_sprice,
            max_sprice=max_sprice,
            put_on_sale_start_date=time_result.put_on_sale_start_date or "",
            put_on_sale_end_date=time_result.put_on_sale_end_date or "",
            start_date=time_result.start_date,
            end_date=time_result.end_date,
            category_id_list=category_result.category_id_list,
            region_ids=normalize_text(region_result.region_ids),
            country_list=normalize_text(region_result.country_list),
            platform=normalize_text(platform_route_result.platform),
            platform_type_list=normalize_text(platform_route_result.platform_type_list),
            feature=normalize_text(attribute_result.feature),
            style=normalize_text(attribute_result.style),
            label=normalize_text(attribute_result.label),
            color=normalize_text(attribute_result.color),
            brand=normalize_text(brand_result.brand),
            body_type=attribute_result.body_type,
            type=normalize_text(platform_route_result.type),
            new_type=normalize_text(platform_route_result.new_type) or "",
            text=normalize_text(text_title_result.text),
            zone_type=normalize_text(platform_route_result.zone_type),
            on_sale_flag=time_result.on_sale_flag,
            flag=platform_route_result.flag if platform_route_result.flag is not None else 2,
            user_data=platform_route_result.user_data if platform_route_result.user_data is not None else 0,
            sort_field=normalize_text(text_title_result.sort_field) or "默认",
            limit=limit,
            title=normalize_text(text_title_result.title),
        )

        param_result.label = self._map_design_detail_label(
            getattr(param_result, "label", None),
            selection_dict.get("common_selections", {}) or {},
        )

        # 根据 zone_type 修正 type 字段（对齐 n8n 逻辑）
        # n8n 逻辑：当 zoneType 是 amazon/temu 时，type 应该与专区类型一致
        zone_type = getattr(param_result, "zone_type", None)
        shop_zone_type = selection_dict.get("shop_zone_type")
        if shop_zone_type in ("amazon", "temu"):
            if shop_zone_type != zone_type:
                param_result.zone_type = shop_zone_type
                zone_type = shop_zone_type
                logger.debug(f"[海外探款] 店铺引用覆盖zone_type: {shop_zone_type}")
        else:
            platform_zone_type = selection_dict.get("platform_zone_type")
            if platform_zone_type in ("amazon", "temu") and platform_zone_type != zone_type:
                param_result.zone_type = platform_zone_type
                zone_type = platform_zone_type
                logger.debug(f"[海外探款] 平台引用覆盖zone_type: {platform_zone_type}")
        if zone_type not in ("amazon", "temu"):
            inferred_zone = (
                self._infer_zone_type(getattr(param_result, "type", None))
                or self._infer_zone_type(req.abroad_type)
            )
            if inferred_zone and inferred_zone != zone_type:
                param_result.zone_type = inferred_zone
                zone_type = inferred_zone
                logger.debug(f"[海外探款] 使用abroad_type推断zone_type: {inferred_zone}")
        if zone_type == "amazon":
            param_result.type = "亚马逊"
        elif zone_type == "temu":
            param_result.type = "Temu"
        # 如果 zone_type 为空或其他值，保持 LLM 解析的 type 值（可能是"独立站"或"shein"）

        # 如果有店铺引用，清除被误解析为品牌的店铺名
        shop_name = selection_dict.get("shop_name")
        shop_id = selection_dict.get("shop_id")
        if shop_id and shop_name:
            # 如果 LLM 解析的 brand 与店铺名相同，清除 brand（避免误过滤）
            if param_result.brand and param_result.brand.lower() == shop_name.lower():
                logger.debug(f"[海外探款] 清除被误解析为品牌的店铺名: {param_result.brand}")
                param_result.brand = None
            # 同时清除 platform 字段中的店铺名
            if param_result.platform and param_result.platform.lower() == shop_name.lower():
                logger.debug(f"[海外探款] 清除被误解析为平台的店铺名: {param_result.platform}")
                param_result.platform = None
        if shop_id and zone_type in ("amazon", "temu"):
            if getattr(param_result, "flag", None) != 2:
                logger.debug(f"[海外探款] 店铺查询强制走商品库(flag=2)")
            param_result.flag = 2

        # 如果有品牌引用但 LLM 没有解析出品牌，强制设置品牌
        brand_name = selection_dict.get("brand_name")
        if brand_name and not param_result.brand:
            param_result.brand = brand_name
            logger.debug(f"[海外探款] 强制设置品牌参数: {brand_name}")

        if category_root_id is None:
            category_root_id = self._extract_category_root_id(
                industry_value, selection_dict.get("category_paths", [])
            )
        prefix_set = self._build_category_prefix_set(
            selection_dict.get("category_paths", []), category_root_id
        )
        if not prefix_set and category_root_id:
            prefix_set = self._build_category_prefix_set(
                selection_dict.get("category_paths", []), None
            )
        normalized_categories = self._normalize_category_id_list(
            getattr(param_result, "category_id_list", []) or [], prefix_set
        )
        if normalized_categories != getattr(param_result, "category_id_list", []):
            logger.debug(f"[海外探款] 类目路径校验完成: {normalized_categories}")
        param_result.category_id_list = normalized_categories

        return param_result

    def _llm_param_merge_node(self, state: AbroadGoodsWorkflowState) -> dict[str, Any]:
        """LLM 合并节点：修正冲突/过窄条件"""
        req: WorkflowRequest = state["request"]
        param_result: AbroadGoodsParseParam = state["param_result"]

        ref_helper = QueryReferenceHelper.from_request(req)
        format_query = ref_helper.replace_placeholders(text=req.user_query)
        industry_value = self._override_industry_by_query(req.industry, format_query)

        prompt_param = {
            "user_query": format_query,
            "user_preferences": req.user_preferences or "",
            "industry": industry_value or "",
            "abroad_type": req.abroad_type or "",
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "category_id_list": json.dumps(param_result.category_id_list or [], ensure_ascii=False),
            "feature": param_result.feature or "",
            "style": param_result.style or "",
            "color": param_result.color or "",
            "label": param_result.label or "",
            "brand": param_result.brand or "",
            "text": param_result.text or "",
            "title": param_result.title or "",
            "platform": param_result.platform or "",
            "platform_type_list": param_result.platform_type_list or "",
            "region_ids": param_result.region_ids or "",
            "country_list": param_result.country_list or "",
            "type": param_result.type or "",
            "new_type": param_result.new_type or "",
            "sort_field": param_result.sort_field or "",
            "min_sprice": param_result.min_sprice,
            "max_sprice": param_result.max_sprice,
            "put_on_sale_start_date": param_result.put_on_sale_start_date or "",
            "put_on_sale_end_date": param_result.put_on_sale_end_date or "",
            "start_date": param_result.start_date or "",
            "end_date": param_result.end_date or "",
        }

        try:
            messages: list[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
                prompt_key=CozePromptHubKey.ABROAD_GOODS_PARAM_MERGE_PROMPT.value,
                variables=prompt_param,
            )
            llm: BaseChatModel = llm_factory.get_llm(
                LlmProvider.OPENROUTER.name,
                LlmModelName.OPENROUTER_GEMINI_3_FLASH_PREVIEW.value,
            )
            structured_llm = llm.with_structured_output(AbroadGoodsParamMergeResult).with_retry(
                stop_after_attempt=2
            )
            merge_result = structured_llm.with_config(run_name="参数合并").invoke(messages)
        except Exception as exc:
            logger.warning(f"[海外探款商品] 参数合并失败: {exc}")
            return {}

        updated = self._apply_param_merge_result(
            param_result=param_result,
            merge_result=merge_result,
        )
        return {"param_result": updated}

    @staticmethod
    def _apply_param_merge_result(
        *,
        param_result: AbroadGoodsParseParam,
        merge_result: AbroadGoodsParamMergeResult,
    ) -> AbroadGoodsParseParam:
        def normalize_text(value: Any) -> str | None:
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        def clamp_value(value: int | None, min_value: int, max_value: int) -> int | None:
            if value is None:
                return None
            if value < min_value:
                return min_value
            if value > max_value:
                return max_value
            return value

        updated = param_result.model_copy(deep=True)

        for field_name in (
            "put_on_sale_start_date",
            "put_on_sale_end_date",
            "start_date",
            "end_date",
            "sort_field",
            "text",
            "title",
        ):
            value = normalize_text(getattr(merge_result, field_name))
            if value is not None:
                setattr(updated, field_name, value)

        for field_name in (
            "min_sale_volume_total",
            "max_sale_volume_total",
            "min_sprice",
            "max_sprice",
            "on_sale_flag",
        ):
            value = getattr(merge_result, field_name)
            if value is None:
                continue
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    continue
                if value.lstrip("-").isdigit():
                    value = int(value)
                else:
                    continue
            if isinstance(value, float):
                value = int(value)
            if not isinstance(value, int):
                continue
            if field_name in {"min_sprice", "max_sprice"}:
                value = clamp_value(value, 0, 999999)
            if field_name in {"min_sale_volume_total", "max_sale_volume_total"}:
                value = clamp_value(value, 0, 999999)
            setattr(updated, field_name, value)

        if (
            updated.min_sprice is not None
            and updated.max_sprice is not None
            and updated.min_sprice > updated.max_sprice
        ):
            updated.min_sprice, updated.max_sprice = updated.max_sprice, updated.min_sprice

        if (
            updated.min_sale_volume_total is not None
            and updated.max_sale_volume_total is not None
            and updated.min_sale_volume_total > updated.max_sale_volume_total
        ):
            updated.min_sale_volume_total, updated.max_sale_volume_total = (
                updated.max_sale_volume_total,
                updated.min_sale_volume_total,
            )

        return updated

    def _parallel_parse_node(self, state: AbroadGoodsWorkflowState) -> dict[str, Any]:
        """并行解析节点 - 对齐 n8n 的并行执行逻辑

        对应 n8n 节点: 多个并行执行的节点（排序解析、品牌召回等）

        功能:
        - 并行执行排序类型解析、品牌召回等任务
        - 提高解析效率，与 n8n 的并行执行逻辑一致

        返回:
            sort_param_result: 排序类型解析结果
            brand_tags: 品牌标签列表
            style_list: 规范化风格列表
            user_filters: 用户画像筛选项
        """
        req = state["request"]
        param_result: AbroadGoodsParseParam = state["param_result"]
        style_list = self._normalize_style_list(getattr(param_result, "style", None))
        futures = {}
        # 排序解析
        futures["sort"] = thread_pool.submit_with_context(
            self._parse_sort, req.user_query, param_result
        )

        # 用户画像标签 (对应 n8n 的 "拿到用选品户画像" + "封装用户画像标签"，仅 user_data=1 时启用)
        if getattr(param_result, "user_data", 0) == 1:
            futures["user_tags"] = thread_pool.submit_with_context(
                self._fetch_user_profile_tags,
                req.user_id,
            )

        # 品牌召回
        brand = getattr(param_result, "brand", None) or ""
        if brand:
            brand_str = ",".join(brand) if isinstance(brand, list) else str(brand)
            futures["brand"] = thread_pool.submit_with_context(
                self._recall_and_clean_brand_tags, brand_str, req
            )

        # 收集结果
        sort_result = None
        brand_tags = []
        user_tags: list[str] = []
        user_filters: list[UserTagFilterItem] = []

        for key, future in futures.items():
            try:
                result = future.result(timeout=15)
                if key == "sort":
                    sort_result = result
                elif key == "user_tags":
                    user_tags = result or []
                elif key == "brand":
                    brand_tags = result or []
            except Exception as e:
                logger.warning(f"[海外探款商品] {key} 解析失败: {e}")

        if user_tags:
            user_filters = [
                UserTagFilterItem(name=tag, filter_type="user_tag", value=tag)
                for tag in user_tags
                if isinstance(tag, str) and tag.strip()
            ]

        return {
            "sort_param_result": sort_result,
            "user_filters": user_filters,
            "style_list": style_list,
            "brand_tags": brand_tags,  # 品牌召回标签
        }

    def _parse_sort(
        self, user_query: str, param_result: AbroadGoodsParseParam
    ) -> AbroadGoodsSortResult | None:
        """解析排序项（并行任务）"""
        try:
            prompt_param = {
                "user_query": user_query,
                "sort_field": param_result.sort_field or "默认",
                "query_type": param_result.type or "hot_sale",
            }

            messages = coze_loop_client_provider.get_langchain_messages(
                prompt_key=CozePromptHubKey.ABROAD_GOODS_SORT_TYPE_PARSE_PROMPT.value,
                variables=prompt_param,
            )

            llm = llm_factory.get_llm(
                LlmProvider.OPENROUTER.name, LlmModelName.OPENROUTER_GEMINI_2_5_FLASH.value
            )
            structured_llm = llm.with_structured_output(AbroadGoodsSortResult).with_retry(
                stop_after_attempt=2
            )
            return structured_llm.with_config(run_name="排序项解析").invoke(messages)
        except Exception as e:
            logger.warning(f"[海外探款商品] 排序解析失败: {e}")
            return None

    def _track_llm_parameters(
        self,
        req: WorkflowRequest,
        param_result: AbroadGoodsParseParam,
        sort_result: AbroadGoodsSortResult | None,
        user_filters: list[UserTagFilterItem],
        brand_tags: list[str],
        platform_types: list[Any],
    ) -> None:
        """记录 LLM 参数解析结果（对齐 n8n 埋点1）"""
        if not param_result:
            return

        param_payload = param_result.model_dump(by_alias=True, exclude_none=False)
        if "userData" in param_payload:
            param_payload["user_data"] = param_payload.pop("userData")
        param_payload["userid"] = req.user_id
        param_payload["teamid"] = req.team_id

        sort_field = 1
        sort_name = "最新上架"
        if sort_result and sort_result.sort_type_final:
            sort_field = sort_result.sort_type_final
            sort_name = sort_result.sort_type_final_name or sort_name

        sort_payload = {"output": {"sortField_new": sort_field, "sortField_new_name": sort_name}}

        brand_list = []
        for tag in brand_tags or []:
            tag_str = str(tag).strip()
            if tag_str:
                brand_list.append({"brand": tag_str})
        brand_payload = {"brandList": brand_list}

        filter_payload_list = []
        tag_values: list[str] = []
        for item in user_filters or []:
            if isinstance(item, UserTagFilterItem):
                payload = item.model_dump(exclude_none=True)
                if payload:
                    filter_payload_list.append(payload)
                value = item.value or item.name
            elif isinstance(item, dict):
                payload = {k: v for k, v in item.items() if v is not None}
                if payload:
                    filter_payload_list.append(payload)
                value = item.get("value") or item.get("name")
            else:
                value = None
            if value:
                tag_values.append(str(value).strip())

        user_tag_payload = {"output": {"values": ",".join([v for v in tag_values if v])}}
        filters_payload = {"output": {"user_tags": filter_payload_list}}

        platform_list = []
        for item in platform_types or []:
            if item is None:
                continue
            item_str = str(item).strip()
            if item_str:
                platform_list.append(item_str)
        platform_payload = {"output": {"platfotmType": platform_list}}

        llm_parameters = json.dumps(
            [
                brand_payload,
                {"output": param_payload},
                sort_payload,
                user_tag_payload,
                filters_payload,
                platform_payload,
            ],
            ensure_ascii=False,
        )

        def insert_track() -> None:
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

        thread_pool.submit_with_context(insert_track)

    def _match_sites_node(self, state: AbroadGoodsWorkflowState) -> dict[str, Any]:
        """站点匹配节点 - 对齐 n8n 的站点检索逻辑

        对应 n8n 节点: "Call '海外探款已上线站点检索'" + 站点数量判断

        功能:
        - 构建 tag_text: type_platform_countryList
        - 命中站点引用且 ID 为数字时，直接匹配单站点
        - 否则调用独立站子工作流进行站点匹配
        - 处理平台信息，判断是否为单选站点

        返回:
            tag_text: 站点匹配检索文本
            match_tags: 匹配到的站点列表
            platform_types: 平台类型列表
            is_single_platform: 是否为单选站点
            site_count: 站点数量
            has_sites: 是否有站点结果
        """
        param_result: Any = state.get("param_result")
        site_ref_present = bool(state.get("site_id") or state.get("site_name"))
        brand_value = getattr(param_result, "brand", None) if param_result else None
        has_brand = False
        if isinstance(brand_value, list):
            has_brand = any(str(item).strip() for item in brand_value)
        elif brand_value is not None:
            has_brand = bool(str(brand_value).strip())
        if has_brand and not site_ref_present:
            logger.debug("[海外探款商品] 品牌查询且无站点引用，跳过站点匹配")
            return {
                "tag_text": "",
                "match_tags": [],
                "platform_types": [],
                "is_single_platform": False,
                "site_count": 0,
                "has_sites": False,
            }

        # 从 param_result 中提取站点相关信息
        # 对应 n8n 的"上游数据1"节点逻辑
        platform: str = str(getattr(param_result, "platform", "") or "")
        if has_brand and not site_ref_present and platform:
            logger.debug("[海外探款商品] 品牌查询且无站点引用，忽略platform字段用于站点匹配")
            platform = ""
        country_list: Any = getattr(param_result, "country_list", []) or []

        # 构建站点匹配请求参数
        if country_list and isinstance(country_list, str):
            country_list_str: str = country_list
        elif isinstance(country_list, list):
            country_list_str = ",".join(country_list) if country_list else ""
        else:
            country_list_str = ""

        # 构建 tag_text: type_platform_countryList
        # 对应 n8n 中的逻辑
        type_value: str = str(getattr(param_result, "type", "") or "独立站")
        tag_text: str = f"{type_value}_{platform}_{country_list_str}"

        # 调用站点检索 API - 传递 user_query 和 param_result 用于 LLM 清洗
        req: Any = state["request"]
        subgraph = get_independent_sites_subgraph()
        site_id = state.get("site_id")
        site_name = state.get("site_name")
        if site_id:
            site_id_str = str(site_id).strip()
            if site_id_str.isdigit():
                logger.debug(
                    f"[海外探款商品] 使用站点引用强制匹配: {site_name} (ID: {site_id_str})"
                )
                match_tags = [f"{site_id_str},{site_name or ''}"]
                process_result = subgraph._process_platforms_node({"match_tags": match_tags})
                return {
                    "tag_text": tag_text,
                    "match_tags": match_tags,
                    "platform_types": process_result.get("platform_types", []),
                    "is_single_platform": process_result.get("is_single_site", False),
                    "site_count": process_result.get("site_count", 0),
                    "has_sites": process_result.get("has_sites", False),
                }
            logger.warning(
                f"[海外探款商品] 站点引用ID非数字，忽略强制匹配: {site_name} (ID: {site_id})"
            )
        try:
            match_state: dict[str, Any] = {
                "tag_text": tag_text,
                "user_query": req.user_query,
                "param_result": param_result,
            }
            match_result: dict[str, Any] = subgraph._match_sites_node(match_state)
            match_tags: list[Any] = match_result.get("match_tags", [])
        except Exception as e:
            logger.warning(f"[海外探款商品] 站点匹配失败: {e}")
            match_tags = []

        # 处理平台信息
        process_result: dict[str, Any] = subgraph._process_platforms_node({"match_tags": match_tags})
        platform_types: list[Any] = process_result.get("platform_types", [])
        is_single_platform: bool = process_result.get("is_single_site", False)

        return {
            "tag_text": tag_text,
            "match_tags": match_tags,
            "platform_types": platform_types,
            "is_single_platform": is_single_platform,
            "site_count": process_result.get("site_count", 0),
            "has_sites": process_result.get("has_sites", False),
        }

    def _site_detail_search_node(self, state: AbroadGoodsWorkflowState) -> dict[str, Any]:
        """站点详情搜索节点 - 对齐 n8n 的独立站子工作流调用

        对应 n8n 节点: 独立站选品子工作流调用

        功能:
        - 调用 IndependentSitesSubGraph 执行独立站商品搜索
        - 统一主/兜底结果，输出标准字段供后续节点使用

        返回:
            api_request: 统一后的请求体
            api_resp: 最终 API 响应
            result_count: 结果数量
            api_success: 是否成功
            browsed_count: 浏览数量
            goods_list: 商品列表
            request_path: 请求路径
            request_body: 请求体
            platform_name: 平台名称
            used_fallback: 是否使用兜底
            fallback_attempted: 是否尝试兜底
            fallback_result_count: 兜底结果数量
        """
        req = state["request"]
        param_result: AbroadGoodsParseParam = state["param_result"]
        sort_param_result: AbroadGoodsSortResult = state.get("sort_param_result")

        sort_type = sort_param_result.sort_type_final if sort_param_result else 1

        subgraph_input = {
            "user_id": req.user_id,
            "team_id": req.team_id,
            "user_query": req.user_query,
            "param_result": param_result,
            "sort_type": sort_type,
            "title": getattr(req, "title", ""),
            "tag_text": state.get("tag_text"),
            "match_tags": state.get("match_tags"),
            "brand_tags": state.get("brand_tags"),
        }

        subgraph = get_independent_sites_subgraph()
        try:
            result = subgraph.run(subgraph_input)
            primary_count = result.get("primary_result_count", 0) or 0
            secondary_count = result.get("secondary_result_count", 0) or 0
            fallback_attempted = result.get("fallback_attempted", False) or ("secondary_params" in result)
            used_fallback = result.get("used_fallback", False) or fallback_attempted

            total_count = result.get("final_result_count", 0) or 0
            final_resp = result.get("final_api_resp")
            final_success = result.get("final_success", False)

            request_path = result.get("request_path") or result.get(
                "secondary_request_path" if used_fallback else "primary_request_path"
            )
            request_params = result.get("request_params") or result.get(
                "secondary_params" if used_fallback else "primary_params"
            )
            request_body = (
                json.dumps(request_params, ensure_ascii=False)
                if request_params is not None
                else None
            )
            style_list = state.get("style_list")
            brand_tags = state.get("brand_tags")
            platform_override = state.get("platform_types") or None
            api_request = self._build_goods_request(
                param_result,
                sort_type=sort_type,
                platform_types=platform_override,
                style_list=style_list,
                brand_tags=brand_tags,
            )
            api_resp = final_resp
            api_success = final_success
            browsed_count = primary_count + secondary_count if fallback_attempted else primary_count

            return {
                "api_request": api_request,
                "api_resp": api_resp,
                "result_count": total_count,
                "api_success": api_success,
                "browsed_count": browsed_count,
                "goods_list": result.get("goods_list", []),
                "request_path": request_path,
                "request_body": request_body,
                "platform_name": "独立站",
                "used_fallback": used_fallback,
                "fallback_attempted": fallback_attempted,
                "fallback_result_count": secondary_count if fallback_attempted else 0,
            }
        except Exception as e:
            logger.error(f"[海外探款商品-独立站] SubGraph 调用失败: {e}")
            return {
                "api_request": None,
                "api_resp": None,
                "result_count": 0,
                "api_success": False,
                "browsed_count": 0,
                "goods_list": [],
            }

    def _goods_center_search_node(self, state: AbroadGoodsWorkflowState) -> dict[str, Any]:
        """全网数据搜索节点 - 对齐 n8n 的 'goods-center/goods-zone-list' API 调用

        对应 n8n 节点: "goods-center/goods-zone-list" API 调用

        功能:
        - 调用全网商品搜索API（独立站数据）
        - 使用完整的筛选参数（价格、类目、时间等）
        - 如果结果不足，会触发兜底逻辑

        API 路径: goods-center/goods-zone-list
        """
        req = state["request"]
        param_result: AbroadGoodsParseParam = state["param_result"]
        sort_param_result: AbroadGoodsSortResult = state.get("sort_param_result")
        platform_types = state.get("platform_types")
        style_list = state.get("style_list")
        brand_tags = state.get("brand_tags")
        platform_override = platform_types or None

        sort_type = sort_param_result.sort_type_final if sort_param_result else 1
        api_request = self._build_goods_request(
            param_result,
            sort_type=sort_type,
            platform_types=platform_override,
            style_list=style_list,
            brand_tags=brand_tags,
        )
        request_body = api_request.model_dump_json(by_alias=True, exclude_none=True)

        api_client = get_abroad_api()
        try:
            page_result = api_client.search_goods(
                user_id=req.user_id,
                team_id=req.team_id,
                params=api_request,
            )
            result_count = page_result.result_count
            if result_count is None:
                result_count = len(page_result.result_list)
            return {
                "api_request": api_request,
                "api_resp": page_result,
                "result_count": result_count,
                "api_success": True,
                "browsed_count": result_count,
                "request_path": "goods-center/goods-zone-list",
                "request_body": request_body,
                "platform_name": "独立站",
            }
        except Exception as e:
            logger.error(f"[海外探款商品] 全网数据搜索失败: {e}")
            return {
                "api_request": api_request,
                "api_resp": None,
                "result_count": 0,
                "api_success": False,
                "browsed_count": 0,
            }

    def _monitor_site_new_node(self, state: AbroadGoodsWorkflowState) -> dict[str, Any]:
        """监控站点 - 上新 - 对齐 n8n 的监控站点新品逻辑

        对应 n8n 节点: 监控站点新品相关节点

        功能:
        - 调用监控站点新品 API
        - 使用完整的筛选参数
        - 如果结果不足，会触发兜底逻辑

        API 路径: goods-list/monitor-site-new-list
        """
        req = state["request"]
        param_result: AbroadGoodsParseParam = state["param_result"]
        sort_param_result: AbroadGoodsSortResult = state.get("sort_param_result")
        platform_types = state.get("platform_types")
        style_list = state.get("style_list")
        brand_tags = state.get("brand_tags")
        platform_override = platform_types or None

        sort_type = sort_param_result.sort_type_final if sort_param_result else 1
        api_request = self._build_goods_request(
            param_result,
            sort_type=sort_type,
            platform_types=platform_override,
            style_list=style_list,
            brand_tags=brand_tags,
        )
        request_body = api_request.model_dump_json(by_alias=True, exclude_none=True)

        api_client = get_abroad_api()
        try:
            page_result = api_client.monitor_site_new(
                user_id=req.user_id,
                team_id=req.team_id,
                params=api_request,
            )
            result_count = page_result.result_count
            if result_count is None:
                result_count = len(page_result.result_list)
            return {
                "api_request": api_request,
                "api_resp": page_result,
                "result_count": result_count,
                "api_success": True,
                "browsed_count": result_count,
                "request_path": "goods-list/monitor-site-new-list",
                "request_body": request_body,
                "platform_name": "独立站",
            }
        except Exception as e:
            logger.error(f"[海外探款商品] 监控上新失败: {e}")
            return {
                "api_request": api_request,
                "api_resp": None,
                "result_count": 0,
                "api_success": False,
                "browsed_count": 0,
            }

    def _monitor_site_hot_node(self, state: AbroadGoodsWorkflowState) -> dict[str, Any]:
        """监控站点 - 热销 - 对齐 n8n 的监控站点热销逻辑

        对应 n8n 节点: 监控站点热销相关节点

        功能:
        - 调用监控站点热销 API
        - 使用完整的筛选参数
        - 如果结果不足，会触发兜底逻辑

        API 路径: goods-list/monitor-site-hot-list
        """
        req = state["request"]
        param_result: AbroadGoodsParseParam = state["param_result"]
        sort_param_result: AbroadGoodsSortResult = state.get("sort_param_result")
        platform_types = state.get("platform_types")
        style_list = state.get("style_list")
        brand_tags = state.get("brand_tags")
        platform_override = platform_types or None

        sort_type = sort_param_result.sort_type_final if sort_param_result else 1
        api_request = self._build_goods_request(
            param_result,
            sort_type=sort_type,
            platform_types=platform_override,
            style_list=style_list,
            brand_tags=brand_tags,
        )
        request_body = api_request.model_dump_json(by_alias=True, exclude_none=True)

        api_client = get_abroad_api()
        try:
            page_result = api_client.monitor_site_hot(
                user_id=req.user_id,
                team_id=req.team_id,
                params=api_request,
            )
            result_count = page_result.result_count
            if result_count is None:
                result_count = len(page_result.result_list)
            return {
                "api_request": api_request,
                "api_resp": page_result,
                "result_count": result_count,
                "api_success": True,
                "browsed_count": result_count,
                "request_path": "goods-list/monitor-site-hot-list",
                "request_body": request_body,
                "platform_name": "独立站",
            }
        except Exception as e:
            logger.error(f"[海外探款商品] 监控热销失败: {e}")
            return {
                "api_request": api_request,
                "api_resp": None,
                "result_count": 0,
                "api_success": False,
                "browsed_count": 0,
            }

    def _fallback_goods_center_node(self, state: AbroadGoodsWorkflowState) -> dict[str, Any]:
        """全网数据兜底节点 - 对齐 n8n 的兜底逻辑

        对应 n8n 节点: 全网数据搜索的兜底处理

        功能:
        - 当主 API 结果不足时触发
        - 使用简化后的筛选条件（移除复杂筛选参数）

        API 路径: goods-center/goods-zone-list（与主 API 相同，但参数简化）
        """
        req = state["request"]
        origin_request: AbroadGoodsSearchRequest = state.get("api_request")

        if not origin_request:
            logger.error("[海外探款商品] 全网数据兜底失败: 主API请求为空")
            return {
                "fallback_api_resp": None,
                "fallback_result_count": 0,
                "fallback_api_success": False,
            }

        platform_name = state.get("platform_name", "海外探款")
        self._get_pusher(req=req).complete_phase(phase_name="商品筛选中", variables={
            "datasource": "海外商品数据库",
            "platform": platform_name,
            "browsed_count": f"{str(random.randint(100000, 1000000))}",
            "filter_result_text": "当前筛选出的商品数量不足，不满足列表需求，我将再次搜索确保无遗漏"
        })

        # 判断是否为品牌查询：如果原请求有brand参数，则保留brand过滤
        has_brand = origin_request.brand and len(origin_request.brand) > 0
        simple_request = origin_request.to_simplified(keep_brand=has_brand)
        
        # 兜底时检查sortType：独立站商品可能没有销量数据，sortType=8（热销排序）会返回0结果
        # 兜底策略：降级到最新上架排序（sortType=1），确保有数据返回
        if simple_request.sort_type == 8:
            logger.info(
                f"[海外探款商品] 兜底模式: sortType从{simple_request.sort_type}(热销)降级为1(最新上架)"
                + (f", 保留brand过滤={simple_request.brand}" if has_brand else "")
            )
            simple_request = simple_request.model_copy(update={"sort_type": 1})
        
        api_client = get_abroad_api()
        try:
            page_result = api_client.search_goods(
                user_id=req.user_id,
                team_id=req.team_id,
                params=simple_request,
            )
            result_count = page_result.result_count
            if result_count is None:
                result_count = len(page_result.result_list)
            return {
                "fallback_api_request": simple_request,
                "fallback_api_resp": page_result,
                "fallback_result_count": result_count,
                "fallback_api_success": True,
                "fallback_request_path": "goods-center/goods-zone-list",
                "fallback_request_body": simple_request.model_dump_json(
                    by_alias=True, exclude_none=True
                ),
            }
        except Exception as e:
            logger.error(f"[海外探款商品] 全网兜底失败: {e}")
            return {
                "fallback_api_request": simple_request,
                "fallback_api_resp": None,
                "fallback_result_count": 0,
                "fallback_api_success": False,
            }

    def _fallback_monitor_new_node(self, state: AbroadGoodsWorkflowState) -> dict[str, Any]:
        """监控上新兜底节点 - 对齐 n8n 的兜底逻辑

        对应 n8n 节点: 监控站点新品的兜底处理

        功能:
        - 当主 API 结果不足时触发
        - 使用简化后的筛选条件（移除复杂筛选参数）

        API 路径: goods-list/monitor-site-new-list（与主 API 相同，但参数简化）
        """
        req = state["request"]
        origin_request: AbroadGoodsSearchRequest = state.get("api_request")

        if not origin_request:
            logger.error("[海外探款商品] 监控上新兜底失败: 主API请求为空")
            return {
                "fallback_api_resp": None,
                "fallback_result_count": 0,
                "fallback_api_success": False,
            }

        platform_name = state.get("platform_name", "海外探款")
        self._get_pusher(req=req).complete_phase(phase_name="商品筛选中", variables={
            "datasource": "海外商品数据库",
            "platform": platform_name,
            "browsed_count": f"{str(random.randint(100000, 1000000))}",
            "filter_result_text": "当前筛选出的商品数量不足，不满足列表需求，我将再次搜索确保无遗漏"
        })

        # 判断是否为品牌查询：如果原请求有brand参数，则保留brand过滤
        has_brand = origin_request.brand and len(origin_request.brand) > 0
        simple_request = origin_request.to_simplified(keep_brand=has_brand)
        
        # 兜底时检查sortType：独立站商品可能没有销量数据，sortType=8（热销排序）会返回0结果
        if simple_request.sort_type == 8:
            logger.info(
                f"[海外探款商品-监控新品] 兜底模式: sortType从{simple_request.sort_type}(热销)降级为1(最新上架)"
                + (f", 保留brand过滤={simple_request.brand}" if has_brand else "")
            )
            simple_request = simple_request.model_copy(update={"sort_type": 1})
        
        api_client = get_abroad_api()
        try:
            page_result = api_client.monitor_site_new(
                user_id=req.user_id,
                team_id=req.team_id,
                params=simple_request,
            )
            result_count = page_result.result_count
            if result_count is None:
                result_count = len(page_result.result_list)
            return {
                "fallback_api_request": simple_request,
                "fallback_api_resp": page_result,
                "fallback_result_count": result_count,
                "fallback_api_success": True,
                "fallback_request_path": "goods-list/monitor-site-new-list",
                "fallback_request_body": simple_request.model_dump_json(
                    by_alias=True, exclude_none=True
                ),
            }
        except Exception as e:
            logger.error(f"[海外探款商品] 监控上新兜底失败: {e}")
            return {
                "fallback_api_request": simple_request,
                "fallback_api_resp": None,
                "fallback_result_count": 0,
                "fallback_api_success": False,
            }

    def _fallback_monitor_hot_node(self, state: AbroadGoodsWorkflowState) -> dict[str, Any]:
        """监控热销兜底节点 - 对齐 n8n 的兜底逻辑

        对应 n8n 节点: 监控站点热销的兜底处理

        功能:
        - 当主 API 结果不足时触发
        - 使用简化后的筛选条件（移除复杂筛选参数）

        API 路径: goods-list/monitor-site-hot-list（与主 API 相同，但参数简化）
        """
        req = state["request"]
        origin_request: AbroadGoodsSearchRequest = state.get("api_request")

        if not origin_request:
            logger.error("[海外探款商品] 监控热销兜底失败: 主API请求为空")
            return {
                "fallback_api_resp": None,
                "fallback_result_count": 0,
                "fallback_api_success": False,
            }

        platform_name = state.get("platform_name", "海外探款")
        self._get_pusher(req=req).complete_phase(phase_name="商品筛选中", variables={
            "datasource": "海外商品数据库",
            "platform": platform_name,
            "browsed_count": f"{str(random.randint(100000, 1000000))}",
            "filter_result_text": "当前筛选出的商品数量不足，不满足列表需求，我将再次搜索确保无遗漏"
        })

        # 判断是否为品牌查询：如果原请求有brand参数，则保留brand过滤
        has_brand = origin_request.brand and len(origin_request.brand) > 0
        simple_request = origin_request.to_simplified(keep_brand=has_brand)
        
        # 兜底时检查sortType：独立站商品可能没有销量数据，sortType=8（热销排序）会返回0结果
        if simple_request.sort_type == 8:
            logger.info(
                f"[海外探款商品-监控热销] 兜底模式: sortType从{simple_request.sort_type}(热销)降级为1(最新上架)"
                + (f", 保留brand过滤={simple_request.brand}" if has_brand else "")
            )
            simple_request = simple_request.model_copy(update={"sort_type": 1})
        
        api_client = get_abroad_api()
        try:
            page_result = api_client.monitor_site_hot(
                user_id=req.user_id,
                team_id=req.team_id,
                params=simple_request,
            )
            result_count = page_result.result_count
            if result_count is None:
                result_count = len(page_result.result_list)
            return {
                "fallback_api_request": simple_request,
                "fallback_api_resp": page_result,
                "fallback_result_count": result_count,
                "fallback_api_success": True,
                "fallback_request_path": "goods-list/monitor-site-hot-list",
                "fallback_request_body": simple_request.model_dump_json(
                    by_alias=True, exclude_none=True
                ),
            }
        except Exception as e:
            logger.error(f"[海外探款商品] 监控热销兜底失败: {e}")
            return {
                "fallback_api_request": simple_request,
                "fallback_api_resp": None,
                "fallback_result_count": 0,
                "fallback_api_success": False,
            }

    def _amazon_search_goods_node(self, state: AbroadGoodsWorkflowState) -> dict[str, Any]:
        """Amazon 专区商品搜索 - 对齐 n8n 的 Amazon 分支逻辑

        对应 n8n 节点: Amazon 选品子工作流调用

        功能:
        - 调用 AmazonSubGraph 执行 Amazon 商品搜索
        - 统一主/兜底结果，输出标准字段供后续节点使用

        返回:
            api_request: 统一后的请求体
            api_resp: 最终 API 响应
            result_count: 结果数量
            api_success: 是否成功
            browsed_count: 浏览数量
            request_path: 请求路径
            request_body: 请求体
            platform_name: 平台名称
            used_fallback: 是否使用兜底
            fallback_attempted: 是否尝试兜底
            fallback_result_count: 兜底结果数量
        """
        req = state["request"]
        param_result: AbroadGoodsParseParam = state["param_result"]
        sort_param_result: AbroadGoodsSortResult = state.get("sort_param_result")

        # 获取排序类型
        sort_type = sort_param_result.sort_type_final if sort_param_result else 1

        # 构建 SubGraph 输入
        subgraph_input = {
            "user_id": req.user_id,
            "team_id": req.team_id,
            "user_query": req.user_query,
            "param_result": param_result,
            "sort_type": sort_type,
            "flag": getattr(param_result, "flag", 2),  # 1=监控台, 2=商品库
            "new_type": getattr(param_result, "new_type", ""),  # 新品/热销
            "title": getattr(param_result, "title", ""),
        }
        shop_id = state.get("shop_id")
        if shop_id:
            subgraph_input["shop_id"] = shop_id
        platform_type_override = state.get("platform_type_override")
        if platform_type_override is not None:
            subgraph_input["amazon_platform_type"] = platform_type_override
        platform_name_override = state.get("platform_name_override")
        if platform_name_override:
            subgraph_input["platform_name_override"] = platform_name_override

        logger.debug(
            f"[海外探款商品-Amazon] 调用 SubGraph, flag={subgraph_input['flag']}, new_type={subgraph_input['new_type']}"
        )

        # 调用 SubGraph
        subgraph = get_amazon_subgraph()
        result = subgraph.run(subgraph_input)

        # 提取结果
        result_count = result.get("result_count", 0)
        api_resp = result.get("api_resp")
        fallback_resp = result.get("fallback_api_resp")
        fallback_count = result.get("fallback_result_count", 0)
        primary_success = result.get("api_success", False)
        fallback_attempted = "fallback_params" in result

        used_fallback = fallback_attempted
        final_resp = fallback_resp if used_fallback else api_resp
        final_count = fallback_count if used_fallback else result_count
        final_success = bool(fallback_resp) if used_fallback else primary_success
        browsed_count = fallback_count if used_fallback else (result_count or 0)

        request_path = (
            result.get("request_path")
            or result.get("fallback_request_path" if used_fallback else "request_path")
            or "amazon/goods/list"
        )
        request_params = (
            result.get("request_params")
            or result.get("fallback_params" if used_fallback else "api_params")
        )
        request_body = (
            json.dumps(request_params, ensure_ascii=False) if request_params is not None else None
        )

        # 使用真实请求参数作为 api_request（与其他工作流/输出对齐）
        api_request = (
            request_params
            if request_params is not None
            else AbroadGoodsSearchRequest.from_parse_param(param_result, sort_type=sort_type)
        )

        return {
            "api_request": api_request,
            "api_resp": final_resp,
            "result_count": final_count,
            "api_success": final_success,
            "browsed_count": browsed_count,
            "request_path": request_path,
            "request_body": request_body,
            "platform_name": "亚马逊",
            "used_fallback": used_fallback,
            "fallback_attempted": fallback_attempted,
            "fallback_result_count": fallback_count,
        }

    def _temu_search_goods_node(self, state: AbroadGoodsWorkflowState) -> dict[str, Any]:
        """Temu 专区商品搜索 - 对齐 n8n 的 Temu 分支逻辑

        对应 n8n 节点: Temu 选品子工作流调用

        功能:
        - 调用 TemuSubGraph 执行 Temu 商品搜索
        - 统一主/兜底结果，输出标准字段供后续节点使用

        返回:
            api_request: 统一后的请求体
            api_resp: 最终 API 响应
            result_count: 结果数量
            api_success: 是否成功
            browsed_count: 浏览数量
            request_path: 请求路径
            request_body: 请求体
            platform_name: 平台名称
            used_fallback: 是否使用兜底
            fallback_attempted: 是否尝试兜底
            fallback_result_count: 兜底结果数量
        """
        req = state["request"]
        param_result: AbroadGoodsParseParam = state["param_result"]
        sort_param_result: AbroadGoodsSortResult = state.get("sort_param_result")

        # 获取排序类型（数字）
        sort_type = sort_param_result.sort_type_final if sort_param_result else 1

        # 构建子工作流输入状态
        subgraph_state = {
            "user_id": req.user_id,
            "team_id": req.team_id,
            "user_query": req.user_query,
            "param_result": param_result,
            "sort_type": sort_type,
            "flag": getattr(param_result, "flag", 2),  # 1=监控台, 2=商品库
            "new_type": getattr(param_result, "new_type", "热销"),  # "新品" 或 "热销"
            "title": getattr(req, "title", ""),
        }
        shop_id = state.get("shop_id")
        if shop_id:
            subgraph_state["shop_id"] = shop_id
        platform_type_override = state.get("platform_type_override")
        if platform_type_override is not None:
            subgraph_state["temu_platform_type"] = platform_type_override
        platform_name_override = state.get("platform_name_override")
        if platform_name_override:
            subgraph_state["platform_name_override"] = platform_name_override

        # 执行 Temu 子工作流
        subgraph = get_temu_subgraph()
        try:
            result = subgraph.run(subgraph_state)

            result_count = result.get("result_count", 0)
            api_resp = result.get("api_resp")
            fallback_resp = result.get("fallback_api_resp")
            fallback_count = result.get("fallback_result_count", 0)
            primary_success = result.get("api_success", False)
            fallback_attempted = "fallback_params" in result
            primary_has_results = (
                (result_count is not None and result_count > 0)
                or bool(api_resp and getattr(api_resp, "result_list", None))
            )
            used_fallback = (not primary_has_results) and fallback_resp is not None
            final_resp = api_resp if primary_has_results else fallback_resp
            final_count = result_count if primary_has_results else fallback_count
            final_success = primary_success if primary_has_results else bool(fallback_resp)
            browsed_count = (
                (result_count or 0) + (fallback_count or 0)
                if fallback_resp is not None
                else (result_count or 0)
            )
            request_path = (
                result.get("request_path")
                or result.get("fallback_request_path" if used_fallback else "request_path")
                or "temu/goods/list"
            )
            request_params = (
                result.get("request_params")
                or result.get("fallback_params" if used_fallback else "api_params")
            )
            request_body = (
                json.dumps(request_params, ensure_ascii=False)
                if request_params is not None
                else None
            )

            api_request = request_params if request_params is not None else param_result
            return {
                "api_request": api_request,
                "api_resp": final_resp,
                "result_count": final_count,
                "api_success": final_success,
                "browsed_count": browsed_count,
                "goods_list": result.get("goods_list", []),
                "request_path": request_path,
                "request_body": request_body,
                "platform_name": "Temu",
                "used_fallback": used_fallback,
                "fallback_attempted": fallback_attempted,
                "fallback_result_count": fallback_count,
            }
        except Exception as e:
            logger.error(f"[海外探款商品-Temu] 子工作流执行失败: {e}")
            return {
                "api_request": param_result,
                "api_resp": None,
                "result_count": 0,
                "api_success": False,
                "browsed_count": 0,
            }

    def _merge_search_result_node(self, state: AbroadGoodsWorkflowState) -> dict[str, Any]:
        """搜索结果汇聚节点 - 对齐 n8n 的结果合并逻辑

        对应 n8n 节点: 结果合并相关节点

        功能:
        - 统一处理主 API 和兜底 API 的结果
        - 判断是否使用了兜底
        - 累加浏览数量（主搜索 + 兜底搜索）
        - 返回最终的统一结果格式

        返回:
            merged_api_resp: 合并后的 API 响应
            merged_result_count: 合并后的结果数量
            browsed_count: 总浏览数量
            api_request: 最终请求参数（对象或字典）
            request_path: 最终请求路径
            request_body: 最终请求体 JSON
            used_fallback: 是否使用了兜底
        """
        api_resp = state.get("api_resp")
        result_count = state.get("result_count", 0)
        fallback_resp = state.get("fallback_api_resp")
        fallback_count = state.get("fallback_result_count", 0)
        api_request = state.get("api_request")
        request_path = state.get("request_path")
        request_body = state.get("request_body")
        fallback_request_path = state.get("fallback_request_path")
        fallback_request_body = state.get("fallback_request_body")

        # 判断是否使用了兜底 (fallback_resp 存在即意味着走了 fallback 分支)
        fallback_used_here = fallback_resp is not None
        used_fallback = fallback_used_here or state.get("used_fallback", False)

        if fallback_used_here:
            # 走了 fallback 路径，使用兜底结果
            final_resp = fallback_resp
            final_count = fallback_count
            # browsed_count 应该累加主搜索和兜底搜索的浏览数
            browsed_count = (result_count or 0) + (fallback_count or 0)
        else:
            # 没走 fallback，使用主搜索结果
            final_resp = api_resp
            final_count = result_count
            # 未走 fallback，使用已有浏览数（如子工作流汇总）或主搜索结果数
            existing_browsed = state.get("browsed_count")
            browsed_count = (
                existing_browsed if existing_browsed is not None else (result_count or 0)
            )

        final_request = state.get("fallback_api_request") if fallback_used_here else api_request
        resolved_request_path = fallback_request_path if fallback_used_here else request_path
        resolved_request_body = fallback_request_body if fallback_used_here else request_body
        if not resolved_request_body and final_request is not None:
            if hasattr(final_request, "model_dump_json"):
                resolved_request_body = final_request.model_dump_json(by_alias=True, exclude_none=True)
            elif isinstance(final_request, dict):
                resolved_request_body = json.dumps(final_request, ensure_ascii=False)

        return {
            "merged_api_resp": final_resp,
            "merged_result_count": final_count,
            "browsed_count": browsed_count,
            "api_request": final_request,
            "request_path": resolved_request_path,
            "request_body": resolved_request_body,
            "used_fallback": used_fallback,
        }

    def _post_process_node(self, state: AbroadGoodsWorkflowState) -> dict[str, Any]:
        """后处理节点 - 对齐 n8n 的后处理逻辑

        对应 n8n 节点: "Split Out" → "Edit Fields" → "Aggregate" → "商品属性接口"

        功能:
        - 处理商品列表数据
        - 调用商品属性接口获取标签信息
        - 格式化商品数据为统一格式

        返回:
            processed_goods_list: 处理后的商品列表
            goods_labels: 商品标签信息
        """
        req = state["request"]
        merged_api_resp = state.get("merged_api_resp")

        if not merged_api_resp or not merged_api_resp.result_list:
            return {
                "processed_goods_list": [],
                "goods_labels": [],
            }

        # 调试：检查返回商品的店铺信息
        expected_shop_id = state.get("shop_id")
        if expected_shop_id:
            shop_ids_in_result = set()
            for goods in merged_api_resp.result_list:
                if hasattr(goods, "shop_id") and goods.shop_id:
                    shop_ids_in_result.add(str(goods.shop_id))
            if shop_ids_in_result:
                logger.debug(f"[海外探款] 期望店铺ID: {expected_shop_id}, 返回商品的店铺ID: {shop_ids_in_result}")
                if len(shop_ids_in_result) > 1 or str(expected_shop_id) not in shop_ids_in_result:
                    logger.warning(f"[海外探款] 店铺筛选可能未生效! 期望: {expected_shop_id}, 实际: {shop_ids_in_result}")

        # 提取商品信息
        processed_list = []
        for goods in merged_api_resp.result_list:
            processed_list.append(
                {
                    "goods_id": goods.goods_id,
                    "goods_name": goods.goods_name,
                    "goods_img": goods.goods_img,
                    "sprice": goods.sprice,
                }
            )

        # 调用商品属性接口 (对齐 n8n 的 "商品属性接口" 节点)
        goods_labels = []
        if processed_list:
            api_client = get_abroad_api()
            goods_for_labels = [{"productId": g["goods_id"]} for g in processed_list]
            goods_labels = api_client.query_goods_labels(
                user_id=str(req.user_id),
                team_id=str(req.team_id),
                goods_list=goods_for_labels,
            )

        return {
            "processed_goods_list": processed_list,
            "goods_labels": goods_labels,
        }

    def _has_result_node(self, state: AbroadGoodsWorkflowState) -> dict[str, Any]:
        """有结果分支 - 对齐 n8n 的有结果处理逻辑

        对应 n8n 节点: 有结果时的处理分支

        功能:
        - 推送筛选进度消息
        - 推送数据源消息
        - 处理有结果的情况
        - 准备最终结果数据

        返回:
            has_query_result: 是否有查询结果
            entity_simple_data: 简化的实体数据
        """
        req = state["request"]
        api_request = state.get("api_request")
        param_result: AbroadGoodsParseParam = state["param_result"]
        merged_api_resp = state.get("merged_api_resp")
        user_filters = state.get("user_filters", [])
        fallback_attempted = state.get("fallback_attempted", False) or (
            state.get("fallback_api_request") is not None
        )
        fallback_count = state.get("fallback_result_count") or 0
        final_count = state.get("merged_result_count", 0) or 0
        browsed_count = state.get("browsed_count", 0) or 0
        platform_name = state.get("platform_name", "海外探款")

        # 推送筛选进度消息（对齐 abroad_ins）
        pusher = self._get_pusher(req=req)
        if fallback_attempted:
            fallback_has_results = fallback_count > 0
            if fallback_has_results and final_count > 0:
                retry_text = (
                    "商品筛选完成，当前筛选出的商品数量符合报表要求，接下来我将根据数据绘制商品列表"
                )
                browsed_count_display = str(random.randint(100000, 1000000))
            else:
                retry_text = (
                    f"当前筛选出的商品数量为{final_count}，我可能需要提醒用户调整筛选维度"
                )
                browsed_count_display = str(random.randint(100000, 1000000))
            # 走了兜底路径，推送二次筛选消息
            pusher.complete_phase(phase_name="二次筛选中", variables={
                "datasource": "海外商品数据库",
                "platform": platform_name,
                "browsed_count": browsed_count_display,
                "retry_result_text": retry_text,
            })
        else:
            # 首次查询成功，推送主筛选消息
            pusher.complete_phase(phase_name="商品筛选中", variables={
                "datasource": "海外商品数据库",
                "platform": platform_name,
                "browsed_count": f"{str(random.randint(100000, 1000000))}",
                "filter_result_text": f"商品筛选完成，当前筛选出的商品数量满足需求，无需二次筛选，接下来我将根据数据完成商品列表"
            })

        # 成功路径的过程消息
        pusher.complete_phase("生成列表中")
        pusher.complete_phase("选品完成")

        # 推送任务状态消息
        task_status_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="任务状态",
            status="RUNNING",
            content_type=WorkflowMessageContentType.TASK_STATUS.value,
            content=ParameterDataContent(
                data=ParameterData(task_status=1)
            ),
            create_ts=int(round(time.time() * 1000)),
        )
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            task_status_message.model_dump_json(),
        )

        request_path = state.get("request_path") or "goods-center/goods-zone-list"
        request_body = state.get("request_body")
        if request_body and not isinstance(request_body, str):
            request_body = json.dumps(request_body, ensure_ascii=False)
        if not request_body and api_request:
            if hasattr(api_request, "model_dump_json"):
                request_body = api_request.model_dump_json(by_alias=True, exclude_none=True)
            else:
                request_body = json.dumps(api_request, ensure_ascii=False)

        # 推送路径消息（对齐 n8n data-source）
        body_obj = None
        if request_body:
            if isinstance(request_body, str):
                try:
                    body_obj = json.loads(request_body)
                except Exception:
                    body_obj = None
            elif isinstance(request_body, dict):
                body_obj = request_body
        platform_type = body_obj.get("platformType") if isinstance(body_obj, dict) else None

        data_payload = None
        if request_path in {"external/for-zxy/site-goods-list", "goods-list/site-goods-list"}:
            data_payload = {"entity_type": 1, "content": "abroad-independentStation"}
        elif request_path == "goods-list/monitor-site-new-list":
            data_payload = {"entity_type": 1, "content": "aborad-MonitoringSite-new"}
        elif request_path == "goods-list/monitor-site-hot-list":
            data_payload = {"entity_type": 1, "content": "aborad-MonitoringSite-hot"}
        elif request_path == "goods-center/goods-zone-list":
            data_payload = {"entity_type": 1, "content": "abroad-item-all"}
        elif request_path == "amazon/goods/monitor-shop-new-list":
            data_payload = {"entity_type": 1, "content": "abroad-amazon-monitor-shop-new"}
        elif request_path == "amazon/goods/monitor-shop-hot-list":
            data_payload = {"entity_type": 1, "content": "abroad-amazon-monitor-shop-hot"}
        elif request_path == "amazon/goods/list":
            data_payload = {"entity_type": 1, "content": "abroad-amazon-item-all"}

        if data_payload and platform_type is not None and request_path in {
            "external/for-zxy/site-goods-list",
            "goods-list/site-goods-list",
            "amazon/goods/monitor-shop-new-list",
            "amazon/goods/monitor-shop-hot-list",
            "amazon/goods/list",
        }:
            data_payload["query_params"] = [str(platform_type)]

        if data_payload:
            path_message = BaseRedisMessage(
                session_id=req.session_id,
                reply_message_id=req.message_id,
                reply_id=f"reply_{req.message_id}",
                reply_seq=0,
                operate_id="输出参数",
                status="RUNNING",
                content_type=8,
                content=CustomDataContent(data=data_payload),
                create_ts=int(round(time.time() * 1000)),
            )
            redis_client.list_left_push(
                RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
                path_message.model_dump_json(),
            )

        # 推送参数消息
        if request_body:
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
                        request_body=request_body,
                        actions=["view", "export", "download"],
                        title=param_result.title,
                        entity_type=3,
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
        if merged_api_resp and merged_api_resp.result_list:
            entity_simple_info_list = [
                {
                    "商品id": e.goods_id,
                    "title": e.goods_name,
                    "siteName": e.platform,
                    "price": e.sprice,
                    "imageUrl": e.goods_img,
                }
                for e in merged_api_resp.result_list
            ]
        else:
            entity_simple_info_list = []

        return {
            "has_query_result": True,
            "entity_simple_data": entity_simple_info_list,
        }

    def _no_result_node(self, state: AbroadGoodsWorkflowState) -> dict[str, Any]:
        """无结果分支 - 对齐 n8n 的无结果处理逻辑

        对应 n8n 节点: 无结果时的处理分支

        功能:
        - 推送无结果消息
        - 处理无结果的情况
        - 提供用户友好的提示信息

        返回:
            has_query_result: False（无结果）
        """
        req = state["request"]
        fallback_attempted = state.get("fallback_attempted", False) or (
            state.get("fallback_api_request") is not None
        )
        platform_name = state.get("platform_name", "海外探款")

        # 推送二次筛选消息（对齐 abroad_ins）
        pusher = self._get_pusher(req=req)
        if fallback_attempted:
            pusher.complete_phase(phase_name="二次筛选中", variables={
                "datasource": "海外商品数据库",
                "platform": platform_name,
                "browsed_count": f"{str(random.randint(100000, 1000000))}",
                "retry_result_text": "当前筛选出的商品数量为0，我可能需要提醒用户调整筛选维度",
            })

        # 选品失败
        pusher.fail_phase(phase_name="生成列表失败", error_message="我未能完成列表绘制，原因是没有数据\n需要提醒用户调整筛选维度，才能更好的获取数据")
        pusher.fail_phase(phase_name="选品未完成", error_message=None)
        # 推送任务状态消息
        task_status_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="任务状态",
            status="RUNNING",
            content_type=WorkflowMessageContentType.TASK_STATUS.value,
            content=ParameterDataContent(
                data=ParameterData(task_status=0)
            ),
            create_ts=int(round(time.time() * 1000)),
        )
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            task_status_message.model_dump_json(),
        )

        # 推送无结果消息
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

        return {
            "has_query_result": False,
        }

    def _package_result_node(self, state: AbroadGoodsWorkflowState) -> dict[str, Any]:
        """封装返回结果节点 - 对齐 n8n 的结果封装逻辑

        对应 n8n 节点: 结果封装相关节点

        功能:
        - 封装最终的工作流响应
        - 推送路径消息（数据源信息）
        - 记录结束埋点
        - 构建 WorkflowResponse 对象

        返回:
            workflow_response: 最终的工作流响应对象
        """
        req = state["request"]
        param_result: AbroadGoodsParseParam | None = state.get("param_result")
        if param_result:
            sort_param_result: AbroadGoodsSortResult | None = state.get("sort_param_result")
            user_filters = state.get("user_filters") or []
            brand_tags = state.get("brand_tags") or []
            platform_types = state.get("platform_types") or []
            if not platform_types:
                platform_override = state.get("platform_type_override")
                if platform_override:
                    platform_types = (
                        platform_override
                        if isinstance(platform_override, list)
                        else [platform_override]
                    )
            self._track_llm_parameters(
                req,
                param_result,
                sort_param_result,
                user_filters,
                brand_tags,
                platform_types,
            )
        has_query_result = state.get("has_query_result", False)
        entity_data = state.get("entity_simple_data", [])
        result_count = state.get("merged_result_count", 0)
        request_path = state.get("request_path") or ""
        request_body = state.get("request_body") or ""

        # 结束埋点 (对齐 n8n 的 "埋点" 节点)
        # 注: 表中只有基础字段,暂不记录详细结果
        def insert_end_track() -> None:
            try:
                # 埋点记录已在 init_state_node 中完成
                # 如需记录详细结果,需先在数据库表中添加相应字段
                pass
            except Exception as e:
                logger.warning(f"[海外探款商品] 埋点记录失败: {e}")

        thread_pool.submit_with_context(insert_end_track)

        if has_query_result:
            # entity_data 已是字典列表，无需调用 model_dump()
            entity_dicts = entity_data
            response = WorkflowResponse(
                select_result="基于以上条件，您的选品任务已经完成，还有其他需要帮助的地方吗？",
                relate_data=json.dumps(entity_dicts, ensure_ascii=False),
            )
        else:
            response = WorkflowResponse(
                select_result="无结果，可能与价格、销量、筛选条件有关",
                relate_data=None,
            )
        return {"workflow_response": response}


__all__ = ["AbroadGoodsGraph"]
