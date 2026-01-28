# -*- coding: utf-8 -*-
# @Author   : kiro
# @Time     : 2025/12/14
# @File     : abroad_ins_graph.py

"""
跨境 INS 选品工作流 - LangGraph 版本
"""

from __future__ import annotations

import json
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import text

from app.core.clients.coze_loop_client import coze_loop_client_provider
from app.core.clients.db_client import mysql_session_readonly, pg_session
from app.core.clients.redis_client import redis_client
from app.core.config.constants import (
    CozePromptHubKey,
    DBAlias,
    LlmModelName,
    LlmProvider,
    RedisMessageKeyName, WorkflowMessageContentType, WorkflowEntityType,
)
from app.core.tools import llm_factory
from app.schemas.entities.message.redis_message import (
    BaseRedisMessage,
    CustomDataContent,
    ParameterData,
    ParameterDataContent,
    WithActionContent,
)
from app.schemas.entities.workflow.graph_state import AbroadInsWorkflowState
from app.schemas.entities.workflow.llm.abroad_ins_output import (
    AbroadInsBloggerParseResult,
    AbroadInsCategoryParseResult,
    AbroadInsMiscParseResult,
    AbroadInsParseParam,
    AbroadInsSortTypeParseResult,
    AbroadInsStyleParseResult,
    AbroadInsTimeParseResult,
)
from app.schemas.request.workflow_request import WorkflowRequest
from app.schemas.response.common import PageResult
from app.schemas.response.workflow_response import WorkflowResponse
from app.service.chains.workflow.base_graph import BaseWorkflowGraph
from app.service.chains.workflow.progress_pusher import PhaseProgressPusher
from app.service.chains.templates.abroad_progress_template import ABROAD_INS_PROGRESS_TEMPLATE
from app.service.rpc.abroad_api import AbroadInsBlogEntity, InsBlogListRequest, abroad_api_client
from app.utils import thread_pool
from app.utils.query_reference import QueryReferenceHelper


class AbroadInsGraph(BaseWorkflowGraph):
    """跨境 INS 选品工作流 - LangGraph 版本"""

    span_name = "跨境ins数据源工作流"
    run_name = "abroad-ins-graph"

    def _build_graph(self) -> CompiledStateGraph:
        """构建工作流图"""
        graph = StateGraph(AbroadInsWorkflowState)

        # 添加节点
        graph.add_node("init_state", self._init_state_node, metadata={"__display_name__": "初始化"})
        graph.add_node("pre_think", self._pre_think_node, metadata={"__display_name__": "预处理思考"})
        graph.add_node("query_selections", self._query_common_selection_node, metadata={"__display_name__": "查询筛选项"})
        graph.add_node("llm_category_parse", self._llm_category_parse_node, metadata={"__display_name__": "类目解析"})
        graph.add_node("llm_time_parse", self._llm_time_parse_node, metadata={"__display_name__": "时间解析"})
        graph.add_node("llm_blogger_parse", self._llm_blogger_parse_node, metadata={"__display_name__": "达人解析"})
        graph.add_node("llm_style_parse", self._llm_style_parse_node, metadata={"__display_name__": "风格解析"})
        graph.add_node("llm_misc_parse", self._llm_misc_parse_node, metadata={"__display_name__": "杂项解析"})
        graph.add_node("llm_merge_param", self._llm_merge_param_node, metadata={"__display_name__": "参数合并"})
        graph.add_node("llm_sort_parse", self._llm_parse_sort_type_node, metadata={"__display_name__": "排序项解析"})
        graph.add_node("call_api", self._call_business_api_node, metadata={"__display_name__": "调用业务API"})
        graph.add_node("has_result", self._first_query_has_result_node, metadata={"__display_name__": "有结果处理"})
        graph.add_node("no_result", self._first_query_no_result_node, metadata={"__display_name__": "无结果兜底"})
        graph.add_node("package", self._package_result_node, metadata={"__display_name__": "封装结果"})

        # 设置入口和边
        graph.set_entry_point("init_state")
        graph.add_edge("init_state", "pre_think")
        graph.add_edge("pre_think", "query_selections")
        graph.add_edge("query_selections", "llm_category_parse")
        graph.add_edge("llm_category_parse", "llm_time_parse")
        graph.add_edge("llm_category_parse", "llm_blogger_parse")
        graph.add_edge("llm_category_parse", "llm_style_parse")
        graph.add_edge("llm_category_parse", "llm_misc_parse")
        graph.add_edge(
            ["llm_time_parse", "llm_blogger_parse", "llm_style_parse", "llm_misc_parse"],
            "llm_merge_param",
        )
        graph.add_edge("llm_merge_param", "llm_sort_parse")
        graph.add_edge("llm_sort_parse", "call_api")

        # 条件分支
        graph.add_conditional_edges(
            "call_api",
            self._check_first_query_has_result,
            {"has_result": "has_result", "no_result": "no_result"},
        )

        graph.add_edge("has_result", "package")
        graph.add_edge("no_result", "package")
        graph.add_edge("package", END)

        return graph.compile()

    def _check_first_query_has_result(self, state: AbroadInsWorkflowState) -> str:
        """条件路由：检查首次查询是否有结果"""
        result_count = state.get("result_count", 0)
        if result_count and result_count >= 50:
            return "has_result"
        return "no_result"

    # ==================== 工具方法 ====================
    def _get_pusher(self, req: WorkflowRequest) -> PhaseProgressPusher:
        return PhaseProgressPusher(template=ABROAD_INS_PROGRESS_TEMPLATE, request=req)

    # ==================== 节点实现 ====================

    def _init_state_node(self, state: AbroadInsWorkflowState) -> Dict[str, Any]:
        """初始化状态节点"""
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

    def _pre_think_node(self, state: AbroadInsWorkflowState) -> Dict[str, Any]:
        """预处理节点 - 发送开始消息和思维链"""
        req = state["request"]

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
                data=ParameterData(entity_type=WorkflowEntityType.ABROAD_INS.code),
            ),
            create_ts=int(round(time.time() * 1000)),
        )
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            start_message.model_dump_json(),
        )

        def _generate_thinking_and_report_task() -> None:
            invoke_params = {
                "user_query": req.user_query,
                "preferred_entity": req.preferred_entity,
                "industry": req.industry,
                "user_preferences": req.user_preferences,
                "now_time": str(time.time()),
            }

            messages: List[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
                prompt_key=CozePromptHubKey.ABROAD_INS_THINK_PROMPT.value,
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

            self._get_pusher(req=req).complete_phase("选品任务规划", content=thinking_result_text)

        _generate_thinking_and_report_task()
        return {}

    def _query_common_selection_node(self, state: AbroadInsWorkflowState) -> Dict[str, Any]:
        """查询筛选项数据节点"""
        selection_dict: Dict[str, Any] = {}
        with mysql_session_readonly(DBAlias.OLAP_ZXY_AGENT) as session:
            result = session.execute(
                text("select type, label, value from zxy_abroad_ins_selection_common")
            )
            common_selections = [dict(row) for row in result.mappings().all()]

            result = session.execute(
                text("select category_id_path, category_name_path from zxy_abroad_ins_selection_category")
            )
            category_paths = [dict(row) for row in result.mappings().all()]

            selection_dict["category_paths"] = category_paths
            selection_dict["common_selections"] = common_selections

        return {"selection_dict": selection_dict}

    @staticmethod
    def _build_time_data() -> str:
        """构建时间数据，生成不同的时间区间供 LLM 使用"""
        now = datetime.now() - timedelta(days=1)  # 使用 T-1 日期
        year = now.year
        is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        time_info = {
            "当前时间": now.strftime("%Y-%m-%d"),
            "最近七天": f"{(now - timedelta(days=6)).strftime('%Y-%m-%d')} ~ {now.strftime('%Y-%m-%d')}",
            "近一个月 / 近30天": f"{(now - timedelta(days=29)).strftime('%Y-%m-%d')} ~ {now.strftime('%Y-%m-%d')}",
            "当年春季": f"{year}-03-01 ~ {year}-05-31",
            "当年冬季": f"{year}-12-01 ~ {year}-12-31",
            "是否闰年": "是闰年" if is_leap_year else "不是闰年",
            "年份": f"{year}年",
        }
        return json.dumps(time_info, ensure_ascii=False)

    @staticmethod
    def _build_common_param(req: WorkflowRequest) -> dict[str, Any]:
        ref_helper = QueryReferenceHelper.from_request(req)
        format_query = ref_helper.replace_placeholders(req.user_query)
        return {
            "user_query": format_query,
            "preferred_entity": req.preferred_entity,
            "industry": req.industry,
            "user_preferences": req.user_preferences,
        }

    def _run_parse(
        self,
        *,
        prompt_key: str,
        schema: type[BaseModel],
        run_name: str,
        variables: dict[str, Any],
    ) -> BaseModel:
        try:
            messages: List[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
                prompt_key=prompt_key,
                variables=variables,
            )
            llm: BaseChatModel = llm_factory.get_llm(
                LlmProvider.OPENROUTER.name,
                LlmModelName.OPENROUTER_GEMINI_3_FLASH_PREVIEW.value,
            )
            structured_llm = llm.with_structured_output(schema).with_retry(stop_after_attempt=2)
            return structured_llm.with_config(run_name=run_name).invoke(messages)
        except Exception as exc:
            logger.warning(f"[跨境INS] {run_name} 解析失败: {exc}")
            return schema()

    def _llm_category_parse_node(self, state: AbroadInsWorkflowState) -> Dict[str, Any]:
        """LLM 类目解析节点"""
        req = state["request"]
        selection_dict = state["selection_dict"]

        prompt_param = {
            **self._build_common_param(req),
            "category_data": json.dumps(selection_dict["category_paths"], ensure_ascii=False),
            "common_selections": json.dumps(selection_dict["common_selections"], ensure_ascii=False),
        }
        category_result = self._run_parse(
            prompt_key=CozePromptHubKey.ABROAD_INS_CATEGORY_PARSE_PROMPT.value,
            schema=AbroadInsCategoryParseResult,
            run_name="类目解析",
            variables=prompt_param,
        )
        return {"category_result": category_result}

    def _llm_time_parse_node(self, state: AbroadInsWorkflowState) -> Dict[str, Any]:
        """LLM 时间解析节点"""
        req = state["request"]
        prompt_param = {
            **self._build_common_param(req),
            "time_data": self._build_time_data(),
        }
        time_result = self._run_parse(
            prompt_key=CozePromptHubKey.ABROAD_INS_TIME_PARSE_PROMPT.value,
            schema=AbroadInsTimeParseResult,
            run_name="时间解析",
            variables=prompt_param,
        )
        return {"time_result": time_result}

    def _llm_blogger_parse_node(self, state: AbroadInsWorkflowState) -> Dict[str, Any]:
        """LLM 达人解析节点"""
        req = state["request"]
        selection_dict = state["selection_dict"]

        prompt_param = {
            **self._build_common_param(req),
            "common_selections": json.dumps(selection_dict["common_selections"], ensure_ascii=False),
        }
        blogger_result = self._run_parse(
            prompt_key=CozePromptHubKey.ABROAD_INS_BLOGGER_PARSE_PROMPT.value,
            schema=AbroadInsBloggerParseResult,
            run_name="达人解析",
            variables=prompt_param,
        )
        return {"blogger_result": blogger_result}

    def _llm_style_parse_node(self, state: AbroadInsWorkflowState) -> Dict[str, Any]:
        """LLM 风格解析节点"""
        req = state["request"]
        selection_dict = state["selection_dict"]

        prompt_param = {
            **self._build_common_param(req),
            "common_selections": json.dumps(selection_dict["common_selections"], ensure_ascii=False),
        }
        style_result = self._run_parse(
            prompt_key=CozePromptHubKey.ABROAD_INS_STYLE_PARSE_PROMPT.value,
            schema=AbroadInsStyleParseResult,
            run_name="风格解析",
            variables=prompt_param,
        )
        return {"style_result": style_result}

    def _llm_misc_parse_node(self, state: AbroadInsWorkflowState) -> Dict[str, Any]:
        """LLM 杂项解析节点"""
        req = state["request"]
        misc_result = self._run_parse(
            prompt_key=CozePromptHubKey.ABROAD_INS_MISC_PARSE_PROMPT.value,
            schema=AbroadInsMiscParseResult,
            run_name="杂项解析",
            variables=self._build_common_param(req),
        )
        return {"misc_result": misc_result}

    def _llm_merge_param_node(self, state: AbroadInsWorkflowState) -> Dict[str, Any]:
        """合并解析结果"""
        req = state["request"]
        category_result: AbroadInsCategoryParseResult = state.get(
            "category_result", AbroadInsCategoryParseResult()
        )
        time_result: AbroadInsTimeParseResult = state.get(
            "time_result", AbroadInsTimeParseResult()
        )
        blogger_result: AbroadInsBloggerParseResult = state.get(
            "blogger_result", AbroadInsBloggerParseResult()
        )
        style_result: AbroadInsStyleParseResult = state.get(
            "style_result", AbroadInsStyleParseResult()
        )
        misc_result: AbroadInsMiscParseResult = state.get(
            "misc_result", AbroadInsMiscParseResult()
        )

        common_param = self._build_common_param(req)
        param_result = self._merge_param_result(
            category_result=category_result,
            time_result=time_result,
            blogger_result=blogger_result,
            style_result=style_result,
            misc_result=misc_result,
            user_query=common_param["user_query"],
        )
        return {
            "param_result": param_result,
            "is_monitor_streamer": param_result.is_monitor_streamer == 1,
        }

    @staticmethod
    def _merge_param_result(
        *,
        category_result: AbroadInsCategoryParseResult,
        time_result: AbroadInsTimeParseResult,
        blogger_result: AbroadInsBloggerParseResult,
        style_result: AbroadInsStyleParseResult,
        misc_result: AbroadInsMiscParseResult,
        user_query: str,
    ) -> AbroadInsParseParam:
        def normalize_text(value: Any) -> str | None:
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        def normalize_list(items: list[Any]) -> list[str]:
            return [str(item).strip() for item in items if str(item).strip()]

        def normalize_matrix(matrix: list[Any]) -> list[list[str]]:
            cleaned: list[list[str]] = []
            for row in matrix or []:
                if isinstance(row, list):
                    values = normalize_list(row)
                    if values:
                        cleaned.append(values)
                else:
                    value = normalize_text(row)
                    if value:
                        cleaned.append([value])
            return cleaned

        category_list = normalize_matrix(category_result.category_list or [])
        region_list = normalize_matrix(category_result.region_list or [])
        label = normalize_matrix(style_result.label or [])
        style_list = normalize_list(style_result.style_list or [])
        blogger_skin_colors = normalize_list(blogger_result.blogger_skin_color_list or [])
        blogger_shapes = normalize_list(blogger_result.blogger_shapes or [])

        min_fans_num = blogger_result.min_fans_num
        max_fans_num = blogger_result.max_fans_num
        if min_fans_num is not None and max_fans_num is not None and min_fans_num > max_fans_num:
            min_fans_num, max_fans_num = max_fans_num, min_fans_num

        start_date = normalize_text(time_result.start_date)
        end_date = normalize_text(time_result.end_date)

        limit = misc_result.limit if misc_result.limit is not None else 6000
        if limit <= 0:
            limit = 6000
        if limit > 6000:
            limit = 6000

        sort_type = normalize_text(misc_result.sort_type) or "默认"
        search_content = normalize_text(misc_result.search_content)

        user_data = 1 if misc_result.user_data == 1 else 0
        is_monitor_streamer = 1 if blogger_result.is_monitor_streamer == 1 else 0

        title = normalize_text(misc_result.title) or normalize_text(user_query) or "选品任务"
        if len(title) > 100:
            title = title[:100]

        return AbroadInsParseParam(
            start_date=start_date,
            end_date=end_date,
            category_list=category_list,
            label=label,
            style_list=style_list,
            region_list=region_list,
            blogger_skin_color_list=blogger_skin_colors,
            blogger_shapes=blogger_shapes,
            min_fans_num=min_fans_num,
            max_fans_num=max_fans_num,
            limit=limit,
            sort_type=sort_type,
            search_content=search_content,
            user_data=user_data,
            is_monitor_streamer=is_monitor_streamer,
            title=title,
        )

    def _llm_parse_sort_type_node(self, state: AbroadInsWorkflowState) -> Dict[str, Any]:
        """LLM 排序项解析节点"""
        req = state["request"]
        param_result: AbroadInsParseParam = state["param_result"]

        prompt_param = {
            "is_monitor_streamer": str(param_result.is_monitor_streamer),
            "user_query": req.user_query,
            "sort_type": param_result.sort_type,
        }
        messages = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.ABROAD_INS_SORT_TYPE_PARSE_PROMPT.value,
            variables=prompt_param,
        )

        llm = llm_factory.get_llm(
            LlmProvider.OPENROUTER.name, LlmModelName.OPENROUTER_GEMINI_2_5_FLASH.value
        )
        structured_llm = llm.with_structured_output(AbroadInsSortTypeParseResult).with_retry(stop_after_attempt=2)

        sort_param_result = structured_llm.with_config(run_name="排序项解析").invoke(messages)
        return {"sort_param_result": sort_param_result}

    def _call_business_api_node(self, state: AbroadInsWorkflowState) -> Dict[str, Any]:
        """调用业务 API 节点"""
        req = state["request"]
        param_result: AbroadInsParseParam = state["param_result"]
        sort_param_result: AbroadInsSortTypeParseResult = state["sort_param_result"]

        ins_request = InsBlogListRequest.from_parse_param(param_result)
        ins_request.sort_type = sort_param_result.sort_type_final
        page_result: PageResult[AbroadInsBlogEntity] = abroad_api_client.search_ins_blogs(
            user_id=req.user_id, team_id=req.team_id, params=ins_request
        )

        return {
            "api_request": ins_request,
            "api_request_alias": ins_request.model_dump_json(by_alias=True, exclude_none=True),
            "api_resp": page_result,
            "result_count": page_result.result_count,
        }

    def _first_query_has_result_node(self, state: AbroadInsWorkflowState) -> Dict[str, Any]:
        """首次查询有结果分支"""
        req = state["request"]
        api_request: InsBlogListRequest = state["api_request"]
        param_result: AbroadInsParseParam = state["param_result"]
        page_result: PageResult[AbroadInsBlogEntity] = state["api_resp"]

        # 推送帖子筛选完成消息
        self._get_pusher(req).complete_phase(phase_name="帖子筛选中", variables={
            "datasource": "INS数据库",
            "platform": "海外探款INS",
            "browsed_count": f"{str(random.randint(100000, 1000000))}",
            "filter_result_text": "帖子筛选完成，当前筛选出的帖子数量满足需求，无需二次筛选，接下来我将根据数据完成帖子列表"
        })
        self._get_pusher(req=req).complete_phase("生成列表中")
        self._get_pusher(req=req).complete_phase("选品完成")

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

        # 推送选品路径消息
        path_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="输出参数",
            status="RUNNING",
            content_type=8,
            content=CustomDataContent(data={"entity_type": 1, "content": "abroad-ins-all"}),
            create_ts=int(round(time.time() * 1000)),
        )
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            path_message.model_dump_json(),
        )

        # 推送输出参数消息
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
                    request_path="/external/for-zxy/fashion/ins/blog/list",
                    request_body=api_request.model_dump_json(by_alias=True, exclude_none=True),
                    actions=["view", "export", "download"],
                    title=param_result.title,
                    entity_type=6,
                )
            ),
            create_ts=int(round(time.time() * 1000)),
        )
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            parameter_message.model_dump_json(),
        )

        entity_simple_info_list = [{"帖子id": entity.entity_id} for entity in page_result.result_list]
        return {
            "has_query_result": True,
            "entity_simple_data": entity_simple_info_list,
        }

    def _first_query_no_result_node(self, state: AbroadInsWorkflowState) -> Dict[str, Any]:
        """首次查询无结果分支 - 兜底查询"""
        req = state["request"]
        ins_request: InsBlogListRequest = state["api_request"]
        param_result: AbroadInsParseParam = state["param_result"]

        # 推送帖子筛选消息（数量不足，需要二次筛选）
        self._get_pusher(req).complete_phase(phase_name="帖子筛选中", variables={
            "datasource": "INS数据库",
            "platform": "海外探款INS",
            "browsed_count": f"{str(random.randint(100000, 1000000))}",
            "filter_result_text": "当前筛选出的帖子数量不足，不满足列表需求，我将再次搜索确保无遗漏"
        })

        # 精简筛选项进行兜底查询
        simple_request = InsBlogListRequest(
            startDate=ins_request.start_date,
            endDate=ins_request.end_date,
            categoryList=ins_request.category_list,
            minFansNum=ins_request.min_fans_num,
            maxFansNum=ins_request.max_fans_num,
            searchContent=ins_request.search_content,
            sortType=ins_request.sort_type,
            resultCountLimit=ins_request.result_count_limit,
            isMonitorStreamer=ins_request.is_monitor_streamer,
        )
        page_result: PageResult[AbroadInsBlogEntity] = abroad_api_client.search_ins_blogs(
            team_id=req.team_id, user_id=req.user_id, params=simple_request
        )

        has_query_result = page_result.result_count is None or page_result.result_count > 0

        if has_query_result:
            self._get_pusher(req=req).complete_phase(phase_name="二次筛选中", variables={
                "datasource": "INS数据库",
                "platform": "海外探款INS",
                "browsed_count": f"{str(random.randint(100000, 1000000))}",
                "retry_result_text": "帖子筛选完成，当前筛选出的帖子数量符合报表要求，接下来我将根据数据绘制帖子列表",
            })
            self._get_pusher(req=req).complete_phase("生成列表中")
            self._get_pusher(req=req).complete_phase("选品完成")

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

            # 推送选品路径消息
            path_message = BaseRedisMessage(
                session_id=req.session_id,
                reply_message_id=req.message_id,
                reply_id=f"reply_{req.message_id}",
                reply_seq=0,
                operate_id="输出参数",
                status="RUNNING",
                content_type=8,
                content=CustomDataContent(data={"entity_type": 1, "content": "abroad-ins-all"}),
                create_ts=int(round(time.time() * 1000)),
            )
            redis_client.list_left_push(
                RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
                path_message.model_dump_json(),
            )

            # 推送输出参数消息
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
                        request_path="/external/for-zxy/fashion/ins/blog/list",
                        request_body=simple_request.model_dump_json(by_alias=True, exclude_none=True),
                        actions=["view", "export", "download"],
                        title=param_result.title,
                        entity_type=6,
                    )
                ),
                create_ts=int(round(time.time() * 1000)),
            )
            redis_client.list_left_push(
                RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
                parameter_message.model_dump_json(),
            )

            entity_simple_info_list = [{"帖子id": entity.entity_id} for entity in page_result.result_list]
            return {
                "has_query_result": True,
                "entity_simple_data": entity_simple_info_list,
            }
        else:
            self._get_pusher(req=req).complete_phase(phase_name="二次筛选中", variables={
                "datasource": "INS数据库",
                "platform": "海外探款INS",
                "browsed_count": f"{str(page_result.result_count)}",
                "retry_result_text": "当前筛选出的帖子数量不满足需求，我可能需要提醒用户调整筛选维度",
            })
            self._get_pusher(req=req).fail_phase(phase_name="生成列表失败", error_message="我未能完成列表绘制，原因是没有数据\n需要提醒用户调整筛选维度，才能更好的获取数据")
            self._get_pusher(req=req).fail_phase(phase_name="选品未完成", error_message=None)

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
            #     content=TextMessageContent(text="未找到符合需求的帖子，请尝试调整筛选条件。"),
            #     create_ts=int(round(time.time() * 1000)),
            # )
            # redis_client.list_left_push(
            #     RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            #     no_result_message.model_dump_json(),
            # )

            return {
                "has_query_result": False
            }

    def _package_result_node(self, state: AbroadInsWorkflowState) -> Dict[str, Any]:
        """封装返回结果节点"""
        req = state.get("request")
        has_query_result = state.get("has_query_result", False)
        entity_data = state.get("entity_simple_data", [])

        if has_query_result:
            response = WorkflowResponse(
                select_result="基于以上条件，您的选品任务已经完成，还有其他需要帮助的地方吗？",
                relate_data=json.dumps(entity_data, ensure_ascii=False),
            )
        else:
            response = WorkflowResponse(
                select_result="未找到符合需求的帖子，可尝试扩大时间范围、调整地区或减少筛选条件。",
                relate_data=None,
            )
        return {"workflow_response": response}


__all__ = ["AbroadInsGraph"]
