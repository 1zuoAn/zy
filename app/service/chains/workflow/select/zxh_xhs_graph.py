# -*- coding: utf-8 -*-
# @Author   : kiro
# @Time     : 2025/12/14
# @File     : zxh_xhs_graph.py

"""
知小红 小红书选品工作流 - LangGraph 版本
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from sqlalchemy import text

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
)
from app.core.tools import llm_factory
from app.schemas.entities.message.redis_message import (
    BaseRedisMessage,
    CustomDataContent,
    ParameterData,
    ParameterDataContent,
    TaskProgressContent,
    TaskProgressItem,
    TextMessageContent,
    WithActionContent,
)
from app.schemas.entities.workflow.graph_state import ZxhXhsWorkflowState
from app.schemas.entities.workflow.llm_output import (
    RagCleanedResult,
    ZxhXhsParseParam,
    ZxhXhsSortTypeParseResult,
)
from app.schemas.response.common import PageResult
from app.schemas.response.workflow_response import WorkflowResponse
from app.service.chains.workflow.base_graph import BaseWorkflowGraph
from app.service.rpc.volcengine_kb_api import volcengine_kb_client
from app.service.rpc.zxh_api import (
    XhsBlogEntity,
    XhsNoteMonitorRequest,
    XhsNoteSearchRequest,
    zxh_api_client,
)
from app.utils import thread_pool


class ZxhXhsGraph(BaseWorkflowGraph):
    """知小红 小红书选品工作流 - LangGraph 版本"""

    span_name = "知小红小红书数据源工作流"
    run_name = "zxh-xhs-graph"

    # 进度步骤定义
    PROGRESS_STEPS = [
        {"name": "理解需求", "detail": "正在分析您的选品需求..."},
        {"name": "解析参数", "detail": "正在提取品类、时间等条件..."},
        {"name": "知识检索", "detail": "正在检索相关话题标签..."},
        {"name": "搜索笔记", "detail": "正在搜索符合条件的小红书笔记..."},
        {"name": "筛选结果", "detail": None},  # 动态更新
        {"name": "生成结果", "detail": "正在整理选品结果..."},
    ]

    def _build_graph(self) -> CompiledStateGraph:
        """构建工作流图"""
        graph = StateGraph(ZxhXhsWorkflowState)

        graph.add_node("init_state", self._init_state_node)
        graph.add_node("pre_think", self._pre_think_node)
        graph.add_node("query_selections", self._query_common_selection_node)
        graph.add_node("llm_parse", self._llm_param_parse_node)
        graph.add_node("llm_sort_parse", self._llm_parse_sort_type_node)
        graph.add_node("rag_query", self._rag_query_node)
        graph.add_node("call_api", self._call_business_api_node)
        graph.add_node("has_result", self._first_query_has_result_node)
        graph.add_node("no_result", self._first_query_no_result_node)
        graph.add_node("package", self._package_result_node)

        graph.set_entry_point("init_state")
        graph.add_edge("init_state", "pre_think")
        graph.add_edge("pre_think", "query_selections")
        graph.add_edge("query_selections", "llm_parse")
        graph.add_edge("llm_parse", "llm_sort_parse")
        graph.add_edge("llm_sort_parse", "rag_query")
        graph.add_edge("rag_query", "call_api")

        graph.add_conditional_edges(
            "call_api",
            self._check_first_query_has_result,
            {"has_result": "has_result", "no_result": "no_result"},
        )

        graph.add_edge("has_result", "package")
        graph.add_edge("no_result", "package")
        graph.add_edge("package", END)

        return graph.compile()

    def _check_first_query_has_result(self, state: ZxhXhsWorkflowState) -> str:
        result_count = state.get("result_count")
        if result_count is None:
            result_count = state.get("browsed_count", 0)
        if result_count and result_count >= 5:
            return "has_result"
        return "no_result"

    # ==================== 工具方法 ====================

    def _push_progress(
        self,
        state: ZxhXhsWorkflowState,
        current_step: int,
        step_detail: Optional[str] = None,
    ) -> None:
        """推送任务进度消息"""
        req = state["request"]

        steps = []
        for i, step_def in enumerate(self.PROGRESS_STEPS):
            if i < current_step:
                status = "completed"
            elif i == current_step:
                status = "running"
            else:
                status = "pending"

            detail = step_detail if i == current_step and step_detail else step_def["detail"]
            steps.append(TaskProgressItem(
                step_name=step_def["name"],
                status=status,
                detail=detail,
            ))

        message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="任务进度",
            status="RUNNING",
            content_type=9,
            content=TaskProgressContent(
                current_step=current_step,
                total_steps=len(self.PROGRESS_STEPS),
                steps=steps,
            ),
            create_ts=int(round(time.time() * 1000)),
        )
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            message.model_dump_json(),
        )

    # ==================== 节点实现 ====================

    def _init_state_node(self, state: ZxhXhsWorkflowState) -> Dict[str, Any]:
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

    def _pre_think_node(self, state: ZxhXhsWorkflowState) -> Dict[str, Any]:
        """预处理节点 - 发送开始消息和思维链"""
        req = state["request"]

        start_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="收到任务",
            status="RUNNING",
            content_type=2,
            content=WithActionContent(
                text="收到，我会为你选出想要的款式",
                actions=["view", "export", "download"],
                agent="search",
            ),
            create_ts=int(round(time.time() * 1000)),
        )
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            start_message.model_dump_json(),
        )

        # 推送初始进度
        self._push_progress(state, 0)

        def _generate_thinking_and_report_task() -> None:
            invoke_params = {
                "user_query": req.user_query,
                "preferred_entity": req.preferred_entity,
                "industry": req.industry,
                "user_preferences": req.user_preferences,
                "now_time": str(time.time()),
            }

            messages: List[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
                prompt_key=CozePromptHubKey.ZXH_XHS_THINK_PROMPT.value,
                variables=invoke_params,
            )
            llm: BaseChatModel = llm_factory.get_llm(
                LlmProvider.OPENROUTER.name, LlmModelName.OPENROUTER_GPT_4O.value
            )
            retry_llm = llm.with_retry(stop_after_attempt=2)

            thinking_chain = retry_llm | StrOutputParser()
            thinking_result_text = thinking_chain.with_config(run_name="思维链生成").invoke(messages)
            thinking_result_text = thinking_result_text.replace("\n\n", "\n")

            thinking_result_message = BaseRedisMessage(
                session_id=req.session_id,
                reply_message_id=req.message_id,
                reply_id=f"reply_{req.message_id}",
                reply_seq=0,
                operate_id="思维链",
                status="RUNNING",
                content_type=4,
                content=TextMessageContent(text=thinking_result_text),
                create_ts=int(round(time.time() * 1000)),
            )
            redis_client.list_left_push(
                RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
                thinking_result_message.model_dump_json(),
            )

            searching_message = BaseRedisMessage(
                session_id=req.session_id,
                reply_message_id=req.message_id,
                reply_id=f"reply_{req.message_id}",
                reply_seq=0,
                operate_id="正在检索中",
                status="RUNNING",
                content_type=1,
                content=TextMessageContent(text="正在生成选品任务..."),
                create_ts=int(round(time.time() * 1000)),
            )
            redis_client.list_left_push(
                RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
                searching_message.model_dump_json(),
            )

            starting_select_message = BaseRedisMessage(
                session_id=req.session_id,
                reply_message_id=req.message_id,
                reply_id=f"reply_{req.message_id}",
                reply_seq=0,
                operate_id="标签汇总",
                status="RUNNING",
                content_type=1,
                content=TextMessageContent(text=f"正在{req.preferred_entity}上为你选品..."),
                create_ts=int(round(time.time() * 1000)),
            )
            redis_client.list_left_push(
                RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
                starting_select_message.model_dump_json(),
            )

        thread_pool.submit_with_context(_generate_thinking_and_report_task)
        return {}

    def _query_common_selection_node(self, state: ZxhXhsWorkflowState) -> Dict[str, Any]:
        """查询筛选项数据节点"""
        selection_dict: Dict[str, Any] = {}
        with mysql_session_readonly(DBAlias.OLAP_ZXY_AGENT) as session:
            result = session.execute(
                text("select type, label, value from zxy_zxh_note_selection_common")
            )
            common_selections = [dict(row) for row in result.mappings().all()]

            result = session.execute(
                text("select industry_name, category_id_path from zxy_zxh_note_selection_category")
            )
            category_paths = [dict(row) for row in result.mappings().all()]

            selection_dict["category_paths"] = category_paths
            selection_dict["common_selections"] = common_selections

        # 更新进度 - 解析参数
        self._push_progress(state, 1)

        return {"selection_dict": selection_dict}

    def _llm_param_parse_node(self, state: ZxhXhsWorkflowState) -> Dict[str, Any]:
        req = state["request"]
        selection_dict = state["selection_dict"]

        prompt_param = {
            "user_query": req.user_query,
            "preferred_entity": req.preferred_entity,
            "industry": req.industry,
            "user_preferences": req.user_preferences,
            "common_selections": json.dumps(selection_dict["common_selections"], ensure_ascii=False),
            "category_data": json.dumps(selection_dict["category_paths"], ensure_ascii=False),
        }
        messages: List[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.ZXH_XHS_MAIN_PARAM_PARSE_PROMPT.value,
            variables=prompt_param,
        )

        llm: BaseChatModel = llm_factory.get_llm(
            LlmProvider.OPENROUTER.name, LlmModelName.OPENROUTER_GEMINI_3_FLASH_PREVIEW.value
        )
        structured_llm = llm.with_structured_output(ZxhXhsParseParam).with_retry(stop_after_attempt=2)
        param_result = structured_llm.with_config(run_name="参数解析").invoke(messages)
        return {"param_result": param_result}

    def _rag_query_node(self, state: ZxhXhsWorkflowState) -> Dict[str, Any]:
        """RAG 知识库检索节点"""
        param_result: ZxhXhsParseParam = state["param_result"]

        # 更新进度 - 知识检索
        self._push_progress(state, 2)

        if not param_result.topic_query_text:
            return {}

        kb_response = volcengine_kb_client.simple_chat(
            query=param_result.topic_query_text,
            service_resource_id=VolcKnowledgeServiceId.XHS_TOPIC_KNOWLEDGE.value,
        )

        content_list = [
            item.content for item in kb_response.data.result_list if item.content
        ]

        if not content_list:
            logger.debug("[RAG节点] 知识库无召回内容")
            return {"recall_topic_list": None, "cleaned_topic_list": None}

        cleaned_result = self._clean_rag_content(content_list, state["request"])

        logger.debug(f"[RAG节点] 召回 {len(content_list)} 条内容，清洗后 {len(cleaned_result.content_list)} 条")
        return {
            "recall_topic_list": content_list,
            "cleaned_topic_list": cleaned_result.content_list,
        }

    def _clean_rag_content(self, content_list: List[str], req: Any) -> RagCleanedResult:
        """使用 LLM 清洗 RAG 召回内容"""
        prompt_param = {
            "user_query": req.user_query,
            "content_list": json.dumps(content_list, ensure_ascii=False),
        }

        messages: List[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.ZXH_XHS_TOPIC_RAG_CLEAN_PROMPT.value,
            variables=prompt_param,
        )

        llm: BaseChatModel = llm_factory.get_llm(
            LlmProvider.OPENROUTER.name,
            LlmModelName.OPENROUTER_GEMINI_2_5_FLASH.value,
        )
        structured_llm = llm.with_structured_output(RagCleanedResult).with_retry(stop_after_attempt=2)
        return structured_llm.with_config(run_name="RAG内容清洗").invoke(messages)

    def _llm_parse_sort_type_node(self, state: ZxhXhsWorkflowState) -> Dict[str, Any]:
        req = state["request"]
        param_result: ZxhXhsParseParam = state["param_result"]

        prompt_param = {
            "is_monitor_blogger": str(param_result.is_monitor_blogger),
            "user_query": req.user_query,
            "sort_type": param_result.sort_field,
        }
        messages = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.ZXH_XHS_SORT_TYPE_PARSE_PROMPT.value,
            variables=prompt_param,
        )

        llm = llm_factory.get_llm(
            LlmProvider.OPENROUTER.name, LlmModelName.OPENROUTER_GEMINI_2_5_FLASH.value
        )
        structured_llm = llm.with_structured_output(ZxhXhsSortTypeParseResult).with_retry(stop_after_attempt=2)
        sort_param_result = structured_llm.with_config(run_name="排序项解析").invoke(messages)
        return {"sort_param_result": sort_param_result}

    def _call_business_api_node(self, state: ZxhXhsWorkflowState) -> Dict[str, Any]:
        """调用业务 API 节点"""
        req = state["request"]
        param_result: ZxhXhsParseParam = state["param_result"]
        sort_param_result: ZxhXhsSortTypeParseResult = state["sort_param_result"]
        topic_list = state.get("cleaned_topic_list")

        # 更新进度 - 搜索笔记
        self._push_progress(state, 3)

        is_monitor_query = param_result.is_monitor_blogger == 1

        if is_monitor_query:
            api_request = XhsNoteMonitorRequest.from_parse_param(param_result)
            api_request.sort_field = sort_param_result.sort_type_final
            api_request.sort_order = sort_param_result.sort_order
            api_request.topic_list = topic_list
            api_request.group_id_list = ["-3"]
            api_request.monitor_type = "blogger"
            page_result = zxh_api_client.note_monitor_search(
                user_id=req.user_id, team_id=req.team_id, params=api_request
            )
        else:
            api_request = XhsNoteSearchRequest.from_parse_param(param_result)
            api_request.sort_field = sort_param_result.sort_type_final
            api_request.sort_order = sort_param_result.sort_order
            api_request.topic_list = topic_list
            page_result = zxh_api_client.common_search(
                user_id=req.user_id, team_id=req.team_id, params=api_request
            )

        result_count = page_result.result_count
        if result_count is None:
            result_count = len(page_result.result_list)
        browsed_count = result_count or 0

        # 更新进度 - 显示浏览笔记数
        self._push_progress(state, 4, f"已浏览 {browsed_count} 条笔记")

        return {
            "is_monitor_query": is_monitor_query,
            "api_request": api_request,
            "api_resp": page_result,
            "result_count": result_count,
            "browsed_count": browsed_count,
        }

    def _first_query_has_result_node(self, state: ZxhXhsWorkflowState) -> Dict[str, Any]:
        """首次查询有结果分支"""
        req = state["request"]
        is_monitor_query: bool = state["is_monitor_query"]
        api_request = state["api_request"]
        param_result: ZxhXhsParseParam = state["param_result"]
        page_result: PageResult[XhsBlogEntity] = state["api_resp"]

        # 更新进度 - 生成结果
        self._push_progress(state, 5)

        path = "xhs-note-monitor" if is_monitor_query else "xhs-note-all"
        path_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="输出参数",
            status="RUNNING",
            content_type=8,
            content=CustomDataContent(data={"entity_type": 1, "content": path}),
            create_ts=int(round(time.time() * 1000)),
        )
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            path_message.model_dump_json(),
        )

        request_path = "/user/monitor/xhs-user/note-list" if is_monitor_query else "/notes/common-search"
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
                    request_body=api_request.model_dump_json(by_alias=True, exclude_none=True),
                    actions=["view", "export", "download"],
                    title=param_result.title,
                    entity_type=8,
                )
            ),
            create_ts=int(round(time.time() * 1000)),
        )
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            parameter_message.model_dump_json(),
        )

        entity_simple_info_list = [{"帖子id": entity.note_id} for entity in page_result.result_list]
        return {
            "has_query_result": True,
            "entity_simple_data": entity_simple_info_list,
        }

    def _first_query_no_result_node(self, state: ZxhXhsWorkflowState) -> Dict[str, Any]:
        """首次查询无结果分支 - 兜底查询"""
        req = state["request"]
        param_result: ZxhXhsParseParam = state["param_result"]
        origin_request = state["api_request"]
        is_monitor_query: bool = state["is_monitor_query"]

        # 精简筛选项进行兜底查询
        if is_monitor_query:
            simple_request = XhsNoteMonitorRequest(
                pageNo=origin_request.page_no,
                pageSize=origin_request.page_size,
                minPublishTime=origin_request.min_publish_time,
                maxPublishTime=origin_request.max_publish_time,
                rootSeoList=origin_request.root_seo_list,
                secondSeoList=origin_request.second_seo_list,
                industry=origin_request.industry,
                metricDate=origin_request.metric_date,
                sortField=origin_request.sort_field,
                sortOrder=origin_request.sort_order,
                limit=origin_request.limit,
                groupIdList=origin_request.group_id_list,
                monitorType=origin_request.monitor_type,
            )
            page_result = zxh_api_client.note_monitor_search(
                user_id=req.user_id, team_id=req.team_id, params=simple_request
            )
        else:
            simple_request = XhsNoteSearchRequest(
                pageNo=origin_request.page_no,
                pageSize=origin_request.page_size,
                minPublishTime=origin_request.min_publish_time,
                maxPublishTime=origin_request.max_publish_time,
                rootSeoList=origin_request.root_seo_list,
                secondSeoList=origin_request.second_seo_list,
                industry=origin_request.industry,
                metricDate=origin_request.metric_date,
                sortField=origin_request.sort_field,
                sortOrder=origin_request.sort_order,
                limit=origin_request.limit,
            )
            page_result = zxh_api_client.common_search(
                user_id=req.user_id, team_id=req.team_id, params=simple_request
            )

        result_count = page_result.result_count
        if result_count is None:
            result_count = len(page_result.result_list)
        has_query_result = (result_count or 0) > 0

        if has_query_result:
            # 更新进度 - 生成结果
            self._push_progress(state, 5)

            path = "xhs-note-monitor" if is_monitor_query else "xhs-note-all"
            path_message = BaseRedisMessage(
                session_id=req.session_id,
                reply_message_id=req.message_id,
                reply_id=f"reply_{req.message_id}",
                reply_seq=0,
                operate_id="输出参数",
                status="RUNNING",
                content_type=8,
                content=CustomDataContent(data={"entity_type": 1, "content": path}),
                create_ts=int(round(time.time() * 1000)),
            )
            redis_client.list_left_push(
                RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
                path_message.model_dump_json(),
            )

            request_path = "/user/monitor/xhs-user/note-list" if is_monitor_query else "/notes/common-search"
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
                        request_body=simple_request.model_dump_json(by_alias=True, exclude_none=True),
                        actions=["view", "export", "download"],
                        title=param_result.title,
                        entity_type=8,
                    )
                ),
                create_ts=int(round(time.time() * 1000)),
            )
            redis_client.list_left_push(
                RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
                parameter_message.model_dump_json(),
            )

            entity_simple_info_list = [{"帖子id": entity.note_id} for entity in page_result.result_list]
            return {
                "has_query_result": True,
                "entity_simple_data": entity_simple_info_list,
            }
        else:
            # # 推送无结果消息
            # no_result_message = BaseRedisMessage(
            #     session_id=req.session_id,
            #     reply_message_id=req.message_id,
            #     reply_id=f"reply_{req.message_id}",
            #     reply_seq=0,
            #     operate_id="结果",
            #     status="END",
            #     content_type=1,
            #     content=TextMessageContent(text="未找到符合需求的笔记，请尝试调整筛选条件。"),
            #     create_ts=int(round(time.time() * 1000)),
            # )
            # redis_client.list_left_push(
            #     RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            #     no_result_message.model_dump_json(),
            # )

            return {
                "has_query_result": False,
            }

    def _package_result_node(self, state: ZxhXhsWorkflowState) -> Dict[str, Any]:
        """封装返回结果节点"""
        has_query_result = state.get("has_query_result", False)
        entity_data = state.get("entity_simple_data", [])

        if has_query_result:
            entity_dicts = [e.model_dump() for e in entity_data]
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


__all__ = ["ZxhXhsGraph"]
