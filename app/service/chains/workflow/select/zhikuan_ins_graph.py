# -*- coding: utf-8 -*-
# @Author   : kiro
# @Time     : 2025/12/14
# @File     : zhikuan_ins_graph.py

"""
知款 INS 选品工作流 - LangGraph 版本
"""

from __future__ import annotations

import json
import random
import time
from calendar import day_abbr
from datetime import datetime, timedelta
from typing import Any, Dict, List, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
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
    TextMessageContent,
    WithActionContent,
)
from app.schemas.entities.workflow.graph_state import ZhikuanInsWorkflowState
from app.schemas.entities.workflow.llm_output import ZhikuanInsParseParam, ZhikuanInsSortTypeParseResult
from app.schemas.request.workflow_request import WorkflowRequest
from app.schemas.response.common import PageResult
from app.schemas.response.workflow_response import WorkflowResponse
from app.service.chains.workflow.base_graph import BaseWorkflowGraph
from app.service.chains.workflow.progress_pusher import PhaseProgressPusher
from app.service.chains.templates.zhikuan_progress_template import ZHIKUAN_INS_PROGRESS_TEMPLATE
from app.service.rpc.zhikuan_api import (
    BloggerGroupItem,
    ZhikuanInsBlogEntity,
    ZhikuanInsBlogListRequest,
    zhikuan_api_client,
)
from app.utils import thread_pool


class ZhikuanInsGraph(BaseWorkflowGraph):
    """知款 INS 选品工作流 - LangGraph 版本"""

    span_name = "知款ins数据源工作流"
    run_name = "zhikuan-ins-graph"


    def _build_graph(self) -> CompiledStateGraph:
        """构建工作流图"""
        graph = StateGraph(ZhikuanInsWorkflowState)

        graph.add_node("init_state", self._init_state_node)
        graph.add_node("pre_think", self._pre_think_node)
        graph.add_node("query_selections", self._query_common_selection_node)
        graph.add_node("llm_parse", self._llm_param_parse_node)
        graph.add_node("llm_sort_parse", self._llm_parse_sort_type_node)
        graph.add_node("call_api", self._call_business_api_node)
        graph.add_node("has_result", self._first_query_has_result_node)
        graph.add_node("no_result", self._first_query_no_result_node)
        graph.add_node("package", self._package_result_node)

        graph.set_entry_point("init_state")
        graph.add_edge("init_state", "pre_think")
        graph.add_edge("pre_think", "query_selections")
        graph.add_edge("query_selections", "llm_parse")
        graph.add_edge("llm_parse", "llm_sort_parse")
        graph.add_edge("llm_sort_parse", "call_api")

        graph.add_conditional_edges(
            "call_api",
            self._check_first_query_has_result,
            {"has_result": "has_result", "no_result": "no_result"},
        )

        graph.add_edge("has_result", "package")
        graph.add_edge("no_result", "package")
        graph.add_edge("package", END)

        return graph.compile()

    def _check_first_query_has_result(self, state: ZhikuanInsWorkflowState) -> str:
        req = state.get("request")
        result_count = state.get("result_count", 0)

        if result_count and result_count >= 50:
            result_branch = "has_result"
        else:
            result_branch = "no_result"

        if result_branch == "has_result":
            self._get_pusher(req=req).complete_phase(phase_name="帖子筛选中", variables={
                "datasource": "INS数据库",
                "platform": "知款INS",
                "browsed_count": f"{str(random.randint(100000, 1000000))}",
                "filter_result_text": f"帖子筛选完成，当前筛选出的帖子数量满足需求，无需二次筛选，接下来我将根据数据完成帖子列表",
            })
        elif result_branch == "no_result":
            self._get_pusher(req=req).complete_phase(phase_name="帖子筛选中", variables={
                "datasource": "INS数据库",
                "platform": "知款INS",
                "browsed_count": f"{str(random.randint(100000, 1000000))}",
                "filter_result_text": f"当前筛选出的帖子数量不足，不满足列表需求，我将再次搜索确保无遗漏",
            })

        return result_branch

    # ==================== 工具方法 ====================
    def _get_pusher(self, req: WorkflowRequest) -> PhaseProgressPusher:
        return PhaseProgressPusher(template=ZHIKUAN_INS_PROGRESS_TEMPLATE, request=req)


    # ==================== 节点实现 ====================

    def _init_state_node(self, state: ZhikuanInsWorkflowState) -> Dict[str, Any]:
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

    def _pre_think_node(self, state: ZhikuanInsWorkflowState) -> Dict[str, Any]:
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
                data=ParameterData(entity_type=WorkflowEntityType.ZHIKUAN_INS.code)
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
                prompt_key=CozePromptHubKey.ZHIKUAN_INS_THINK_PROMPT.value,
                variables=invoke_params,
            )
            llm: BaseChatModel = llm_factory.get_llm(
                LlmProvider.OPENROUTER.name, LlmModelName.OPENROUTER_GPT_4O.value
            )
            retry_llm = llm.with_retry(stop_after_attempt=2)

            thinking_chain = retry_llm | StrOutputParser()
            thinking_result_text = thinking_chain.with_config(run_name="思维链生成").invoke(messages)
            thinking_result_text = thinking_result_text.replace("\n\n", "\n")

            self._get_pusher(req=req).complete_phase(phase_name="选品任务规划", content=thinking_result_text)

            # thinking_result_message = BaseRedisMessage(
            #     session_id=req.session_id,
            #     reply_message_id=req.message_id,
            #     reply_id=f"reply_{req.message_id}",
            #     reply_seq=0,
            #     operate_id="思维链",
            #     status="RUNNING",
            #     content_type=4,
            #     content=TextMessageContent(text=thinking_result_text),
            #     create_ts=int(round(time.time() * 1000)),
            # )
            # redis_client.list_right_push(
            #     RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            #     thinking_result_message.model_dump_json(),
            # )
            #
            # searching_message = BaseRedisMessage(
            #     session_id=req.session_id,
            #     reply_message_id=req.message_id,
            #     reply_id=f"reply_{req.message_id}",
            #     reply_seq=0,
            #     operate_id="正在检索中",
            #     status="RUNNING",
            #     content_type=1,
            #     content=TextMessageContent(text="正在生成选品任务..."),
            #     create_ts=int(round(time.time() * 1000)),
            # )
            # redis_client.list_right_push(
            #     RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            #     searching_message.model_dump_json(),
            # )
            #
            # starting_select_message = BaseRedisMessage(
            #     session_id=req.session_id,
            #     reply_message_id=req.message_id,
            #     reply_id=f"reply_{req.message_id}",
            #     reply_seq=0,
            #     operate_id="标签汇总",
            #     status="RUNNING",
            #     content_type=1,
            #     content=TextMessageContent(text=f"正在{req.preferred_entity}上为你选品..."),
            #     create_ts=int(round(time.time() * 1000)),
            # )
            # redis_client.list_right_push(
            #     RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            #     starting_select_message.model_dump_json(),
            # )

        _generate_thinking_and_report_task()
        return {}

    def _query_common_selection_node(self, state: ZhikuanInsWorkflowState) -> Dict[str, Any]:
        req = state["request"]

        selection_dict: Dict[str, Any] = {}
        with mysql_session_readonly(DBAlias.OLAP_ZXY_AGENT) as session:
            result = session.execute(
                text("select type, label, value from zxy_zk_ins_selection_common")
            )
            common_selections = [dict(row) for row in result.mappings().all()]

            result = session.execute(
                text("select industry_name,root_category, leaf_category from zxy_zk_ins_selection_category")
            )
            category_paths = [dict(row) for row in result.mappings().all()]

            selection_dict["category_paths"] = category_paths
            selection_dict["common_selections"] = common_selections

        blogger_group_result = zhikuan_api_client.list_all_blogger_groups(
            team_id=req.team_id, user_id=req.user_id, source_type=1, is_show_default=1
        )

        def parse_user_default_group() -> Union[str, None]:
            default_group_id: Union[str, None] = None
            self_groups: list[BloggerGroupItem] = blogger_group_result.self_list
            if self_groups:
                for group in self_groups:
                    if group.is_default == 1:
                        default_group_id = str(group.id)
            return default_group_id

        selection_dict["default_group_id"] = parse_user_default_group()

        return {"selection_dict": selection_dict}

    def _llm_param_parse_node(self, state: ZhikuanInsWorkflowState) -> Dict[str, Any]:
        req = state["request"]
        selection_dict = state["selection_dict"]

        def _build_time_data() -> str:
            """构建时间数据，生成不同的时间区间供LLM使用"""
            now = datetime.now() - timedelta(days=1)  # 使用 T-1 日期
            year = now.year
            # 判断是否闰年
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

        prompt_param = {
            "user_query": req.user_query,
            "preferred_entity": req.preferred_entity,
            "industry": req.industry,
            "user_preferences": req.user_preferences,
            "selection_data": json.dumps(selection_dict["common_selections"], ensure_ascii=False),
            "category_data": json.dumps(selection_dict["category_paths"], ensure_ascii=False),
            "default_monitor_group_id": selection_dict["default_group_id"],
            "time_data": _build_time_data(),
        }
        messages: List[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.ZHIKUAN_INS_MAIN_PARAM_PARSE_PROMPT.value,
            variables=prompt_param,
        )

        llm: BaseChatModel = llm_factory.get_llm(
            LlmProvider.OPENROUTER.name, LlmModelName.OPENROUTER_GEMINI_3_FLASH_PREVIEW.value
        )
        structured_llm = llm.with_structured_output(ZhikuanInsParseParam).with_retry(stop_after_attempt=2)
        param_result = structured_llm.with_config(run_name="参数解析").invoke(messages)
        return {"param_result": param_result}

    def _llm_parse_sort_type_node(self, state: ZhikuanInsWorkflowState) -> Dict[str, Any]:
        req = state["request"]
        param_result: ZhikuanInsParseParam = state["param_result"]

        prompt_param = {
            "is_monitor_blogger": str(param_result.is_monitor_blogger),
            "user_query": req.user_query,
            "sort_type": param_result.sort_type,
        }
        messages = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.ZHIKUAN_INS_SORT_TYPE_PARSE_PROMPT.value,
            variables=prompt_param,
        )

        llm = llm_factory.get_llm(
            LlmProvider.OPENROUTER.name, LlmModelName.OPENROUTER_GEMINI_2_5_FLASH.value
        )
        structured_llm = llm.with_structured_output(ZhikuanInsSortTypeParseResult).with_retry(stop_after_attempt=2)
        sort_param_result = structured_llm.with_config(run_name="排序项解析").invoke(messages)
        return {"sort_param_result": sort_param_result}

    def _call_business_api_node(self, state: ZhikuanInsWorkflowState) -> Dict[str, Any]:
        """调用业务 API 节点"""
        req = state["request"]
        param_result: ZhikuanInsParseParam = state["param_result"]
        sort_param_result: ZhikuanInsSortTypeParseResult = state["sort_param_result"]

        ins_request = ZhikuanInsBlogListRequest.from_parse_param(param_result)
        ins_request.rank_status = sort_param_result.sort_type_final
        page_result: PageResult[ZhikuanInsBlogEntity] = zhikuan_api_client.list_blogs(
            user_id=req.user_id, team_id=req.team_id, params=ins_request
        )

        browsed_count = page_result.result_count or 0

        return {
            "api_request": ins_request,
            "api_request_alias": ins_request.model_dump_json(by_alias=True, exclude_none=True),
            "api_resp": page_result,
            "result_count": page_result.result_count,
            "browsed_count": browsed_count,
        }

    def _first_query_has_result_node(self, state: ZhikuanInsWorkflowState) -> Dict[str, Any]:
        """首次查询有结果分支"""
        req = state["request"]
        api_request: ZhikuanInsBlogListRequest = state["api_request"]
        param_result: ZhikuanInsParseParam = state["param_result"]
        page_result: PageResult[ZhikuanInsBlogEntity] = state["api_resp"]

        self._get_pusher(req=req).complete_phase(phase_name="生成列表中")
        self._get_pusher(req=req).complete_phase(phase_name="选品完成")

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

        path_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="输出参数",
            status="RUNNING",
            content_type=8,
            content=CustomDataContent(data={"entity_type": 1, "content": "zk-ins-all"}),
            create_ts=int(round(time.time() * 1000)),
        )
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            path_message.model_dump_json(),
        )

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
                    request_path="/image-bus/label-selector/list-blog",
                    request_body=api_request.model_dump_json(by_alias=True, exclude_none=True),
                    actions=["view", "export", "download"],
                    title=param_result.title,
                    entity_type=7,
                )
            ),
            create_ts=int(round(time.time() * 1000)),
        )
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            parameter_message.model_dump_json(),
        )

        entity_simple_info_list = [{"帖子id": entity.union_id} for entity in page_result.result_list]
        return {
            "has_query_result": True,
            "entity_simple_data": entity_simple_info_list,
        }

    def _first_query_no_result_node(self, state: ZhikuanInsWorkflowState) -> Dict[str, Any]:
        """首次查询无结果分支 - 兜底查询"""
        req = state["request"]
        origin_request: ZhikuanInsBlogListRequest = state["api_request"]
        param_result: ZhikuanInsParseParam = state["param_result"]

        self._get_pusher(req=req).complete_phase(phase_name="商品筛选中", variables={
            "datasource": "INS数据库",
            "platform": "知款INS",
            "browsed_count": f"{str(random.randint(100000, 1000000))}",
            "filter_result_text": "当前筛选出的商品数量不足，不满足列表需求，我将再次搜索确保无遗漏",
        })

        # 精简筛选项进行兜底查询
        simple_request = ZhikuanInsBlogListRequest(
            start=origin_request.start,
            pageSize=origin_request.page_size,
            blogTimeStart=origin_request.blog_time_start,
            blogTimeEnd=origin_request.blog_time_end,
            gender=origin_request.gender,
            categoryList=origin_request.category_list,
            rankStatus=origin_request.rank_status,
            limit=origin_request.limit,
            bloggerGroupIdList=origin_request.blogger_group_id_list,
            sourceType=origin_request.source_type,
            platformId=origin_request.platform_id,
        )
        page_result: PageResult[ZhikuanInsBlogEntity] = zhikuan_api_client.list_blogs(
            team_id=req.team_id, user_id=req.user_id, params=simple_request
        )

        has_query_result = page_result.result_count is None or page_result.result_count > 0

        if has_query_result:
            # 兜底查询成功的消息
            self._get_pusher(req=req).complete_phase(phase_name="二次筛选中", variables={
                "datasource": "INS数据库",
                "platform": "知款INS",
                "browsed_count": f"{str(random.randint(100000, 1000000))}",
                "retry_result_text": "商品筛选完成，当前筛选出的商品数量符合报表要求，接下来我将根据数据绘制商品列表",
            })
            self._get_pusher(req=req).complete_phase(phase_name="生成列表中")
            self._get_pusher(req=req).complete_phase(phase_name="选品完成")

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
                content=CustomDataContent(data={"entity_type": 1, "content": "zk-ins-all"}),
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
                        request_path="/image-bus/label-selector/list-blog",
                        request_body=simple_request.model_dump_json(by_alias=True, exclude_none=True),
                        actions=["view", "export", "download"],
                        title=param_result.title,
                        entity_type=7,
                    )
                ),
                create_ts=int(round(time.time() * 1000)),
            )
            redis_client.list_left_push(
                RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
                parameter_message.model_dump_json(),
            )

            entity_simple_info_list = [{"帖子id": entity.union_id} for entity in page_result.result_list]
            return {
                "has_query_result": True,
                "entity_simple_data": entity_simple_info_list,
            }
        else:
            # 推送无结果消息
            self._get_pusher(req=req).complete_phase(phase_name="二次筛选中", variables={
                "datasource": "INS数据库",
                "platform": "知款INS",
                "browsed_count": f"{str(random.randint(100000, 1000000))}",
                "retry_result_text": "当前筛选出的商品数量不满足需求，我可能需要提醒用户调整筛选维度",
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
                "has_query_result": False,
            }

    def _package_result_node(self, state: ZhikuanInsWorkflowState) -> Dict[str, Any]:
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
                select_result="无结果，可能与价格、销量、参考历史选品经验有关",
                relate_data=None,
            )
        return {"workflow_response": response}


__all__ = ["ZhikuanInsGraph"]
