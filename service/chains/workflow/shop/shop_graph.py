# -*- coding: utf-8 -*-
"""
店铺排行工作流 - 对齐 n8n v2.0.2 店铺列表查询
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import text

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
from app.schemas.entities.workflow.graph_state import ShopWorkflowState
from app.schemas.request.workflow_request import WorkflowRequest
from app.schemas.response.workflow_response import WorkflowResponse
from app.service.chains.workflow.base_graph import BaseWorkflowGraph


class ShopCategoryTimeResult(BaseModel):
    """类目/时间解析结果，对齐 n8n 类目时间解析节点输出字段。"""
    model_config = ConfigDict(populate_by_name=True)

    # 统计时间区间（销量/销售额）
    start_date: str = Field(default="", alias="startDate")
    end_date: str = Field(default="", alias="endDate")
    # 上架时间区间（可为空）
    sale_start_date: str = Field(default="", alias="saleStartDate")
    sale_end_date: str = Field(default="", alias="saleEndDate")
    # 类目与根类目
    category_id: List[int] = Field(default_factory=list, alias="category_id")
    root_category_id: Optional[int] = Field(default=None, alias="root_category_id")
    category_name: List[str] = Field(default_factory=list, alias="category_name")
    root_category_name: str = Field(default="", alias="root_category_name")
    # 业务标记与用户偏好
    flag: int = Field(default=2, alias="flag")
    user_data: int = Field(default=0, alias="user_data")
    # 任务标题
    title: str = Field(default="", alias="title")

    @field_validator("category_id", mode="before")
    @classmethod
    def _normalize_category_id(cls, value: Any) -> List[int]:
        # 支持 list/JSON 字符串/逗号分隔字符串，统一转为 int 列表
        if value is None:
            return []
        if isinstance(value, list):
            return [int(v) for v in value if v is not None and str(v).strip()]
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [int(v) for v in parsed if v is not None and str(v).strip()]
            except Exception:
                pass
            cleaned = value.strip().strip("[]")
            if not cleaned:
                return []
            return [int(v) for v in cleaned.split(",") if v.strip().isdigit()]
        return []

    @field_validator("category_name", mode="before")
    @classmethod
    def _normalize_category_name(cls, value: Any) -> List[str]:
        # 支持 list/JSON 字符串/单字符串，统一转为字符串列表
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value if v is not None]
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed if v is not None]
            except Exception:
                pass
            return [value]
        return []

    @field_validator("root_category_id", mode="before")
    @classmethod
    def _normalize_root_category_id(cls, value: Any) -> Optional[int]:
        # 允许 null/空字符串，其他值尽量转为 int
        if value in (None, "", "null"):
            return None
        try:
            return int(value)
        except Exception:
            return None


class ShopPlatformTypeResult(BaseModel):
    """平台类型解析结果：淘宝/天猫/全部。"""
    model_config = ConfigDict(populate_by_name=True)

    # 店铺类型：0=淘宝，1=天猫，None=全部
    shop_type: Optional[int] = Field(default=None, alias="shopType")

    @field_validator("shop_type", mode="before")
    @classmethod
    def _normalize_shop_type(cls, value: Any) -> Optional[int]:
        if value in (None, "", "null"):
            return None
        try:
            return int(value)
        except Exception:
            return None


class ShopLabelTypeResult(BaseModel):
    """店铺标签类型解析结果。"""
    model_config = ConfigDict(populate_by_name=True)

    # 标签类型（可能为空）
    label_type: Optional[str] = Field(default=None, alias="labelType")

    @field_validator("label_type", mode="before")
    @classmethod
    def _normalize_label_type(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        return str(value)


class ShopSortResult(BaseModel):
    """排序字段解析结果。"""
    model_config = ConfigDict(populate_by_name=True)

    # 排序字段与展示名
    sort_field: str = Field(default="", alias="sortField_new")
    sort_field_name: str = Field(default="", alias="sortField_new_name")


class ShopStyleResult(BaseModel):
    """风格标签解析结果。"""
    model_config = ConfigDict(populate_by_name=True)

    # 风格标签列表
    label_style: Optional[List[str]] = Field(default=None, alias="labelStyle")

    @field_validator("label_style", mode="before")
    @classmethod
    def _normalize_label_style(cls, value: Any) -> Optional[List[str]]:
        if value in (None, "", "null"):
            return None
        if isinstance(value, list):
            return [str(v) for v in value if v is not None]
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(v) for v in parsed if v is not None]
            except Exception:
                pass
            return [value]
        return None


class ShopSearchKeyResult(BaseModel):
    """搜索关键词解析结果。"""
    model_config = ConfigDict(populate_by_name=True)

    # 关键词（可为空）
    search_key: Optional[str] = Field(default=None, alias="search_key")

    @field_validator("search_key", mode="before")
    @classmethod
    def _normalize_search_key(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        return str(value)


class ShopGraph(BaseWorkflowGraph):
    """店铺排行工作流"""

    span_name = "店铺排行工作流"
    run_name = "shop-graph"

    def _build_graph(self) -> CompiledStateGraph:
        """构建店铺工作流图。"""
        # 构建固定顺序的两节点图：执行->封装
        graph = StateGraph(ShopWorkflowState)

        graph.add_node("run_shop", self._run_shop_node)
        graph.add_node("package", self._package_result_node)

        graph.set_entry_point("run_shop")
        graph.add_edge("run_shop", "package")
        graph.add_edge("package", END)

        return graph.compile()

    def _llm_candidates(self) -> list[tuple[str, str]]:
        """返回解析用模型清单。"""
        # 店铺解析与思维链统一用 qwen3-max
        return [
            (LlmProvider.DASHSCOPE.name, LlmModelName.DASHSCOPE_QWEN3_MAX.value),
        ]

    def _build_main_prompt_variables(self, req: WorkflowRequest, industry_name: str) -> Dict[str, Any]:
        """构建思维链提示词变量。"""
        # 思维链提示词变量仅保留 n8n 对齐字段
        anchor = datetime.now() - timedelta(days=2)
        industry = industry_name or (req.industry or "").split("#")[0]
        return {
            "user_query": req.user_query or "",
            "preferred_entity": req.preferred_entity or "",
            "industry": industry,
            "current_date": anchor.strftime("%Y-%m-%d"),
        }

    def _push_message(self, message: BaseRedisMessage) -> None:
        """推送消息到 Redis 队列。"""
        # 所有消息统一入队到 redis 队列
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            message.model_dump_json(),
        )

    def _send_start_message(self, req: WorkflowRequest, cost_id: str) -> None:
        """发送收到任务消息。"""
        # n8n: 收到任务消息
        start_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="收到任务",
            status="RUNNING",
            content_type=2,
            content=WithActionContent(
                text="收到，正在查询符合条件的店铺",
                actions=["view", "export"],
                agent="shop_search",
                cost_id=cost_id,
            ),
            create_ts=int(round(time.time() * 1000)),
        )
        self._push_message(start_message)

    def _send_thinking(self, req: WorkflowRequest, thinking_text: str) -> None:
        """发送思维链消息。"""
        # 仅在思维链存在时推送
        if not thinking_text:
            return
        sanitized = thinking_text.replace("\n\n", "\n")
        thinking_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="思维链",
            status="RUNNING",
            content_type=4,
            content=TextMessageContent(text=sanitized),
            create_ts=int(round(time.time() * 1000)),
        )
        self._push_message(thinking_message)

    def _send_searching_messages(self, req: WorkflowRequest) -> None:
        """发送检索中与标签汇总消息。"""
        # n8n: 正在检索中 + 标签汇总
        searching_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="正在检索中",
            status="RUNNING",
            content_type=1,
            content=TextMessageContent(text="正在筛选店铺..."),
            create_ts=int(round(time.time() * 1000)),
        )
        self._push_message(searching_message)

        summary_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="标签汇总",
            status="RUNNING",
            content_type=1,
            content=TextMessageContent(text=f"{req.preferred_entity}..."),
            create_ts=int(round(time.time() * 1000)),
        )
        self._push_message(summary_message)

    def _send_success_messages(self, req: WorkflowRequest, request_body: dict, title: str) -> None:
        """发送完成与输出参数消息。"""
        # n8n: 结果 + 输出参数(8) + 输出参数(5)
        result_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="结果",
            status="RUNNING",
            content_type=1,
            content=TextMessageContent(text="选品任务已完成。"),
            create_ts=int(round(time.time() * 1000)),
        )
        self._push_message(result_message)

        datasource_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="输出参数",
            status="RUNNING",
            content_type=8,
            content=CustomDataContent(data={"entity_type": 1, "content": "tb-shop-rank"}),
            create_ts=int(round(time.time() * 1000)),
        )
        self._push_message(datasource_message)

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
                    request_body=json.dumps(request_body, ensure_ascii=False, separators=(",", ":")),
                    actions=["view", "export"],
                    title=title,
                    entity_type=9,
                    filters=None,
                )
            ),
            create_ts=int(round(time.time() * 1000)),
        )
        self._push_message(parameter_message)

    def _send_failure_message(self, req: WorkflowRequest, cost_id: str) -> None:
        """发送任务失败消息。"""
        # 失败时推送任务失败消息
        failure_message = BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id="任务失败",
            status="RUNNING",
            content_type=10,
            content=WithActionContent(text="任务失败", cost_id=cost_id),
            create_ts=int(round(time.time() * 1000)),
        )
        self._push_message(failure_message)

    def _insert_track(self, req: WorkflowRequest) -> None:
        """写入/更新 query 追踪记录。"""
        # n8n: holo_zhiyi_aiagent_query_prod 按 query upsert
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
                        message_id = :message_id,
                        user_id = :user_id,
                        team_id = :team_id
                    WHERE query = :query
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

    def _insert_llm_track(self, req: WorkflowRequest, llm_parameters: str) -> None:
        """写入/更新 LLM 参数追踪记录。"""
        # n8n: 追加 llm_parameters 字段
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
                        message_id = :message_id,
                        user_id = :user_id,
                        team_id = :team_id,
                        llm_parameters = :llm_parameters
                    WHERE query = :query
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

    def _load_activity_table(self) -> List[dict[str, Any]]:
        """加载活动维表。"""
        # 活动维表缺失时返回空列表（不影响流程）
        try:
            with mysql_session_readonly(DBAlias.OLAP_ZXY_AGENT) as session:
                result = session.execute(
                    text("select activity_name,start_time,end_time from tb_activity_map where status =1;")
                )
                return [dict(row) for row in result.mappings().all()]
        except Exception as e:
            logger.warning(f"[店铺排行] 查询活动维表失败: {e}")
            return []

    def _load_category_table(self) -> List[dict[str, Any]]:
        """加载类目维表。"""
        # 类目维表，表名可由配置覆盖
        try:
            table_name = getattr(settings, "taobao_category_map_table", None) or "zxy_taobao_category_map_flat"
            with mysql_session_readonly(DBAlias.OLAP_ZXY_AGENT) as session:
                result = session.execute(
                    text(
                        f"""
                        select root_category_short_name, root_category_id,category_name,category_id
                        from {table_name}
                        where `status`=1 and category_name is not null;
                    """
                    )
                )
                return [dict(row) for row in result.mappings().all()]
        except Exception as e:
            logger.warning(f"[店铺排行] 查询品类维表失败: {e}")
            return []

    def _invoke_thinking(self, req: WorkflowRequest, industry_name: str) -> str:
        """调用思维链模型生成文本。"""
        # 思维链：用于生成“思维链”消息文本，不参与后续参数解析
        prompt_vars = self._build_main_prompt_variables(req, industry_name)
        messages = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.SHOP_TASK_THINKING_PROMPT.value,
            variables=prompt_vars,
        )

        last_error: Exception | None = None
        for provider, model in self._llm_candidates():
            try:
                llm: BaseChatModel = llm_factory.get_llm(provider, model)
                retry_llm = llm.with_retry(stop_after_attempt=2)
                thinking_chain = retry_llm | StrOutputParser()
                thinking_text = thinking_chain.with_config(run_name="思维链生成").invoke(messages)
                return (thinking_text or "").replace("\n\n", "\n").strip()
            except Exception as e:
                last_error = e
                logger.warning(f"[店铺排行] 思考链模型 {model} 失败，尝试下一模型: {e}")
        if last_error:
            logger.warning(f"[店铺排行] 思考链生成失败: {last_error}")
        return ""

    def _invoke_category_time_parse(
        self,
        user_input: str,
        user_preferences: str,
        industry_name: str,
        category_table: List[dict[str, Any]],
        activity_table: List[dict[str, Any]],
    ) -> ShopCategoryTimeResult:
        """解析时间与类目信息。"""
        # 参数解析：根据用户输入 + 类目/活动维表输出起止时间、类目、标题等
        now = datetime.now()
        current_date = (now - timedelta(days=2)).strftime("%Y-%m-%d")
        default_start = (now - timedelta(days=32)).strftime("%Y-%m-%d")
        default_end = current_date

        category_table_json = json.dumps(category_table, ensure_ascii=False, separators=(",", ":"))
        activity_table_json = json.dumps(activity_table, ensure_ascii=False, separators=(",", ":"))

        last_error: Exception | None = None
        for provider, model in self._llm_candidates():
            try:
                # 传入提示词变量，确保与 n8n 参数一致
                prompt_vars = {
                    "user_input": user_input,
                    "user_preferences": user_preferences,
                    "industry_name": industry_name,
                    "category_table_json": category_table_json,
                    "current_date": current_date,
                    "default_start": default_start,
                    "default_end": default_end,
                    "activity_table_json": activity_table_json,
                }
                messages = coze_loop_client_provider.get_langchain_messages(
                    prompt_key=CozePromptHubKey.SHOP_CATEGORY_TIME_PARSE_PROMPT.value,
                    variables=prompt_vars,
                )
                return self._invoke_structured(
                    provider, model, ShopCategoryTimeResult, messages, "参数解析"
                )
            except Exception as e:
                last_error = e
                logger.warning(f"[店铺排行] 参数解析模型 {model} 失败，尝试下一模型: {e}")
        if last_error:
            logger.warning(f"[店铺排行] 参数解析失败: {last_error}")
        return ShopCategoryTimeResult()

    def _invoke_platform_type(self, query: str) -> ShopPlatformTypeResult:
        """解析平台类型。"""
        # 平台类型解析
        last_error: Exception | None = None
        for provider, model in self._llm_candidates():
            try:
                messages = coze_loop_client_provider.get_langchain_messages(
                    prompt_key=CozePromptHubKey.SHOP_PLATFORM_TYPE_PROMPT.value,
                    variables={"query": query},
                )
                return self._invoke_structured(
                    provider, model, ShopPlatformTypeResult, messages, "平台解析"
                )
            except Exception as e:
                last_error = e
                logger.warning(f"[店铺排行] 平台解析模型 {model} 失败，尝试下一模型: {e}")
        if last_error:
            logger.warning(f"[店铺排行] 平台解析失败: {last_error}")
        return ShopPlatformTypeResult()

    def _invoke_label_type(self, query: str) -> ShopLabelTypeResult:
        """解析标签类型。"""
        # 标签类型解析
        last_error: Exception | None = None
        for provider, model in self._llm_candidates():
            try:
                messages = coze_loop_client_provider.get_langchain_messages(
                    prompt_key=CozePromptHubKey.SHOP_LABEL_TYPE_PROMPT.value,
                    variables={"query": query},
                )
                return self._invoke_structured(
                    provider, model, ShopLabelTypeResult, messages, "标签类型解析"
                )
            except Exception as e:
                last_error = e
                logger.warning(f"[店铺排行] 标签类型解析模型 {model} 失败，尝试下一模型: {e}")
        if last_error:
            logger.warning(f"[店铺排行] 标签类型解析失败: {last_error}")
        return ShopLabelTypeResult()

    def _invoke_sort(self, query: str) -> ShopSortResult:
        """解析排序字段。"""
        # 排序字段解析
        last_error: Exception | None = None
        for provider, model in self._llm_candidates():
            try:
                messages = coze_loop_client_provider.get_langchain_messages(
                    prompt_key=CozePromptHubKey.SHOP_SORT_PROMPT.value,
                    variables={"query": query},
                )
                return self._invoke_structured(
                    provider, model, ShopSortResult, messages, "排序解析"
                )
            except Exception as e:
                last_error = e
                logger.warning(f"[店铺排行] 排序解析模型 {model} 失败，尝试下一模型: {e}")
        if last_error:
            logger.warning(f"[店铺排行] 排序解析失败: {last_error}")
        return ShopSortResult()

    def _invoke_style(self, query: str, root_category_id: Optional[int], root_category_name: str) -> ShopStyleResult:
        """解析风格标签。"""
        # 风格标签解析（依赖根类目）
        last_error: Exception | None = None
        for provider, model in self._llm_candidates():
            try:
                messages = coze_loop_client_provider.get_langchain_messages(
                    prompt_key=CozePromptHubKey.SHOP_STYLE_PROMPT.value,
                    variables={
                        "query": query,
                        "root_category_id": "" if root_category_id is None else str(root_category_id),
                        "root_category_name": root_category_name,
                    },
                )
                return self._invoke_structured(
                    provider, model, ShopStyleResult, messages, "风格解析"
                )
            except Exception as e:
                last_error = e
                logger.warning(f"[店铺排行] 风格解析模型 {model} 失败，尝试下一模型: {e}")
        if last_error:
            logger.warning(f"[店铺排行] 风格解析失败: {last_error}")
        return ShopStyleResult()

    def _invoke_search_key(self, query: str) -> ShopSearchKeyResult:
        """解析搜索关键词。"""
        # 关键词解析
        last_error: Exception | None = None
        for provider, model in self._llm_candidates():
            try:
                messages = coze_loop_client_provider.get_langchain_messages(
                    prompt_key=CozePromptHubKey.SHOP_SEARCH_KEY_PROMPT.value,
                    variables={"query": query},
                )
                return self._invoke_structured(
                    provider, model, ShopSearchKeyResult, messages, "关键词解析"
                )
            except Exception as e:
                last_error = e
                logger.warning(f"[店铺排行] 关键词解析模型 {model} 失败，尝试下一模型: {e}")
        if last_error:
            logger.warning(f"[店铺排行] 关键词解析失败: {last_error}")
        return ShopSearchKeyResult()

    def _invoke_structured(
        self,
        provider: str,
        model: str,
        schema: type[BaseModel],
        messages: list[Any],
        run_name: str,
    ) -> BaseModel:
        """调用结构化输出模型（对齐知衣的 qwen3 调用方式）。"""
        # 直接使用结构化输出 + retry，不做自定义降级解析
        llm: BaseChatModel = llm_factory.get_llm(provider, model)
        if provider == LlmProvider.DASHSCOPE.name:
            llm = llm.bind(response_format={"type": "json_object"})
        structured_llm = llm.with_structured_output(schema).with_retry(stop_after_attempt=2)
        try:
            result = structured_llm.with_config(run_name=run_name).invoke(messages)
            if result is None:
                logger.warning(
                    f"[店铺排行] LLM 返回 None: run={run_name}, provider={provider}, model={model}"
                )
                raise ValueError("LLM returned None")
            return result
        except Exception as e:
            preview = [getattr(msg, "content", str(msg)) for msg in messages[:2]]
            logger.warning(
                "[店铺排行] LLM 调用失败: run=%s provider=%s model=%s error=%s preview=%s",
                run_name,
                provider,
                model,
                e,
                preview,
            )
            raise

    def _build_request_body(
        self,
        platform_type: Optional[int],
        label_type: Optional[str],
        style_labels: Optional[List[str]],
        sort_field: str,
        start_date: str,
        end_date: str,
        root_category_id: Optional[int],
        category_id_list: Optional[List[int]],
        search_key: Optional[str],
        include_category_list: bool,
    ) -> dict[str, Any]:
        """构建店铺 API 请求体。"""
        # n8n 对齐：timeFlag 固定为 "2"，rootCategoryIdList 允许 [null]
        # 字段顺序保持与 n8n params1 一致（影响前端展示/对比）
        payload: dict[str, Any] = {
            "labelType": label_type,
            "shopType": platform_type,
            "labelStyleList": style_labels,
            "rankStatus": sort_field,
            "startDate": start_date,
            "endDate": end_date,
            "timeFlag": "2",
            "limit": 500,
            "pageNo": 1,
            "pageSize": 20,
            "rootCategoryIdList": [root_category_id] if root_category_id is not None else [None],
        }
        if include_category_list:
            payload["categoryIdList"] = category_id_list or []
        payload["mainType"] = 1
        payload["groupIdList"] = []
        if search_key:
            payload["searchkey"] = search_key
        payload["entrance"] = "2"
        return payload

    def _call_shop_api(self, req: WorkflowRequest, payload: dict[str, Any]) -> dict[str, Any]:
        """调用店铺排行 API。"""
        # API 基址从配置读取，重试 3 次，header 与 n8n 一致
        base_url = (settings.zhiyi_api_url or "").rstrip("/")
        if not base_url:
            raise ValueError("zhiyi_api_url 未配置")
        url = f"{base_url}/v2-5-6/rank/shop-list-v3"
        headers = {
            "USER-ID": str(req.user_id),
            "TEAM-ID": str(req.team_id),
            "Content-Type": "application/json",
        }
        logger.debug(f"[店铺排行] 请求 shop-list-v3: {url}")
        for attempt in range(3):
            try:
                response = requests.post(
                    url=url,
                    headers=headers,
                    json=payload,
                    timeout=settings.zhiyi_api_timeout,
                )
                response.raise_for_status()
                return response.json()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(0.5)

    def _has_result(self, response: dict[str, Any]) -> bool:
        """判断 API 是否有结果。"""
        # 仅 success 且 resultCount>0 判定为有结果
        if not response or not response.get("success"):
            return False
        result = response.get("result") or {}
        try:
            return int(result.get("resultCount", 0)) > 0
        except Exception:
            return False

    def _collect_relate_data(self, response: dict[str, Any]) -> str:
        """整理结果列表并序列化为 JSON。"""
        # n8n: 结果列表追加“店铺id”字段，输出为 JSON 字符串
        result = response.get("result") or {}
        result_list = result.get("resultList") or []
        items: list[dict[str, Any]] = []
        for item in result_list:
            if not isinstance(item, dict):
                continue
            entry = dict(item)
            shop_id = entry.get("shopId")
            if shop_id is not None:
                entry["店铺id"] = str(shop_id)
            items.append(entry)
        return json.dumps(items, ensure_ascii=False, separators=(",", ":"))

    def _run_shop_node(self, state: ShopWorkflowState) -> Dict[str, Any]:
        """主节点：解析参数、请求 API、封装返回。"""
        req: WorkflowRequest = state["request"]
        cost_id = f"{req.session_id}_{int(time.time() * 1000)}"

        try:
            # 1) 发送开始/思维链/检索中消息
            self._send_start_message(req, cost_id)

            industry_name = req.industry.split("#")[0] if req.industry else ""
            thinking_text = self._invoke_thinking(req, industry_name)
            self._send_thinking(req, thinking_text)
            self._send_searching_messages(req)

            # 2) 拼接用户输入并写入追踪表
            user_question = req.user_query or ""
            chat_input = getattr(req, "chat_input", None)
            user_input = f"{chat_input or ''}{user_question}"
            self._insert_track(req)

            # 3) 加载类目/活动维表
            activity_table = self._load_activity_table()
            category_table = self._load_category_table()

            # 4) 结构化解析参数（品类/时间/平台/标签/排序/风格/关键词）
            category_result = self._invoke_category_time_parse(
                user_input=user_input,
                user_preferences=req.user_preferences or "",
                industry_name=industry_name,
                category_table=category_table,
                activity_table=activity_table,
            )

            query_text = user_input.replace("\n", "\\n")
            platform_result = self._invoke_platform_type(query_text)
            label_result = self._invoke_label_type(query_text)
            sort_result = self._invoke_sort(query_text)
            style_result = self._invoke_style(
                query_text,
                category_result.root_category_id,
                category_result.root_category_name,
            )
            search_result = self._invoke_search_key(query_text)

            # 5) 记录 LLM 参数到追踪表
            params_list = [
                {"output": {"shopType": platform_result.shop_type}},
                {"output": {"labelType": label_result.label_type}},
                {
                    "output": {
                        "sortField_new": sort_result.sort_field or "saleVolume",
                        "sortField_new_name": sort_result.sort_field_name or "销量",
                    }
                },
                {"output": {"labelStyle": style_result.label_style}},
                {
                    "output": {
                        "startDate": category_result.start_date,
                        "endDate": category_result.end_date,
                        "saleStartDate": category_result.sale_start_date,
                        "saleEndDate": category_result.sale_end_date,
                        "category_id": category_result.category_id,
                        "root_category_id": category_result.root_category_id,
                        "category_name": category_result.category_name,
                        "root_category_name": category_result.root_category_name,
                        "flag": category_result.flag,
                        "user_data": category_result.user_data,
                        "title": category_result.title,
                        "userid": req.user_id,
                        "teamid": req.team_id,
                    }
                },
                {"output": {"search_key": search_result.search_key}},
            ]
            llm_parameters = json.dumps(params_list, ensure_ascii=False, separators=(",", ":"))
            self._insert_llm_track(req, llm_parameters)

            # 6) 首次请求（含类目）
            sort_field = sort_result.sort_field or "saleVolume"
            primary_request = self._build_request_body(
                platform_type=platform_result.shop_type,
                label_type=label_result.label_type,
                style_labels=style_result.label_style,
                sort_field=sort_field,
                start_date=category_result.start_date,
                end_date=category_result.end_date,
                root_category_id=category_result.root_category_id,
                category_id_list=category_result.category_id,
                search_key=search_result.search_key,
                include_category_list=True,
            )

            response = self._call_shop_api(req, primary_request)
            if self._has_result(response):
                self._send_success_messages(req, primary_request, category_result.title)
                relate_data = self._collect_relate_data(response)
                return {
                    "output_text": "基于以上条件，您的选品任务已经完成，还有其他需要帮助的地方吗？",
                    "relate_data": relate_data,
                }

            # 7) 兜底请求（不含类目）
            fallback_request = self._build_request_body(
                platform_type=platform_result.shop_type,
                label_type=label_result.label_type,
                style_labels=style_result.label_style,
                sort_field=sort_field,
                start_date=category_result.start_date,
                end_date=category_result.end_date,
                root_category_id=category_result.root_category_id,
                category_id_list=category_result.category_id,
                search_key=search_result.search_key,
                include_category_list=False,
            )
            response = self._call_shop_api(req, fallback_request)
            if self._has_result(response):
                self._send_success_messages(req, fallback_request, category_result.title)
                relate_data = self._collect_relate_data(response)
                return {
                    "output_text": "基于以上条件，您的选品任务已经完成，还有其他需要帮助的地方吗？",
                    "relate_data": relate_data,
                }

            return {"output_text": "无结果，可能与店铺风格、平台、筛选时间有关", "relate_data": None}
        except Exception as e:
            logger.warning(f"[店铺排行] 执行失败: {e}")
            self._send_failure_message(req, cost_id)
            return {"output_text": "寻找店铺失败", "relate_data": None}

    def _package_result_node(self, state: ShopWorkflowState) -> Dict[str, Any]:
        """封装 workflow_response。"""
        output_text = state.get("output_text", "")
        relate_data = state.get("relate_data")
        return {"workflow_response": WorkflowResponse(select_result=output_text, relate_data=relate_data)}


__all__ = ["ShopGraph"]
