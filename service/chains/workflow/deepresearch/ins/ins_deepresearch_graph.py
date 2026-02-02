# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2026/1/22 21:16
# @File     : ins_deepresearch_graph.py
"""
INS数据洞察深度思考工作流
"""
from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum

from dateutil.relativedelta import relativedelta
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph.state import CompiledStateGraph, StateGraph

from app.core.clients.coze_loop_client import coze_loop_client_provider
from app.core.config.constants import LlmProvider, LlmModelName, CozePromptHubKey, WorkflowEntityType
from app.core.tools import llm_factory
from app.service.chains.workflow.base_graph import BaseWorkflowGraph
from app.service.chains.workflow.deepresearch.deepresearch_message_pusher import DeepresearchMessagePusher
from app.service.chains.workflow.deepresearch.ins.llm_output import InsApiParseParam, InsDeepresearchState


class NodeKey(StrEnum):
    """INS工作流节点枚举"""
    # 通用准备
    INIT = "初始化"
    API_PARAM_PARSE = "API参数解析"

    # 报告生成
    GENERATE_REPORT = "报告生成"

    # 结果输出
    ASSEMBLE_OUTPUT = "组装输出"


class InsDeepresearchGraph(BaseWorkflowGraph):
    """
    数据洞察-INS数据源-深度思考工作流
    """
    span_name: str = "INS数据洞察工作流"

    def __init__(self):
        super().__init__()

    def _build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(InsDeepresearchState)

        # =============定义节点=============
        graph.add_node(NodeKey.INIT, self._init_state_node)
        graph.add_node(NodeKey.API_PARAM_PARSE, self._api_param_parse_node)
        graph.add_node(NodeKey.GENERATE_REPORT, self._generate_report_node)
        graph.add_node(NodeKey.ASSEMBLE_OUTPUT, self._assemble_output_node)

        # =============定义入口和边=============
        graph.set_entry_point(NodeKey.INIT)

        # 线性工作流: INIT → API_PARAM_PARSE → GENERATE_REPORT → ASSEMBLE_OUTPUT
        graph.add_edge(NodeKey.INIT, NodeKey.API_PARAM_PARSE)
        graph.add_edge(NodeKey.API_PARAM_PARSE, NodeKey.GENERATE_REPORT)
        graph.add_edge(NodeKey.GENERATE_REPORT, NodeKey.ASSEMBLE_OUTPUT)

        graph.set_finish_point(NodeKey.ASSEMBLE_OUTPUT)

        return graph.compile()

    # ===节点实现===

    def _init_state_node(self, state: InsDeepresearchState):
        """开始节点，处理初始化"""
        req = state.request
        # ins数据洞察暂时没有深度思考，均按非深度处理即可
        is_thinking = req.thinking if req.thinking else False

        # 创建消息推送器
        pusher = DeepresearchMessagePusher(
            request=req,
            is_thinking=is_thinking,
            workflow_type="ins",
        )

        pusher.push_task_start_msg()
        pusher.push_phase("正在启用趋势分析助手")

        return {
            "is_thinking": is_thinking,
            "message_pusher": pusher,
        }

    def _api_param_parse_node(self, state: InsDeepresearchState):
        """API参数解析节点 - 解析日期参数"""
        req = state.request
        is_thinking = state.is_thinking
        pusher = state.message_pusher

        # 获取LLM
        llm = llm_factory.get_llm(LlmProvider.HUANXIN.name, LlmModelName.HUANXIN_GEMINI_3_FLASH_PREVIEW.value)
        struct_llm = llm.with_structured_output(schema=InsApiParseParam).with_retry(stop_after_attempt=2)

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
            "recent_30day_end_date": t_minus_1.strftime("%Y-%m-%d"),
            "user_query": req.user_query,
            "industry": req.industry,
        }

        messages: list[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.DR_INS_MAIN_PARAM_PARSE_PROMPT.value,
            variables=invoke_params,
        )

        # 执行LLM解析参数
        parse_param: InsApiParseParam = struct_llm.with_config(run_name="解析API参数").invoke(messages)

        return {
            "api_parse_param": parse_param
        }

    def _generate_report_node(self, state: InsDeepresearchState):
        """生成分析报告节点"""
        req = state.request
        param = state.api_parse_param
        pusher = state.message_pusher

        if not param:
            return {"report_text": "参数解析失败，无法生成报告。"}

        pusher.push_phase("正在生成数据洞察报告")

        # 获取LLM
        llm = llm_factory.get_llm(LlmProvider.HUANXIN.name, LlmModelName.HUANXIN_GEMINI_3_FLASH_PREVIEW.value)
        chain = llm.with_retry(stop_after_attempt=2) | StrOutputParser()

        # 准备报告生成参数
        invoke_params = {
            "date_range": f"{param.start_date}至{param.end_date}",
            "user_query": req.user_query,
        }

        messages: list[BaseMessage] = coze_loop_client_provider.get_langchain_messages(
            prompt_key=CozePromptHubKey.DR_INS_REPORT_GENERATE_PROMPT.value,
            variables=invoke_params,
        )

        # 执行LLM生成报告
        report_text = chain.with_config(run_name="生成INS分析报告").invoke(messages)

        pusher.push_phase("报告已生成")

        return {"report_text": report_text}

    def _assemble_output_node(self, state: InsDeepresearchState):
        """组装最终输出"""
        report_text = state.report_text
        pusher = state.message_pusher

        # 推送报告（INS报告无Excel导出）
        pusher.push_report_and_excel_data(
            entity_type=None,
            report_text=report_text or "",
            excel_data_list=[]
        )

        pusher.push_task_finish_status_msg()

        return {}


ins_deepresearch_graph = InsDeepresearchGraph()


__all__ = [
    "InsDeepresearchGraph",
    "ins_deepresearch_graph",
]
