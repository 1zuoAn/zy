# -*- coding: utf-8 -*-
"""定时任务工作流 - 对齐 n8n schedule agent"""
from __future__ import annotations

import json
import time
from typing import Any, Dict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.core.clients.redis_client import redis_client
from app.core.config.constants import RedisMessageKeyName
from app.schemas.entities.workflow.graph_state import ScheduleWorkflowState
from app.schemas.request.schedule_request import ScheduleTaskRequest
from app.schemas.response.workflow_response import WorkflowResponse
from app.service.chains.workflow.base_graph import BaseWorkflowGraph


class ScheduleStartGraph(BaseWorkflowGraph):
    """定时任务开始通知工作流（对齐 n8n scheduled start）。"""

    span_name = "定时任务开始通知"
    run_name = "schedule-start-graph"

    def _build_graph(self) -> CompiledStateGraph:
        """构建单节点图：发送开始通知 -> 封装结果。"""
        graph = StateGraph(ScheduleWorkflowState)
        graph.add_node("send_start", self._send_start_node)
        graph.add_node("package", self._package_result_node)
        graph.set_entry_point("send_start")
        graph.add_edge("send_start", "package")
        graph.add_edge("package", END)
        return graph.compile()

    def _send_start_node(self, state: ScheduleWorkflowState) -> Dict[str, Any]:
        """推送开始处理消息。"""
        req = state["request"]
        payload = {
            "env": "gray",
            "session_id": req.session_id,
            "reply_id": f"reply_{req.message_id}",
            "operate_id": "收到scheduled task任务",
            "reply_seq": 0,
            "reply_message_id": req.message_id,
            "status": "RUNNING",
            "content_type": 2,
            "content": {"text": "收到，开始处理定时任务", "agent": "scheduled tasks"},
            "create_ts": int(round(time.time() * 1000)),
        }
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            json.dumps(payload, ensure_ascii=False),
        )
        return {"output_text": "开始处理定时任务"}

    def _package_result_node(self, state: ScheduleWorkflowState) -> Dict[str, Any]:
        """封装 workflow_response。"""
        output_text = state.get("output_text", "")
        return {"workflow_response": WorkflowResponse(select_result=output_text, relate_data=None)}


class ScheduleTaskGraph(BaseWorkflowGraph):
    """定时任务消息下发工作流（对齐 n8n scheduled tasks）。"""

    span_name = "定时任务消息下发"
    run_name = "schedule-task-graph"

    def _build_graph(self) -> CompiledStateGraph:
        """构建单节点图：发送任务 -> 封装结果。"""
        graph = StateGraph(ScheduleWorkflowState)
        graph.add_node("send_task", self._send_task_node)
        graph.add_node("package", self._package_result_node)
        graph.set_entry_point("send_task")
        graph.add_edge("send_task", "package")
        graph.add_edge("package", END)
        return graph.compile()

    def _send_task_node(self, state: ScheduleWorkflowState) -> Dict[str, Any]:
        """推送单条定时任务消息。"""
        req: ScheduleTaskRequest = state["request"]
        title = req.task_title or "定时任务提醒"
        content = req.task_content or "用户请求创建一个定时任务。"
        cron = req.task_default_cron or ""
        time.sleep(2)
        payload = {
            "env": "gray",
            "session_id": req.session_id,
            "reply_id": f"reply_{req.message_id}",
            "operate_id": "定时任务",
            "reply_seq": 0,
            "reply_message_id": req.message_id,
            "status": "Running",
            "content_type": 5,
            "content": {
                "data": {
                    "actions": ["task"],
                    "title": title,
                    "content": content,
                    "default_cron": cron,
                }
            },
            "create_ts": int(round(time.time() * 1000)),
        }
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            json.dumps(payload, ensure_ascii=False),
        )
        return {"output_text": "定时任务已生成"}

    def _package_result_node(self, state: ScheduleWorkflowState) -> Dict[str, Any]:
        """封装 workflow_response。"""
        output_text = state.get("output_text", "")
        return {"workflow_response": WorkflowResponse(select_result=output_text, relate_data=None)}


__all__ = ["ScheduleStartGraph", "ScheduleTaskGraph"]
