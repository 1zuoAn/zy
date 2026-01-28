"""
定时任务工作流 - 占位实现
"""
from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.schemas.entities.workflow.graph_state import ScheduleWorkflowState
from app.schemas.response.workflow_response import WorkflowResponse
from app.service.chains.workflow.base_graph import BaseWorkflowGraph


class ScheduleGraph(BaseWorkflowGraph):
    """定时任务工作流（占位）"""

    span_name = "定时任务工作流"
    run_name = "schedule-graph"

    def _build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(ScheduleWorkflowState)
        graph.add_node("init_state", self._init_state_node)
        graph.add_node("package", self._package_result_node)

        graph.set_entry_point("init_state")
        graph.add_edge("init_state", "package")
        graph.add_edge("package", END)
        return graph.compile()

    def _init_state_node(self, state: ScheduleWorkflowState) -> Dict[str, Any]:
        return {}

    def _package_result_node(self, state: ScheduleWorkflowState) -> Dict[str, Any]:
        response = WorkflowResponse(select_result="定时任务功能暂未接入", relate_data=None)
        return {"workflow_response": response}


__all__ = ["ScheduleGraph"]
