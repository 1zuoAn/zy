# -*- coding: utf-8 -*-
from __future__ import annotations

import threading
from typing import Any, Type

from loguru import logger

from app.core.config.constants import WorkflowType
from app.schemas.entities.workflow.context import SubWorkflowResult
from app.schemas.response.workflow_response import WorkflowResponse
from app.service.chains.workflow.base_graph import BaseWorkflowGraph
from app.service.chains.workflow.image.create_image_graph import CreateImageGraph
from app.service.chains.workflow.image.edit_image_graph import EditImageGraph
from app.service.chains.workflow.image_search.image_search_graph import ImageSearchGraph
from app.service.chains.workflow.schedule.schedule_graph import ScheduleTaskGraph
from app.service.chains.workflow.select.abroad_goods_graph import AbroadGoodsGraph
from app.service.chains.workflow.select.abroad_ins_graph import AbroadInsGraph
from app.service.chains.workflow.select.douyi_graph import DouyiGraph
from app.service.chains.workflow.select.zhikuan_ins_graph import ZhikuanInsGraph
from app.service.chains.workflow.select.zhiyi_graph import ZhiyiGraph
from app.service.chains.workflow.select.zxh_xhs_graph import ZxhXhsGraph
from app.service.chains.workflow.shop.shop_graph import ShopGraph


class WorkflowDelegate:
    """子工作流委派执行器（单例）。"""

    def __init__(self) -> None:
        self._registry: dict[WorkflowType, Type[BaseWorkflowGraph]] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self._registry.update(
            {
                WorkflowType.SELECT_ZHIYI: ZhiyiGraph,
                WorkflowType.SELECT_DOUYI: DouyiGraph,
                WorkflowType.SELECT_ABROAD_GOODS: AbroadGoodsGraph,
                WorkflowType.MEDIA_ABROAD_INS: AbroadInsGraph,
                WorkflowType.MEDIA_ZHIKUAN_INS: ZhikuanInsGraph,
                WorkflowType.MEDIA_ZXH_XHS: ZxhXhsGraph,
                WorkflowType.IMAGE_SEARCH: ImageSearchGraph,
                WorkflowType.IMAGE_CREATE: CreateImageGraph,
                WorkflowType.IMAGE_EDIT: EditImageGraph,
                WorkflowType.SHOP: ShopGraph,
                WorkflowType.SCHEDULE: ScheduleTaskGraph,
            }
        )

    def register(self, workflow_type: WorkflowType, graph_cls: Type[BaseWorkflowGraph]) -> None:
        """注册子工作流图类。"""
        self._registry[workflow_type] = graph_cls

    def execute(self, workflow_type: WorkflowType, req: Any) -> SubWorkflowResult:
        """执行子工作流并归一化返回。"""
        graph_cls = self._registry.get(workflow_type)
        workflow_name = workflow_type.value if workflow_type else "unknown"
        if not graph_cls:
            return SubWorkflowResult(
                workflow_name=workflow_name,
                output="未找到对应的子工作流",
                relate_data=None,
                success=False,
                error_message="workflow_not_registered",
            )

        try:
            graph = graph_cls()
            state = graph.run(req)
            output, relate_data = self._extract_workflow_response(state)
            return SubWorkflowResult(
                workflow_name=workflow_name,
                output=output,
                relate_data=relate_data,
                success=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"[WorkflowDelegate] 子工作流执行失败: {workflow_name}, error={exc}")
            return SubWorkflowResult(
                workflow_name=workflow_name,
                output="子工作流执行失败",
                relate_data=None,
                success=False,
                error_message=str(exc),
            )

    @staticmethod
    def _extract_workflow_response(state: dict[str, Any]) -> tuple[str, str | None]:
        workflow_response = state.get("workflow_response")
        if isinstance(workflow_response, WorkflowResponse):
            return workflow_response.select_result or "", workflow_response.relate_data
        if isinstance(workflow_response, dict):
            output = (
                workflow_response.get("select_result")
                or workflow_response.get("selectResult")
                or ""
            )
            relate_data = workflow_response.get("relate_data") or workflow_response.get(
                "relateData"
            )
            return output, relate_data

        output = state.get("output_text") or state.get("chat_response") or ""
        return output, None


_delegate_lock = threading.Lock()
_delegate_instance: WorkflowDelegate | None = None


def get_delegate() -> WorkflowDelegate:
    """线程安全获取 WorkflowDelegate 单例。"""
    global _delegate_instance
    if _delegate_instance is None:
        with _delegate_lock:
            if _delegate_instance is None:
                _delegate_instance = WorkflowDelegate()
    return _delegate_instance


__all__ = ["WorkflowDelegate", "get_delegate"]
