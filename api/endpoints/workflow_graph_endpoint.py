# -*- coding: utf-8 -*-
# @Author   : kiro
# @Time     : 2025/12/14
# @File     : workflow_graph_endpoint.py

"""
工作流图可视化 API

提供工作流图结构导出接口，支持 JSON 和 Mermaid 格式。
"""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from app.service.chains.workflow.deepresearch.douyi.douyi_deepresearch_graph import DouyiDeepresearchGraph
from app.service.chains.workflow.deepresearch.zhiyi.zhiyi_deepresearch_graph import ZhiyiDeepresearchGraph
from app.service.chains.workflow.select.abroad_goods_graph import AbroadGoodsGraph
from app.service.chains.workflow.select.abroad_ins_graph import AbroadInsGraph
from app.service.chains.workflow.chat_graph import ChatGraph
from app.service.chains.workflow.select.douyi_graph import DouyiGraph
from app.service.chains.workflow.main_orchestrator_graph import MainOrchestratorGraph
from app.service.chains.workflow.select.zhikuan_ins_graph import ZhikuanInsGraph
from app.service.chains.workflow.select.zhiyi_graph import ZhiyiGraph
from app.service.chains.workflow.select.zxh_xhs_graph import ZxhXhsGraph

router = APIRouter()

_GRAPH_REGISTRY = {
    "main": MainOrchestratorGraph,
    "chat": ChatGraph,
    "abroad-ins": AbroadInsGraph,
    "abroad-goods": AbroadGoodsGraph,
    "zhikuan-ins": ZhikuanInsGraph,
    "zxh-xhs": ZxhXhsGraph,
    "zhiyi": ZhiyiGraph,
    "douyi": DouyiGraph,
    "douyi-deepresearch": DouyiDeepresearchGraph,
    "zhiyi-deepresearch": ZhiyiDeepresearchGraph,
}


@router.get("/list", summary="获取所有可用工作流列表")
def list_workflows() -> Dict[str, Any]:
    """返回所有可用的工作流类型"""
    return {"workflows": list(_GRAPH_REGISTRY.keys())}


@router.get("/{workflow_type}", summary="获取工作流图结构")
def get_workflow_graph(workflow_type: str) -> Dict[str, Any]:
    """
    获取指定工作流的图结构（JSON 格式）

    返回节点和边的列表，可用于前端渲染类似 n8n 的可视化界面。
    """
    if workflow_type not in _GRAPH_REGISTRY:
        raise HTTPException(status_code=404, detail=f"未知工作流类型: {workflow_type}")
    try:
        workflow = _GRAPH_REGISTRY[workflow_type]()
        return workflow.export_json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取图结构失败: {str(e)}")


@router.get("/{workflow_type}/mermaid", summary="获取工作流 Mermaid 图", response_class=PlainTextResponse)
def get_workflow_mermaid(workflow_type: str) -> str:
    """
    获取指定工作流的 Mermaid 图格式

    返回 Mermaid 语法字符串，可直接在支持 Mermaid 的 Markdown 渲染器中显示。
    """
    if workflow_type not in _GRAPH_REGISTRY:
        raise HTTPException(status_code=404, detail=f"未知工作流类型: {workflow_type}")
    try:
        workflow = _GRAPH_REGISTRY[workflow_type]()
        return workflow.export_mermaid()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取 Mermaid 图失败: {str(e)}")
