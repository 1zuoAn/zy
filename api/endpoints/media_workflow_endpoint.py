# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2025/11/26 22:41
# @File     : media_workflow_endpoint.py
from fastapi import APIRouter

from app.schemas.request.workflow_request import MainWorkflowRequest, WorkflowRequest
from app.schemas.response.common import CommonResponse
from app.schemas.response.workflow_response import WorkflowResponse
from app.service.chains.workflow.deepresearch.abroad.abroad_deepresearch_graph import AbroadDeepresearchGraph
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
from app.service.chains.workflow.image.create_image_graph import CreateImageGraph
from app.service.chains.workflow.image.edit_image_graph import EditImageGraph

router = APIRouter()


# ==================== 主工作流 ====================

@router.post("/main", response_model=CommonResponse[WorkflowResponse])
def run_main_workflow(request: MainWorkflowRequest):
    """主编排工作流入口 - 自动识别意图并路由到子工作流"""
    state = MainOrchestratorGraph().run(request)
    return CommonResponse[WorkflowResponse](result=state.get("workflow_response"))


# ==================== 媒体工作流 ====================

@router.post("/abroad-ins", response_model=CommonResponse[WorkflowResponse])
def run_abroad_ins_workflow(request: WorkflowRequest):
    """海外探款-INS数据源工作流入口"""
    state = AbroadInsGraph().run(request)
    return CommonResponse[WorkflowResponse](result=state.get("workflow_response"))


@router.post("/zhikuan-ins", response_model=CommonResponse[WorkflowResponse])
def run_zhikuan_ins_workflow(request: WorkflowRequest):
    """知款-INS数据源工作流入口"""
    state = ZhikuanInsGraph().run(request)
    return CommonResponse[WorkflowResponse](result=state.get("workflow_response"))


@router.post("/zxh-xhs", response_model=CommonResponse[WorkflowResponse])
def run_zxh_xhs_workflow(request: WorkflowRequest):
    """知小红-小红书数据源工作流入口"""
    state = ZxhXhsGraph().run(request)
    return CommonResponse[WorkflowResponse](result=state.get("workflow_response"))


# ==================== 选品工作流 ====================

@router.post("/douyi", response_model=CommonResponse[WorkflowResponse])
def run_douyi_workflow(request: WorkflowRequest):
    """抖衣(抖音)选品工作流入口"""
    state = DouyiGraph().run(request)
    return CommonResponse[WorkflowResponse](result=state.get("workflow_response"))


@router.post("/zhiyi", response_model=CommonResponse[WorkflowResponse])
def run_zhiyi_workflow(request: WorkflowRequest):
    """知衣选品工作流入口"""
    state = ZhiyiGraph().run(request)
    return CommonResponse[WorkflowResponse](result=state.get("workflow_response"))


@router.post("/abroad-goods", response_model=CommonResponse[WorkflowResponse])
def run_abroad_goods_workflow(request: WorkflowRequest):
    """海外探款商品选品工作流入口"""
    state = AbroadGoodsGraph().run(request)
    return CommonResponse[WorkflowResponse](result=state.get("workflow_response"))


# ==================== 其他工作流 ====================

@router.post("/chat", response_model=CommonResponse[WorkflowResponse])
def run_chat_workflow(request: WorkflowRequest):
    """闲聊工作流入口"""
    state = ChatGraph().run(request)
    return CommonResponse[WorkflowResponse](result=state.get("workflow_response"))


# ==================== 图片设计工作流 ====================

@router.post("/create_image", response_model=CommonResponse[WorkflowResponse])
def run_create_image_workflow(request: WorkflowRequest):
    """文生图工作流入口"""
    state = CreateImageGraph().run(request)
    return CommonResponse[WorkflowResponse](result=state.get("workflow_response"))


@router.post("/edit_image", response_model=CommonResponse[WorkflowResponse])
def run_edit_image_workflow(request: WorkflowRequest):
    """图生图工作流入口"""
    state = EditImageGraph().run(request)
    return CommonResponse[WorkflowResponse](result=state.get("workflow_response"))


# ==================== 数据洞察工作流 ====================
@router.post("/zhiyi-deepresearch", response_model=CommonResponse[WorkflowResponse])
def run_zhiyi_deepresearch_workflow(request: WorkflowRequest):
    state = ZhiyiDeepresearchGraph().run(request)
    return CommonResponse[WorkflowResponse](result=state.get("workflow_response"))

@router.post("/douyi-deepresearch", response_model=CommonResponse[WorkflowResponse])
def run_zhiyi_deepresearch_workflow(request: WorkflowRequest):
    state = DouyiDeepresearchGraph().run(request)
    return CommonResponse[WorkflowResponse](result=state.get("workflow_response"))

@router.post("/abroad-deepresearch", response_model=CommonResponse[WorkflowResponse])
def run_abroad_deepresearch_workflow(request: WorkflowRequest):
    state = AbroadDeepresearchGraph().run(request)
    return CommonResponse[WorkflowResponse](result=state.get("workflow_response"))