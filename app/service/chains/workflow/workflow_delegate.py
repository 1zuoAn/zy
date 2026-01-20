# -*- coding: utf-8 -*-
# @Author   : kiro
# @Time     : 2025/12/09
# @File     : workflow_delegate.py

"""
工作流委托器模块

提供 WorkflowDelegate 类，用于管理子工作流的注册、查找和执行。
通过委托模式解耦主编排工作流与具体子工作流实现。
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, TYPE_CHECKING

from loguru import logger

from app.core.config.constants import WorkflowType
from app.core.errors import AppException
from app.schemas.entities.workflow.context import SubWorkflowResult
from app.schemas.request.workflow_request import WorkflowRequest

if TYPE_CHECKING:
    pass


# 工作流工厂函数类型
WorkflowFactory = Callable[[], "BaseWorkflowGraph"]


class WorkflowDelegate:
    """
    工作流委托器
    
    负责管理子工作流的注册、查找和执行。
    通过委托模式解耦主编排工作流与具体子工作流实现。
    
    使用方式：
        delegate = get_delegate()
        result = delegate.execute(WorkflowType.MEDIA_ABROAD_INS, request)
    
    测试时可注入 mock 注册表：
        delegate = WorkflowDelegate(registry={...})
    """
    
    def __init__(self, registry: Optional[Dict[WorkflowType, WorkflowFactory]] = None):
        """
        初始化委托器
        
        Args:
            registry: 可选的工作流注册表，用于测试时注入 mock 工作流
        """
        self._registry: Dict[WorkflowType, WorkflowFactory] = registry if registry is not None else {}
    
    def register(self, workflow_type: WorkflowType, factory: WorkflowFactory) -> None:
        """
        注册工作流工厂函数
        
        Args:
            workflow_type: 工作流类型
            factory: 工作流工厂函数，调用时返回 BaseWorkflowGraph 实例
        """
        self._registry[workflow_type] = factory
        logger.debug(f"[WorkflowDelegate] 注册工作流: {workflow_type.value}")
    
    def execute(self, workflow_type: WorkflowType, request: WorkflowRequest) -> SubWorkflowResult:
        """
        执行指定类型的子工作流
        
        Args:
            workflow_type: 工作流类型
            request: 工作流请求
            
        Returns:
            SubWorkflowResult: 子工作流执行结果
            
        Note:
            当工作流类型未注册时，返回失败的 SubWorkflowResult 而非抛出异常，
            以便调用方统一处理。
        """
        # 查找工作流工厂
        factory = self._registry.get(workflow_type)
        if factory is None:
            error_msg = f"工作流类型未注册: {workflow_type.value}"
            logger.warning(f"[WorkflowDelegate] {error_msg}")
            return SubWorkflowResult(
                workflow_name=workflow_type.value,
                output=f"「{workflow_type.value}」功能正在开发中，敬请期待",
                success=False,
                error_message="workflow_not_registered"
            )
        
        logger.info(
            f"[WorkflowDelegate] 执行工作流: {workflow_type.value}, "
            f"user_id={request.user_id}, session_id={request.session_id}"
        )
        
        try:
            # 创建工作流实例并执行
            workflow = factory()
            state = workflow.run(request)  # 现在返回完整 state

            # 从 state 中提取 workflow_response
            response = state.get("workflow_response")
            if isinstance(response, dict):
                output = response.get("select_result") or "工作流执行完成"
                relate_data = response.get("relate_data")
            else:
                output = response.select_result if response else "工作流执行完成"
                relate_data = response.relate_data if response else None

            # 转换为统一的 SubWorkflowResult
            return SubWorkflowResult(
                workflow_name=workflow_type.value,
                output=output,
                relate_data=relate_data,
                success=True
            )
            
        except AppException as e:
            logger.error(
                f"[WorkflowDelegate] 工作流执行失败: {workflow_type.value}, "
                f"error={e.message}, user_id={request.user_id}, session_id={request.session_id}"
            )
            return SubWorkflowResult(
                workflow_name=workflow_type.value,
                output=f"{workflow_type.value} 执行失败",
                success=False,
                error_message=e.message
            )
            
        except Exception as e:
            logger.exception(
                f"[WorkflowDelegate] 工作流执行异常: {workflow_type.value}, "
                f"error={str(e)}, user_id={request.user_id}, session_id={request.session_id}"
            )
            return SubWorkflowResult(
                workflow_name=workflow_type.value,
                output=f"{workflow_type.value} 执行异常",
                success=False,
                error_message=str(e)
            )
    
    def is_registered(self, workflow_type: WorkflowType) -> bool:
        """
        检查工作流类型是否已注册
        
        Args:
            workflow_type: 工作流类型
            
        Returns:
            bool: 是否已注册
        """
        return workflow_type in self._registry


# 单例实例
_delegate_instance: Optional[WorkflowDelegate] = None


def get_delegate() -> WorkflowDelegate:
    """
    获取全局 WorkflowDelegate 单例实例
    
    首次调用时会自动注册所有子工作流。
    
    Returns:
        WorkflowDelegate: 委托器单例实例
    """
    global _delegate_instance
    
    if _delegate_instance is None:
        _delegate_instance = WorkflowDelegate()
        _register_all_workflows(_delegate_instance)
    
    return _delegate_instance


def _register_all_workflows(delegate: WorkflowDelegate) -> None:
    """
    注册所有子工作流
    
    使用延迟导入避免循环依赖。
    
    Args:
        delegate: 委托器实例
    
    注意：
        目前只有媒体子工作流已实现，其他 6 种意图工作流待实现：
        - SELECT (选品)
        - IMAGE_DESIGN (图片设计)
        - TRENDS (趋势报告)
        - SHOP (店铺)
        - SCHEDULE (定时任务)
        - CHAT (闲聊)
    """
    # === 媒体子工作流 (3种) - 已完成 ===
    
    # 媒体工作流 - 海外探款 INS
    delegate.register(
        WorkflowType.MEDIA_ABROAD_INS,
        lambda: _create_abroad_ins_workflow()
    )
    
    # 媒体工作流 - 知款 INS
    delegate.register(
        WorkflowType.MEDIA_ZHIKUAN_INS,
        lambda: _create_zhikuan_ins_workflow()
    )
    
    # 媒体工作流 - 知小红 小红书
    delegate.register(
        WorkflowType.MEDIA_ZXH_XHS,
        lambda: _create_zxh_xhs_workflow()
    )
    
    # === 闲聊工作流 - 已完成 ===
    delegate.register(
        WorkflowType.CHAT,
        lambda: _create_chat_workflow()
    )

    # === 知衣工作流 - 已完成 ===
    delegate.register(
        WorkflowType.SELECT_ZHIYI,
        lambda: _create_zhiyi_workflow()
    )

    # === 海外探款商品工作流 - 已完成 ===
    delegate.register(
        WorkflowType.SELECT_ABROAD_GOODS,
        lambda: _create_abroad_goods_workflow()
    )

    # === 抖衣(抖音)选品工作流 - 已完成 ===
    delegate.register(
        WorkflowType.SELECT_DOUYI,
        lambda: _create_douyi_workflow()
    )

    # === 图片设计工作流 - 已完成 ===
    delegate.register(
        WorkflowType.IMAGE_CREATE,
        lambda: _create_create_image_workflow()
    )

    delegate.register(
        WorkflowType.IMAGE_EDIT,
        lambda: _create_edit_image_workflow()
    )

    # TODO: 待实现的子工作流
    # - SelectWorkflow (选品)
    # - TrendsWorkflow (趋势报告)
    # - ShopWorkflow (店铺)
    # - ScheduleWorkflow (定时任务)


# === 工厂函数 (延迟导入) ===

def _create_abroad_ins_workflow():
    """延迟导入并创建 AbroadInsGraph 实例"""
    from app.service.chains.workflow.select.abroad_ins_graph import AbroadInsGraph
    return AbroadInsGraph()


def _create_zhikuan_ins_workflow():
    """延迟导入并创建 ZhikuanInsGraph 实例"""
    from app.service.chains.workflow.select.zhikuan_ins_graph import ZhikuanInsGraph
    return ZhikuanInsGraph()


def _create_zxh_xhs_workflow():
    """延迟导入并创建 ZxhXhsGraph 实例"""
    from app.service.chains.workflow.select.zxh_xhs_graph import ZxhXhsGraph
    return ZxhXhsGraph()


def _create_chat_workflow():
    """延迟导入并创建 ChatGraph 实例"""
    from app.service.chains.workflow.chat_graph import ChatGraph
    return ChatGraph()


def _create_zhiyi_workflow():
    """延迟导入并创建 ZhiyiGraph 实例"""
    from app.service.chains.workflow.select.zhiyi_graph import ZhiyiGraph
    return ZhiyiGraph()


def _create_abroad_goods_workflow():
    """延迟导入并创建 AbroadGoodsGraph 实例"""
    from app.service.chains.workflow.select.abroad_goods_graph import AbroadGoodsGraph
    return AbroadGoodsGraph()


def _create_douyi_workflow():
    """延迟导入并创建 DouyiGraph 实例"""
    from app.service.chains.workflow.select.douyi_graph import DouyiGraph
    return DouyiGraph()


def _create_create_image_workflow():
    """延迟导入并创建 CreateImageGraph 实例"""
    from app.service.chains.workflow.image.create_image_graph import CreateImageGraph
    return CreateImageGraph()


def _create_edit_image_workflow():
    """延迟导入并创建 EditImageGraph 实例"""
    from app.service.chains.workflow.image.edit_image_graph import EditImageGraph
    return EditImageGraph()


def reset_delegate() -> None:
    """
    重置单例实例（仅用于测试）
    
    在测试中使用此函数重置全局状态。
    """
    global _delegate_instance
    _delegate_instance = None


__all__ = [
    "WorkflowDelegate",
    "WorkflowFactory",
    "get_delegate",
    "reset_delegate",
]
