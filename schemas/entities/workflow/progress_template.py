# -*- coding: utf-8 -*-
"""
工作流进度模板数据结构

定义工作流进度推送的模板结构，支持变量填充。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PhaseTemplate(BaseModel):
    """阶段模板 - 支持多种状态的模板"""

    name: str = Field(description="阶段名称")
    # 各状态对应的模板，支持 {variable} 占位符
    templates: Dict[str, str] = Field(
        default_factory=dict,
        description="状态模板映射，如 {'running': '正在筛选...', 'success': '筛选完成', 'fail': '筛选失败'}",
    )

    # 兼容旧版本的单一模板字段
    template: Optional[str] = Field(
        default=None, description="[已废弃] 单一模板，建议使用 templates 字典"
    )


class WorkflowProgressTemplate(BaseModel):
    """工作流进度模板"""

    task_name: str = Field(description="任务名称")
    phases: List[PhaseTemplate] = Field(default_factory=list, description="阶段模板列表")

    def get_template(self, phase_name: str, status: str = "running") -> Optional[str]:
        """获取指定阶段和状态的模板"""
        for phase in self.phases:
            if phase.name == phase_name:
                # 优先使用 templates 字典
                if phase.templates and status in phase.templates:
                    return phase.templates[status]
                # 兼容旧版本
                return phase.template
        return None


# 兼容旧版本的别名
ProgressPhase = PhaseTemplate

__all__ = [
    "PhaseTemplate",
    "ProgressPhase",
    "WorkflowProgressTemplate",
]
