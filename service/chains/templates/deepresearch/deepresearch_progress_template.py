# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2026/1/19
# @File     : deepresearch_progress_template.py
"""
数据洞察进度模板数据结构
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class DeepresearchPhaseTemplate(BaseModel):
    """数据洞察阶段模板"""

    name: str = Field(description="阶段名称")
    normal_template: str = Field(default="", description="非深度模式模板")
    thinking_template: str = Field(
        default="", description="深度思考模式模板，支持 {variable} 占位符"
    )


class DeepresearchProgressTemplate(BaseModel):
    """数据洞察进度模板"""

    task_name: str = Field(description="任务名称")
    workflow_type: str = Field(description="工作流类型，如 douyi/zhiyi/abroad")
    phases: List[DeepresearchPhaseTemplate] = Field(
        default_factory=list, description="阶段模板列表"
    )

    def get_phase(self, phase_name: str) -> Optional[DeepresearchPhaseTemplate]:
        """获取指定阶段的模板"""
        for phase in self.phases:
            if phase.name == phase_name:
                return phase
        return None

    def get_template(self, phase_name: str, is_thinking: bool = False) -> Optional[str]:
        """获取指定阶段的模板"""
        phase = self.get_phase(phase_name)
        if not phase:
            return None

        if is_thinking and phase.thinking_template:
            return phase.thinking_template
        return phase.normal_template


__all__ = [
    "DeepresearchPhaseTemplate",
    "DeepresearchProgressTemplate",
]
