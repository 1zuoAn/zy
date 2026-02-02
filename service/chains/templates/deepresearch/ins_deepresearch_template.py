# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2026/1/23 09:37
# @File     : ins_deepresearch_template.py
from app.service.chains.templates.deepresearch.deepresearch_progress_template import (
    DeepresearchProgressTemplate,
    DeepresearchPhaseTemplate
)

INS_DEEPRESEARCH_TEMPLATE = DeepresearchProgressTemplate(
    task_name="数据洞察任务",
    workflow_type="douyi",
    phases=[
        # 非深度思考类型的阶段定义
        DeepresearchPhaseTemplate(
            name="正在启用趋势分析助手",
            normal_template="正在启用趋势分析助手"
        ),
        DeepresearchPhaseTemplate(
            name="正在生成数据洞察报告",
            normal_template="正在生成数据洞察报告"
        ),
        DeepresearchPhaseTemplate(
            name="报告已生成",
            normal_template="报告已生成"
        )
    ]
)

__all__ = ["INS_DEEPRESEARCH_TEMPLATE"]