# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2026/1/19
# @File     : zhiyi_deepresearch_template.py
"""
知衣数据洞察进度模板（简化版）
"""
from app.service.chains.templates.deepresearch.deepresearch_progress_template import (
    DeepresearchPhaseTemplate,
    DeepresearchProgressTemplate,
)

ZHIYI_DEEPRESEARCH_TEMPLATE = DeepresearchProgressTemplate(
    task_name="数据洞察任务",
    workflow_type="zhiyi",
    phases=[
        DeepresearchPhaseTemplate(
            name="数据收集",
            normal_template="正在收集数据...",
            thinking_template="正在收集{shop_name}的店铺数据...",
        ),
        DeepresearchPhaseTemplate(
            name="洞察生成中",
            normal_template="正在生成数据洞察报告",
            thinking_template="正在生成{shop_name}的{insight_type}洞察...",
        ),
        DeepresearchPhaseTemplate(
            name="洞察生成完成",
            normal_template="洞察报告已生成完成",
            thinking_template="已完成{shop_name}的完整数据洞察分析",
        ),
        DeepresearchPhaseTemplate(
            name="启动助手",
            normal_template="正在启用趋势分析助手",
        ),
    ],
)

__all__ = ["ZHIYI_DEEPRESEARCH_TEMPLATE"]
