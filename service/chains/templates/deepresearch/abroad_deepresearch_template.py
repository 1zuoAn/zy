# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2026/1/19
# @File     : abroad_deepresearch_template.py
"""
海外探款数据洞察进度模板
"""
from app.service.chains.templates.deepresearch.deepresearch_progress_template import (
    DeepresearchPhaseTemplate,
    DeepresearchProgressTemplate,
)

ABROAD_DEEPRESEARCH_TEMPLATE = DeepresearchProgressTemplate(
    task_name="数据洞察任务",
    workflow_type="abroad",
    phases=[
        # 深度思考模式输出
        DeepresearchPhaseTemplate(
            name="数据收集",
            thinking_template="""从<custom-thinking data-name="市场分析" data-href="https://bi-gray.zhiyitech.cn/cross-border/market-analysis/overview"></custom-thinking>模块查询到海外探款{industry_name}的行业销售数据；
            汇总已经获取到的行业数据，以及时间、平台等数据；
            接下来开始数据分析；
            """,
        ),
        DeepresearchPhaseTemplate(
            name="洞察生成中",
            thinking_template="""正在生成趋势洞察...
            趋势洞察生成成功；""",
        ),
        DeepresearchPhaseTemplate(
            name="洞察生成完成"
        ),
        # 非深度思考模式输出
        DeepresearchPhaseTemplate(
            name="正在启用趋势分析助手"
        ),
        DeepresearchPhaseTemplate(
            name="正在生成数据洞察报告"
        ),
        DeepresearchPhaseTemplate(
            name="报告已生成",
        )
    ],
)

__all__ = ["ABROAD_DEEPRESEARCH_TEMPLATE"]
