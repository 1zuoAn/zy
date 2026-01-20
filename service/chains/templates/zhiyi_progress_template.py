# -*- coding: utf-8 -*-
"""
知衣选品工作流进度模板（对齐 INS 风格）
"""
from app.schemas.entities.workflow.progress_template import (
    PhaseTemplate,
    WorkflowProgressTemplate,
)

ZHIYI_PROGRESS_TEMPLATE = WorkflowProgressTemplate(
    task_name="选品任务",
    phases=[
        PhaseTemplate(
            name="选品任务规划",
            # 内容由 LLM 生成，通过 content 传入
        ),
        PhaseTemplate(
            name="商品筛选中",
            template="""正在「{datasource}」中根据选品需求筛选商品
我已在「{platform}」平台数据内浏览{browsed_count}个商品
正在根据需求筛选商品中...
{filter_result_text}""",
        ),
        PhaseTemplate(
            name="二次筛选中",
            template="""正在「{datasource}」中再次筛选商品
我已在「{platform}」平台数据内额外浏览{browsed_count}个商品
正在根据需求筛选商品中...
{retry_result_text}""",
        ),
        PhaseTemplate(
            name="生成列表中",
            template="""正在根据筛选出的商品生成商品列表
列表已完成""",
        ),
        PhaseTemplate(
            name="选品完成",
            template="""选品任务已完成""",
        ),
        PhaseTemplate(
            name="生成列表失败",
            template="""生成列表失败""",
        ),
        PhaseTemplate(
            name="选品未完成",
            template="""选品未完成""",
        ),
    ],
)

__all__ = ["ZHIYI_PROGRESS_TEMPLATE"]
