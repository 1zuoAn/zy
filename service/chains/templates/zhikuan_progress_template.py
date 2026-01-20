# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2025/12/18 18:01
# @File     : zhikuan_progress_template.py
from app.schemas.entities.workflow.progress_template import (
    PhaseTemplate,
    WorkflowProgressTemplate,
)

ZHIKUAN_INS_PROGRESS_TEMPLATE = WorkflowProgressTemplate(
    task_name="选品任务",
    phases=[
        PhaseTemplate(
            name="选品任务规划",
        ),
        PhaseTemplate(
            name="帖子筛选中",
            template="""正在「{datasource}」中根据选品需求筛选帖子
我已在INS平台数据内浏览{browsed_count}个帖子
正在根据需求筛选商品中...
{filter_result_text}""",
        ),
        PhaseTemplate(
            name="二次筛选中",
            template="""正在「{datasource}」中再次筛选帖子
我已在INS平台数据内额外浏览{browsed_count}个帖子
正在根据需求筛选帖子中...
{retry_result_text}""",
        ),
        PhaseTemplate(
            name="生成列表中",
            template="""正在根据筛选出的商品生成帖子列表
列表已完成""",
        ),
        PhaseTemplate(
            name="选品完成",
            template="""选品任务已完成""",
        ),
    ],
)

__all__ = ["ZHIKUAN_INS_PROGRESS_TEMPLATE"]