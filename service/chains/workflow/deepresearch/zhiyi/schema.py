# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2026/1/14 17:23
# @File     : schema.py
from typing import Literal

from pydantic import BaseModel, Field


# 存储一些工作流产出的中间结果类
class ZhiyiThinkingApiParseParam(BaseModel):
    """
    API参数解析结果模型
    用于LLM结构化输出，解析用户问题中的查询参数
    """
    root_category_id: int | None = Field(default=None, description="根类目ID")
    root_category_id_name: str | None = Field(default=None, description="根类目名称")
    category_id: int | None = Field(
        default=None,
        description="类目ID，品类维表中没有对应时为null，不要把rootCategoryId当作categoryId填入，也不要给出相似的categoryId"
    )
    category_id_name: str | None = Field(default=None, description="类目名称，没有对应categoryId时为null")
    start_date: str | None = Field(default=None, description="开始日期，ISO 8601格式，例如 '2025-11-01'")
    end_date: str | None = Field(default=None, description="结束日期，ISO 8601格式，例如 '2025-11-30'")
    date_type: str | None = Field(default=None, description="日期类型，例如 'week' 或 'month'，不明确时返回null")
    table_type: Literal["1", "2", "3"] = Field(default="1",
        description="图表类型：'1'-品类分析（默认）、'2'-属性分析（长袖/长裙/季节/场景/尺码/面料/品牌/颜色等）、'3'-价格带分析（价格带相关/价格区间等）。注意：若categoryId为null，则不可为'2'"
    )
    is_shop: bool = Field(default=False, description="是否查询店铺数据，True表示查询店铺，False表示查询大盘")
    shop_name: str | None = Field(default=None, description="店铺名称，当is_shop为True时必填")



class ZhiyiCategoryFormatItem(BaseModel):
    root_category_id: str = Field(default=None)
    root_category_id_name: str = Field(default=None)
    category_id: str = Field(default=None)
    category_name: str = Field(default=None)

class ZhiyiParsedCategory(BaseModel):
    category_list: list[ZhiyiCategoryFormatItem] = Field(default_factory=list, description="处理后的类目列表")


