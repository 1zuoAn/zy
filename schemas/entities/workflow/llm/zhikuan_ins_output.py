# -*- coding: utf-8 -*-
from typing import List, Optional

from pydantic import BaseModel, Field


class ZhikuanInsCategoryParseResult(BaseModel):
    """知款 INS 类目解析结果"""
    class Config:
        populate_by_name = True

    gender: Optional[str] = Field(default=None, alias="gender", description="行业，对应品类维表的industry字段")
    category_list: List[str] = Field(
        default_factory=list,
        alias="categoryList",
        description="类目列表，可多选，包含root_category和leaf_category",
    )


class ZhikuanInsTimeParseResult(BaseModel):
    """知款 INS 时间解析结果"""
    class Config:
        populate_by_name = True

    blog_time_start: Optional[str] = Field(
        default=None, alias="blogTimeStart", description="帖子发布时间范围开始，格式 yyyy-MM-dd"
    )
    blog_time_end: Optional[str] = Field(
        default=None, alias="blogTimeEnd", description="帖子发布时间范围结束，格式 yyyy-MM-dd"
    )


class ZhikuanInsBloggerParseResult(BaseModel):
    """知款 INS 博主与监控解析结果"""
    class Config:
        populate_by_name = True

    is_monitor_blogger: Optional[int] = Field(
        default=None, ge=0, le=1, alias="isMonitorBlogger", description="是否只看监控达人，1为是，0为否"
    )
    blogger_group_id_list: List[str] = Field(
        default_factory=list, alias="bloggerGroupIdList", description="达人分组ID列表"
    )
    blogger_tags: List[str] = Field(
        default_factory=list, alias="bloggerTags", description="达人标签列表，如地区标签"
    )
    blogger_skin_colors: List[str] = Field(
        default_factory=list,
        alias="bloggerSkinColors",
        description="达人肤色列表，可选值：黑人、白人、黄种人、棕种人、小麦色",
    )
    blogger_body_shapes: List[str] = Field(
        default_factory=list,
        alias="bloggerBodyShapes",
        description="达人体型列表，可选值：正常体型、大码",
    )
    min_fans_num: Optional[int] = Field(default=None, alias="minFansNum", description="达人粉丝数下界")
    max_fans_num: Optional[int] = Field(default=None, alias="maxFansNum", description="达人粉丝数上界")


class ZhikuanInsStyleParseResult(BaseModel):
    """知款 INS 风格与细节解析结果"""
    class Config:
        populate_by_name = True

    image_style_list: List[str] = Field(
        default_factory=list, alias="imageStyleList", description="风格标签列表"
    )
    label_matrix: List[List[str]] = Field(
        default_factory=list,
        alias="labelMatrix",
        description="设计细节筛选项二维数组，外层可多选，子数组为同类型筛选项",
    )


class ZhikuanInsMiscParseResult(BaseModel):
    """知款 INS 杂项解析结果（标题/排序/限制等）"""
    class Config:
        populate_by_name = True

    sort_type: Optional[str] = Field(default=None, alias="sortType", description="排序方式")
    limit: Optional[int] = Field(default=None, alias="limit", description="返回数据条数，最大6000")
    user_data: Optional[int] = Field(
        default=None, ge=0, le=1, alias="user_data", description="是否按个人偏好选款，1为是，0为否"
    )
    title: Optional[str] = Field(default=None, alias="title", description="选品任务标题，10-15字")


class ZhikuanInsParseParam(BaseModel):
    """知款 INS数据源大模型参数解析输出参数"""
    class Config:
        populate_by_name = True

    source_type: Optional[str] = Field(default=None, alias="sourceType", description="数据来源类型，监控达人时固定为1")
    platform_id: str = Field(default="11", alias="platformId", description="平台ID，固定为11")
    blog_time_start: Optional[str] = Field(default=None, alias="blogTimeStart", description="帖子发布时间范围开始，格式 yyyy-MM-dd")
    blog_time_end: Optional[str] = Field(default=None, alias="blogTimeEnd", description="帖子发布时间范围结束，格式 yyyy-MM-dd")
    gender: str = Field(..., alias="gender", description="行业，对应品类维表的industry字段，如女装、女童等")
    category_list: List[str] = Field(default_factory=list, alias="categoryList", description="类目列表，可多选，包含root_category和leaf_category")
    blogger_tags: List[str] = Field(default_factory=list, alias="bloggerTags", description="达人标签列表，如地区标签")
    blogger_skin_colors: List[str] = Field(default_factory=list, alias="bloggerSkinColors", description="达人肤色列表，可选值：黑人、白人、黄种人、棕种人、小麦色")
    blogger_body_shapes: List[str] = Field(default_factory=list, alias="bloggerBodyShapes", description="达人体型列表，可选值：正常体型、大码")
    min_fans_num: Optional[int] = Field(default=None, alias="minFansNum", description="达人粉丝数下界")
    max_fans_num: Optional[int] = Field(default=None, alias="maxFansNum", description="达人粉丝数上界")
    image_style_list: List[str] = Field(default_factory=list, alias="imageStyleList", description="风格标签列表")
    label_matrix: List[List[str]] = Field(default_factory=list, alias="labelMatrix", description="设计细节筛选项二维数组，外层可多选，子数组为同类型筛选项（如颜色、面料、袖型等）")
    sort_type: str = Field(default="默认", alias="sortType", description="排序方式")
    limit: int = Field(default=6000, le=6000, alias="limit", description="返回数据条数，最大6000")
    user_data: int = Field(default=0, ge=0, le=1, alias="user_data", description="是否按个人偏好选款，1为是，0为否")
    is_monitor_blogger: int = Field(default=0, ge=0, le=1, alias="isMonitorBlogger", description="是否只看监控达人，1为是，0为否")
    blogger_group_id_list: List[str] = Field(default_factory=list, alias="bloggerGroupIdList", description="达人分组ID列表")
    title: str = Field(..., min_length=1, max_length=100, alias="title", description="选品任务标题，10-15字")


class ZhikuanInsSortTypeParseResult(BaseModel):
    """
    知款 INS排序项解析结果

    排序规则：
    - isMonitorBlogger=0时：默认/精选=47, 最新发布=1, 点赞最多=2, 评论最多=3
    - isMonitorBlogger=1时：默认/最新发布=1, 点赞最多=2, 评论最多=3
    """
    class Config:
        populate_by_name = True

    sort_type_final: str = Field(
        default="47",
        alias="sortTypeFinal",
        description="最终输出的排序编号，isMonitorBlogger=0时为数字编号，isMonitorBlogger=1时为数字编号",
    )
    sort_type_final_name: str = Field(
        default="精选",
        alias="sortTypeFinalName",
        description="排序项的中文名称，如精选、最新发布、点赞最多、评论最多",
    )
