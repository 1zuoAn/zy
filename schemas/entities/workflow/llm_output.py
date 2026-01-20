# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2025/11/28 00:15
# @File     : llm_output.py
from typing import List, Literal, Optional, Any

from pydantic import BaseModel, Field, AliasChoices, ConfigDict, field_validator


class IntentClassifyResult(BaseModel):
    """意图分类结果"""
    intent: Literal["选品", "图搜", "生图改图", "趋势报告", "媒体", "店铺", "定时任务", "聊天机器人", "other"]
    confidence: float = Field(ge=0.0, le=1.0, description="置信度")
    reasoning: str = Field(description="推理过程")



class BoxLabelItem(BaseModel):
    """小红书筛选标签项"""
    class Config:
        populate_by_name = True

    key: str = Field(description="标签键，例如 '面料'、'领型'")
    value_list: List[str] = Field(default_factory=list, alias="valueList", description="标签值列表")


class ZxhXhsParseParam(BaseModel):
    """知小红 小红书数据源大模型输出参数"""
    class Config:
        populate_by_name = True

    industry: Optional[str] = Field(default=None, alias="industry", description="行业，固定为2")
    root_seo_list: Optional[List[str]] = Field(default=None, alias="rootSeoList", description="一级类目列表")
    second_seo_list: Optional[List[str]] = Field(default=None, alias="secondSeoList", description="二级类目列表")
    note_type: Optional[str] = Field(default=None, alias="noteType", description="笔记类型，0为图文，1为视频")
    box_label_color: Optional[str] = Field(default=None, alias="boxLabelColor", description="标签颜色")
    box_label_ret_list: Optional[List[BoxLabelItem]] = Field(default=None, alias="boxLabelRetList", description="筛选标签列表")
    style_list: Optional[List[str]] = Field(default=None, alias="styleList", description="风格标签列表")
    exclude_user_black_flag: Optional[str] = Field(default=None, alias="excludeUserBlackFlag", description="是否排除黑名单用户")
    exclude_user_black_word_flag: Optional[str] = Field(default=None, alias="excludeUserBlackWordFlag", description="是否排除黑名单关键词")
    min_fan_num: Optional[int] = Field(default=None, alias="minFanNum", description="粉丝数下界")
    max_fan_num: Optional[int] = Field(default=None, alias="maxFanNum", description="粉丝数上界")
    min_agg_like_num: Optional[int] = Field(default=None, alias="minAggLikeNum", description="点赞数下界")
    max_agg_like_num: Optional[int] = Field(default=None, alias="maxAggLikeNum", description="点赞数上界")
    min_agg_comment_num: Optional[int] = Field(default=None, alias="minAggCommentNum", description="评论数下界")
    max_agg_comment_num: Optional[int] = Field(default=None, alias="maxAggCommentNum", description="评论数上界")
    min_agg_collect_num: Optional[int] = Field(default=None, alias="minAggCollectNum", description="收藏数下界")
    max_agg_collect_num: Optional[int] = Field(default=None, alias="maxAggCollectNum", description="收藏数上界")
    max_publish_time: Optional[str] = Field(default=None, alias="maxPublishTime", description="发布时间上界，格式 yyyy-MM-dd")
    min_publish_time: Optional[str] = Field(default=None, alias="minPublishTime", description="发布时间下界，格式 yyyy-MM-dd")
    metric_date: Optional[str] = Field(default=None, alias="metricDate", description="数据统计周期")
    sort_field: Optional[str] = Field(default=None, alias="sortField", description="排序字段")
    sort_order: Optional[str] = Field(default=None, alias="sortOrder", description="排序顺序")
    is_monitor_blogger: Optional[int] = Field(default=None, alias="isMonitorBlogger", description="是否只看监控博主")
    user_data: Optional[int] = Field(default=None, alias="user_data", description="是否按个人偏好选款，1为是，0为否")
    title: Optional[str] = Field(default=None, alias="title", description="选品任务标题，10-15字")
    limit: Optional[int] = Field(default=None, alias="limit", description="返回数据条数，最大6000")
    topic_query_text: Optional[str] = Field(default=None, alias="topicQueryText", description="话题查询文本")
    search_like: Optional[str] = Field(default=None, alias="searchLike", description="搜索关键词")
    search_field_list: Optional[List[str]] = Field(default=None, alias="searchFieldList", description="搜索字段列表")

class ZxhXhsSortTypeParseResult(BaseModel):
    """知小红 小红书排序项解析结果"""
    class Config:
        populate_by_name = True

    sort_type_final: Optional[str] = Field(default=None, alias="sortTypeFinal", description="最终输出的排序项")
    sort_type_final_name: Optional[str] = Field(default=None, alias="sortTypeFinalName", description="排序项的描述名称")
    sort_order: Optional[str] = Field(default=None, alias="sortOrder", description="排序顺序，asc升序/desc降序")

class RagCleanedResult(BaseModel):
    """RAG 清洗结果"""
    content_list: list[str] = Field(default_factory=list, description="清洗后的标签列表")


# =====以下为兼容性保留，后续删除======

class AbroadInsParseParam(BaseModel):
    """
    海外探款ins数据源大模型输出参数
    """
    class Config:
        # 允许通过字段名（下划线风格）或别名（驼峰风格）进行填充
        populate_by_name = True

    start_date: Optional[str] = Field(default=None, alias="startDate", description="范围开始时间，格式 yyyy-MM-dd")
    end_date: Optional[str] = Field(default=None, alias="endDate", description="范围结束时间，格式 yyyy-MM-dd")
    category_list: List[List[str]] = Field(default_factory=list, alias="categoryList", description="类目路径二维数组，例如 [['女装', '上衣']]")
    label: List[List[str]] = Field(default_factory=list, alias="label", description="设计细节/普通筛选项路径二维数组，例如 [['面料', '提花']]")
    style_list: List[str] = Field(default_factory=list, alias="styleList", description="风格标签列表")
    region_list: List[List[str]] = Field(default_factory=list, alias="regionList", description="地区列表，例如 [['北美', '美国']]")
    blogger_skin_color_list: List[str] = Field(default_factory=list, alias="bloggerSkinColorList", description="达人肤色列表，可选值为'黑人', '白人', '黄种人', '棕种人', '小麦色'")
    blogger_shapes: List[str] = Field(default_factory=list, alias="bloggerShapes", description="达人体型列表，可选值为'正常体型', '大码'")
    min_fans_num: Optional[int] = Field(default=None, alias="minFansNum", description="达人粉丝数下界")
    max_fans_num: Optional[int] = Field(default=None, alias="maxFansNum", description="达人粉丝数上界")
    limit: int = Field(default=6000, le=6000, alias="limit", description="返回数据条数，最大6000")
    sort_type: str = Field(default="默认", alias="sortType", description="排序方式，例如 'SOURCE_TIME_DESC' 或 '默认'")
    search_content: str = Field(default=None, alias="searchContent", description="搜索内容")
    user_data: int = Field(default=0, ge=0, le=1, alias="user_data", description="是否按个人偏好选款，1为是，0为否")
    is_monitor_streamer: int = Field(default=0, ge=0, le=1, alias="isMonitorStreamer", description="是否只看监控达人，1为是，0为否")
    title: str = Field(..., min_length=0, max_length=100, alias="title", description="选品任务标题，10-15字")

class AbroadSortTypeParseResult(BaseModel):
    sort_type_final: str = Field(description="最终输出的排序项")
    sort_type_final_name: str = Field(description="排序项的描述名称")



class DouyiParseParam(BaseModel):
    """抖衣(抖音)数据源大模型输出参数"""
    class Config:
        populate_by_name = True

    root_category_id: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("rootCategoryId", "root_category_id"),
        serialization_alias="rootCategoryId",
        description="一级品类ID",
    )
    category_id_list: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("categoryIdList", "category_id_list", "category_id"),
        serialization_alias="categoryIdList",
        description="品类ID列表，逗号分隔",
    )
    @field_validator("category_id_list", mode="before")
    @classmethod
    def _normalize_category_id_list(cls, value: Any) -> Any:
        if isinstance(value, list):
            return ",".join(str(item).strip() for item in value if str(item).strip())
        return value

    # n8n 版本输出为 minSprice/maxSprice，这里兼容两套字段名
    min_price: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("minPrice", "minSprice"),
        serialization_alias="minPrice",
        description="最低价格（元）",
    )
    max_price: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("maxPrice", "maxSprice"),
        serialization_alias="maxPrice",
        description="最高价格（元）",
    )

    put_on_sale_start_date: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("putOnSaleStartDate", "put_on_sale_start_date"),
        serialization_alias="putOnSaleStartDate",
        description="上架开始日期，格式 yyyy-MM-dd",
    )
    put_on_sale_end_date: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("putOnSaleEndDate", "put_on_sale_end_date"),
        serialization_alias="putOnSaleEndDate",
        description="上架结束日期，格式 yyyy-MM-dd",
    )

    # n8n 版本还会输出 startDate/endDate（用于排序统计时间范围）
    start_date: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("startDate", "start_date"),
        serialization_alias="startDate",
        description="统计开始日期，格式 yyyy-MM-dd",
    )
    end_date: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("endDate", "end_date"),
        serialization_alias="endDate",
        description="统计结束日期，格式 yyyy-MM-dd",
    )

    year_season: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("yearSeason", "year_season"),
        serialization_alias="yearSeason",
        description="年份季节",
    )
    is_monitor_shop: Optional[int] = Field(
        default=0, alias="isMonitorShop", description="是否只看监控店铺，1为是，0为否"
    )
    is_monitor_streamer: Optional[int] = Field(
        default=0, alias="isMonitorStreamer", description="是否只看监控达人，1为是，0为否"
    )
    sort_field: Optional[str] = Field(default="默认", alias="sortField", description="排序字段")
    limit: Optional[int] = Field(default=6000, alias="limit", description="返回数据条数，最大6000")
    sale_style: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("saleStyle", "sale_style"),
        serialization_alias="saleStyle",
        description="销售方式/风格",
    )
    properties: Optional[str] = Field(default=None, alias="properties", description="属性词文本")
    has_live_sale: Optional[int] = Field(default=None, alias="hasLiveSale", description="是否本期有直播销售")
    has_video_sale: Optional[int] = Field(default=None, alias="hasVideoSale", description="是否本期有作品销售")
    has_card_sale: Optional[int] = Field(default=None, alias="hasCardSale", description="是否本期有商品卡销售")
    only_live_sale: Optional[int] = Field(default=None, alias="onlyLiveSale", description="仅看直播销售")
    only_video_sale: Optional[int] = Field(default=None, alias="onlyVideoSale", description="仅看作品销售")
    only_card_sale: Optional[int] = Field(default=None, alias="onlyCardSale", description="仅看商品卡销售")
    title: Optional[str] = Field(default=None, alias="title", description="选品任务标题")
    user_data: Optional[int] = Field(
        default=0,
        validation_alias=AliasChoices("user_data", "userData"),
        serialization_alias="userData",
        description="是否使用用户画像，1为是，0为否",
    )
    type: Optional[str] = Field(default=None, alias="type", description="类型标识（如热销/新品等）")
    brand: Optional[str] = Field(default=None, alias="brand", description="品牌关键词")


class DouyiSortTypeParseResult(BaseModel):
    """抖衣(抖音)排序项解析结果"""
    class Config:
        populate_by_name = True

    sort_type_final: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("sortTypeFinal", "sortField_new", "sortFieldNew"),
        serialization_alias="sortTypeFinal",
        description="最终输出的排序项",
    )
    sort_type_final_name: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("sortTypeFinalName", "sortField_new_name", "sortFieldNewName"),
        serialization_alias="sortTypeFinalName",
        description="排序项的描述名称",
    )


class DouyiUserTagResult(BaseModel):
    """抖衣用户画像标签解析结果"""
    values: Optional[str] = Field(default="", description="逗号分隔的标签字符串")




class ZhiyiParseParam(BaseModel):
    """知衣选品数据源大模型输出参数 - 完整版"""
    model_config = ConfigDict(populate_by_name=True)

    # 销量范围
    low_volume: int = Field(default=0, alias="low_volume", description="销量下界")
    high_volume: int = Field(default=99999999, alias="high_volume", description="销量上界")

    # 价格范围
    low_price: int = Field(default=0, alias="low_price", description="价格下界（元）")
    high_price: int = Field(default=999999, alias="high_price", description="价格上界（元）")

    # 统计时间范围
    start_date: Optional[str] = Field(default=None, alias="startDate", description="统计开始时间，格式 yyyy-MM-dd")
    end_date: Optional[str] = Field(default=None, alias="endDate", description="统计结束时间，格式 yyyy-MM-dd")

    # 上架时间范围
    sale_start_date: Optional[str] = Field(default=None, alias="saleStartDate", description="上架开始时间，格式 yyyy-MM-dd")
    sale_end_date: Optional[str] = Field(default=None, alias="saleEndDate", description="上架结束时间，格式 yyyy-MM-dd")

    # 品类
    category_id: List[int] = Field(default_factory=list, alias="category_id", description="品类ID列表")
    root_category_id: Optional[int] = Field(default=None, alias="root_category_id", description="一级品类ID")
    category_name: List[str] = Field(default_factory=list, alias="category_name", description="品类名称列表")
    root_category_name: Optional[str] = Field(default=None, alias="root_category_name", description="一级品类名称")

    # 属性与搜索
    properties: Optional[str] = Field(default=None, alias="properties", description="属性词，逗号分隔")
    query_title: Optional[str] = Field(default=None, alias="queryTitle", description="搜索词")
    brand: Optional[str] = Field(default=None, alias="brand", description="品牌")
    style_text: Optional[str] = Field(default=None, alias="styleText", description="风格描述文本，空格分隔")

    # 类型标识
    type: str = Field(default="热销", alias="type", description="热销/新品")
    shop_type: Optional[str] = Field(default="null", alias="shopType", description="店铺类型：0=C店，1=天猫，null=全部")
    flag: int = Field(default=2, alias="flag", description="1=监控店铺，2=全网数据")
    user_data: int = Field(default=0, alias="user_data", description="是否按个人偏好选款，1为是，0为否")
    shop_switch: List[str] = Field(default_factory=list, alias="shop_switch", description="店铺类型筛选")

    # 排序
    sort_field: Optional[str] = Field(default="默认", alias="sortField", description="排序字段")

    # 其他
    limit: int = Field(default=6000, le=6000, alias="limit", description="返回数据条数，最大6000")
    title: Optional[str] = Field(default=None, alias="title", description="选品任务标题")


class ZhiyiSortResult(BaseModel):
    """知衣排序项解析结果"""
    model_config = ConfigDict(populate_by_name=True)

    sort_type_final: str = Field(default="", alias="sortTypeFinal", description="最终输出的排序项编码")
    sort_type_final_name: str = Field(default="", alias="sortTypeFinalName", description="排序项的中文名称")



class AbroadGoodsParseParam(BaseModel):
    """海外探款商品数据源大模型输出参数"""
    class Config:
        populate_by_name = True

    # 销量范围
    min_sale_volume_total: Optional[int] = Field(default=None, alias="minSaleVolumeTotal", description="总销量下界")
    max_sale_volume_total: Optional[int] = Field(default=None, alias="maxSaleVolumeTotal", description="总销量上界")

    # 价格范围
    min_sprice: Optional[int] = Field(default=None, alias="minSprice", description="最低价格（元）")
    max_sprice: Optional[int] = Field(default=None, alias="maxSprice", description="最高价格（元）")

    # 时间范围
    put_on_sale_start_date: Optional[str] = Field(default=None, alias="putOnSaleStartDate", description="上架开始日期，格式 yyyy-MM-dd")
    put_on_sale_end_date: Optional[str] = Field(default=None, alias="putOnSaleEndDate", description="上架结束日期，格式 yyyy-MM-dd")
    start_date: Optional[str] = Field(default=None, alias="startDate", description="统计时间开始时间，格式yyyy-MM-dd")
    end_date: Optional[str] = Field(default=None, alias="endDate", description="统计时间结束时间，格式yyyy-MM-dd")

    # 品类
    category_id_list: list[list[str]] = Field(default_factory=list, alias="categoryIdList", description="品类ID列表")

    # 地区
    region_ids: Optional[str] = Field(default=None, alias="regionIds", description="地区ID，逗号分隔")
    country_list: Optional[str] = Field(default=None, alias="countryList", description="国家列表，逗号分隔")

    # 平台
    platform: Optional[str] = Field(default=None, alias="platform", description="平台，如 Shein, Temu")
    platform_type_list: Optional[str] = Field(default=None, alias="platformTypeList", description="平台类型列表")

    # 属性筛选
    feature: Optional[str] = Field(default=None, alias="feature", description="特征/款式")
    style: Optional[str] = Field(default=None, alias="style", description="风格描述文本")
    label: Optional[str] = Field(default=None, alias="label", description="标签")
    color: Optional[str] = Field(default=None, alias="color", description="颜色")
    brand: Optional[str] = Field(default=None, alias="brand", description="品牌")
    body_type: Optional[list[str] | str] = Field(
        default=None, alias="bodyType", description="适用体型"
    )

    # 类型
    type: Optional[str] = Field(default=None, alias="type", description="平台类型/查询类型")
    new_type: Optional[str] = Field(default=None, alias="newType", description="新品类型")
    text: Optional[str] = Field(default=None, alias="text", description="搜索文本")
    zone_type: Optional[str] = Field(default=None, alias="zoneType", description="专区类型，temu/amazon")
    on_sale_flag: Optional[int] = Field(default=None, alias="onSaleFlag", description="是否在售")

    # 模式标识
    flag: int = Field(default=2, alias="flag", description="1=监控店铺，2=全网数据")
    user_data: int = Field(default=0, alias="userData", description="是否按个人偏好选款，1为是，0为否")

    # 排序
    sort_field: Optional[str] = Field(default="默认", alias="sortField", description="排序字段")

    # 其他
    limit: int = Field(default=6000, le=6000, alias="limit", description="返回数据条数，最大6000")
    title: Optional[str] = Field(default=None, alias="title", description="选品任务标题")


class AbroadGoodsSortResult(BaseModel):
    """海外探款商品排序项解析结果"""
    class Config:
        populate_by_name = True

    # n8n 中 sortType 是数字: 1=最新上架, 8=近30天热销, 默认 1
    sort_type_final: int = Field(default=1, alias="sortTypeFinal", description="排序类型编码(1=最新上架,8=近30天热销)")
    sort_type_final_name: str = Field(default="最新上架", alias="sortTypeFinalName", description="排序项的中文名称")
