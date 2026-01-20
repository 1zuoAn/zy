# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


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


class ZhiyiLlmParametersItem(BaseModel):
    """知衣 llm_parameters 埋点项"""
    model_config = ConfigDict(populate_by_name=True)

    property_list: Optional[list[dict[str, Any]]] = Field(
        default=None, alias="propertyList", description="属性列表"
    )
    output: Optional[dict[str, Any]] = Field(
        default=None, alias="output", description="输出参数"
    )


class ZhiyiCategoryParseResult(BaseModel):
    """知衣类目解析结果"""
    model_config = ConfigDict(populate_by_name=True)

    root_category_id: Optional[int] = Field(
        default=None, alias="root_category_id", description="一级品类ID"
    )
    category_id: List[int] = Field(
        default_factory=list, alias="category_id", description="品类ID列表"
    )
    root_category_name: Optional[str] = Field(
        default=None, alias="root_category_name", description="一级品类名称"
    )
    category_name: List[str] = Field(
        default_factory=list, alias="category_name", description="品类名称列表"
    )
    title: Optional[str] = Field(default=None, alias="title", description="选品任务标题")
    query_title: Optional[str] = Field(default=None, alias="queryTitle", description="搜索词")


class ZhiyiTimeParseResult(BaseModel):
    """知衣时间解析结果"""
    model_config = ConfigDict(populate_by_name=True)

    start_date: Optional[str] = Field(
        default=None, alias="startDate", description="统计开始时间，格式 yyyy-MM-dd"
    )
    end_date: Optional[str] = Field(
        default=None, alias="endDate", description="统计结束时间，格式 yyyy-MM-dd"
    )
    sale_start_date: Optional[str] = Field(
        default=None, alias="saleStartDate", description="上架开始时间，格式 yyyy-MM-dd"
    )
    sale_end_date: Optional[str] = Field(
        default=None, alias="saleEndDate", description="上架结束时间，格式 yyyy-MM-dd"
    )


class ZhiyiNumericParseResult(BaseModel):
    """知衣数值解析结果"""
    model_config = ConfigDict(populate_by_name=True)

    low_volume: Optional[int] = Field(default=None, alias="low_volume", description="销量下界")
    high_volume: Optional[int] = Field(default=None, alias="high_volume", description="销量上界")
    low_price: Optional[int] = Field(default=None, alias="low_price", description="价格下界（元）")
    high_price: Optional[int] = Field(default=None, alias="high_price", description="价格上界（元）")
    limit: Optional[int] = Field(default=None, alias="limit", description="返回数据条数，最大6000")


class ZhiyiRouteParseResult(BaseModel):
    """知衣路由解析结果"""
    model_config = ConfigDict(populate_by_name=True)

    type: Optional[str] = Field(default=None, alias="type", description="热销/新品")
    flag: Optional[int] = Field(default=None, alias="flag", description="1=监控店铺, 2=全网数据")
    user_data: Optional[int] = Field(default=None, alias="user_data", description="是否按个人偏好选款")
    shop_type: Optional[str] = Field(default=None, alias="shopType", description="店铺类型")
    shop_switch: List[str] = Field(default_factory=list, alias="shop_switch", description="店铺类型筛选")
    sort_field: Optional[str] = Field(default=None, alias="sortField", description="排序字段")


class ZhiyiBrandPhraseParseResult(BaseModel):
    """知衣品牌短语解析结果"""
    model_config = ConfigDict(populate_by_name=True)

    brand: Optional[str] = Field(default=None, alias="brand", description="品牌")


class ZhiyiPropertiesParseResult(BaseModel):
    """知衣属性解析结果"""
    model_config = ConfigDict(populate_by_name=True)

    properties: Optional[str] = Field(default=None, alias="properties", description="属性词，逗号分隔")


class ZhiyiSortResult(BaseModel):
    """知衣排序项解析结果"""
    model_config = ConfigDict(populate_by_name=True)

    sort_type_final: str = Field(default="", alias="sortTypeFinal", description="最终输出的排序项编码")
    sort_type_final_name: str = Field(default="", alias="sortTypeFinalName", description="排序项的中文名称")


class ZhiyiShopLlmCleanResult(BaseModel):
    clean_tag_list: list[str] = Field(default_factory=list, description="清洗后保留的店铺列表")


class ZhiyiPropertyLlmCleanResult(BaseModel):
    clean_tag_list: list[str] = Field(default_factory=list, description="清洗后保留的属性列表")


# ============================================================
# 知衣工作流类型定义
# ============================================================

ZhiyiApiBranch = Literal["brand", "monitor_hot", "monitor_new", "all_hot", "all_new"]
QueryTitleMode = Literal["none", "main", "fallback"]
RequestKind = Literal["shop", "standard", "extended"]


@dataclass
class BranchRequestSpec:
    """知衣 API 分支请求配置"""
    kind: RequestKind
    include_property_list: bool = True
    query_title_mode: QueryTitleMode = "main"
    group_id_list: list[int] = field(default_factory=list)
    page_no: int = 1
    page_size: int = 10
    include_start_date: bool = False
    use_param_shop_filter: bool = True


@dataclass
class RequestContext:
    """知衣 API 请求上下文"""
    start_date: str
    end_date: str
    sale_start_date: str
    sale_end_date: str
    category_id_list: list[int]
    root_category_id_list: list[int]
    property_list: list[dict]
    min_volume: int
    max_volume: int
    min_coupon_cprice: int
    max_coupon_cprice: int
    sort_field: str
    query_title_main: str
    query_title_fallback: str
    limit: int
    shop_label_list: list[str]
    shop_type: str | None


class ZhiyiUserTagResult(BaseModel):
    """知衣用户画像标签解析结果"""
    values: str = ""


# ============================================================
# 知衣 API 请求模型
# ============================================================


class ZhiyiStandardRequest(BaseModel):
    """知衣标准请求体（对齐 n8n params/params1/params3）"""
    model_config = ConfigDict(populate_by_name=True)

    start_date: str = Field(default="", alias="startDate")
    end_date: str = Field(default="", alias="endDate")
    sale_start_date: str = Field(default="", alias="saleStartDate")
    sale_end_date: str = Field(default="", alias="saleEndDate")
    category_id_list: list[int] = Field(default_factory=list, alias="categoryIdList")
    height: list = Field(default_factory=list)
    group_id_list: list[int] = Field(default_factory=list, alias="groupIdList")
    sale_type_status: None = Field(default=None, alias="saleTypeStatus")
    shop_label_list: list[str] = Field(default_factory=list, alias="shopLabelList")
    min_volume: int = Field(default=0, alias="minVolume")
    max_volume: int = Field(default=99999999, alias="maxVolume")
    shop_type: str | None = Field(default=None, alias="shopType")
    min_collect: None = Field(default=None, alias="minCollect")
    max_collect: None = Field(default=None, alias="maxCollect")
    month_min_volume: None = Field(default=None, alias="monthMinVolume")
    month_max_volume: None = Field(default=None, alias="monthMaxVolume")
    first_day_min_volume: None = Field(default=None, alias="firstDayMinVolume")
    first_day_max_volume: None = Field(default=None, alias="firstDayMaxVolume")
    root_category_id_list: list[int] = Field(default_factory=list, alias="rootCategoryIdList")
    wise_filter_flag: None = Field(default=None, alias="wiseFilterFlag")
    page_size: int = Field(default=10, alias="pageSize")
    page_no: int = Field(default=1, alias="pageNo")
    rank_type_list: None = Field(default=None, alias="rankTypeList")
    exclude_monitored_shop_flag: None = Field(default=None, alias="excludeMonitoredShopFlag")
    sort_type: str = Field(default="desc", alias="sortType")
    sort_field: str = Field(default="", alias="sortField")
    type: None = Field(default=None)
    min_coupon_cprice: int = Field(default=0, alias="minCouponCprice")
    max_coupon_cprice: int = Field(default=999999, alias="maxCouponCprice")
    min_history_min_left_cprice: None = Field(default=None, alias="minHistoryMinLeftCprice")
    max_history_min_left_cprice: None = Field(default=None, alias="maxHistoryMinLeftCprice")
    year_season_list: list = Field(default_factory=list, alias="yearSeasonList")
    age: list = Field(default_factory=list)
    query_title: str = Field(default="", alias="queryTitle")
    limit: int = Field(default=6000)
    property_list: list[dict] | None = Field(default=None, alias="propertyList")


class ZhiyiExtendedRequest(BaseModel):
    """知衣扩展请求体（对齐 n8n params2）"""
    model_config = ConfigDict(populate_by_name=True)

    root_category_id_list: list[int] = Field(default_factory=list, alias="rootCategoryIdList")
    category_id_list: list[int] = Field(default_factory=list, alias="categoryIdList")
    sale_start_date: str = Field(default="", alias="saleStartDate")
    sale_end_date: str = Field(default="", alias="saleEndDate")
    wise_filter_flag: None = Field(default=None, alias="wiseFilterFlag")
    page_no: int = Field(default=1, alias="pageNo")
    page_size: int = Field(default=10, alias="pageSize")
    limit: int = Field(default=6000)
    min_coupon_cprice: int = Field(default=0, alias="minCouponCprice")
    max_coupon_cprice: int = Field(default=999999, alias="maxCouponCprice")
    min_history_min_left_cprice: None = Field(default=None, alias="minHistoryMinLeftCprice")
    max_history_min_left_cprice: None = Field(default=None, alias="maxHistoryMinLeftCprice")
    shop_type: str | None = Field(default=None, alias="shopType")
    min_volume: int = Field(default=0, alias="minVolume")
    max_volume: int = Field(default=99999999, alias="maxVolume")
    min_collect: None = Field(default=None, alias="minCollect")
    max_collect: None = Field(default=None, alias="maxCollect")
    shop_label_list: list[str] = Field(default_factory=list, alias="shopLabelList")
    month_min_volume: None = Field(default=None, alias="monthMinVolume")
    month_max_volume: None = Field(default=None, alias="monthMaxVolume")
    min_sale_collect_rate: None = Field(default=None, alias="minSaleCollectRate")
    max_sale_collect_rate: None = Field(default=None, alias="maxSaleCollectRate")
    first_day_min_volume: None = Field(default=None, alias="firstDayMinVolume")
    first_day_max_volume: None = Field(default=None, alias="firstDayMaxVolume")
    first_week_min_volume: None = Field(default=None, alias="firstWeekMinVolume")
    first_week_max_volume: None = Field(default=None, alias="firstWeekMaxVolume")
    first_14day_min_volume: None = Field(default=None, alias="first14dayMinVolume")
    first_14day_max_volume: None = Field(default=None, alias="first14dayMaxVolume")
    first_month_min_volume: None = Field(default=None, alias="firstMonthMinVolume")
    first_month_max_volume: None = Field(default=None, alias="firstMonthMaxVolume")
    first_day_min_collect: None = Field(default=None, alias="firstDayMinCollect")
    first_day_max_collect: None = Field(default=None, alias="firstDayMaxCollect")
    first_week_min_collect: None = Field(default=None, alias="firstWeekMinCollect")
    first_week_max_collect: None = Field(default=None, alias="firstWeekMaxCollect")
    first_14day_min_collect: None = Field(default=None, alias="first14dayMinCollect")
    first_14day_max_collect: None = Field(default=None, alias="first14dayMaxCollect")
    first_month_min_collect: None = Field(default=None, alias="firstMonthMinCollect")
    first_month_max_collect: None = Field(default=None, alias="firstMonthMaxCollect")
    query_title: str = Field(default="", alias="queryTitle")
    brand_list: list = Field(default_factory=list, alias="brandList")
    year_season_list: list = Field(default_factory=list, alias="yearSeasonList")
    age_level_list: list = Field(default_factory=list, alias="ageLevelList")
    group_id_list: list[int] = Field(default_factory=list, alias="groupIdList")
    shop_style_list: list = Field(default_factory=list, alias="shopStyleList")
    sale_type_status: None = Field(default=None, alias="saleTypeStatus")
    white_bg_flag: None = Field(default=None, alias="whiteBgFlag")
    province: None = Field(default=None)
    city: None = Field(default=None)
    is_mall_item: None = Field(default=None, alias="isMallItem")
    video_url_flag: None = Field(default=None, alias="videoUrlFlag")
    rank_types: list = Field(default_factory=list, alias="rankTypes")
    exclude_monitored_shop_flag: None = Field(default=None, alias="excludeMonitoredShopFlag")
    exclude_collected_item_flag: None = Field(default=None, alias="excludeCollectedItemFlag")
    exclude_monitored_item_flag: None = Field(default=None, alias="excludeMonitoredItemFlag")
    exclude_viewed_item_flag: None = Field(default=None, alias="excludeViewedItemFlag")
    feature_tag: None = Field(default=None, alias="featureTag")
    is_jhs_item: None = Field(default=None, alias="isJhsItem")
    dy_flag: None = Field(default=None, alias="dyFlag")
    fabric: None = Field(default=None)
    cloth: None = Field(default=None)
    material: None = Field(default=None)
    exclude_pre_sale_types: list = Field(default_factory=list, alias="excludePreSaleTypes")
    sort_type: str = Field(default="desc", alias="sortType")
    sort_field: str = Field(default="", alias="sortField")
    start: int = Field(default=0)
    property_list: list[dict] | None = Field(default=None, alias="propertyList")
    start_date: str | None = Field(default=None, alias="startDate")
    end_date: str | None = Field(default=None, alias="endDate")


class ZhiyiShopRequest(BaseModel):
    """知衣品牌店铺请求体（对齐 n8n params4）"""
    model_config = ConfigDict(populate_by_name=True)

    page_size: int = Field(default=10, alias="pageSize")
    page_no: int = Field(default=1, alias="pageNo")
    shop_id: int = Field(alias="shopId")
    sort_field: str = Field(default="", alias="sortField")
    sort_type: str = Field(default="desc", alias="sortType")
    root_category_id_list: list[int] = Field(default_factory=list, alias="rootCategoryIdList")
    dy_flag: None = Field(default=None, alias="dyFlag")
    category_id_list: list[int] = Field(default_factory=list, alias="categoryIdList")
    wise_filter_flag: None = Field(default=None, alias="wiseFilterFlag")
    limit: int = Field(default=6000)
    genders: None = Field(default=None)
    period_volume_filter: None = Field(default=None, alias="periodVolumeFilter")
    start_date: str = Field(default="", alias="startDate")
    end_date: str = Field(default="", alias="endDate")
    sale_start_date: str = Field(default="", alias="saleStartDate")
    sale_end_date: str = Field(default="", alias="saleEndDate")
    indeterminate_root_category_id_list: list[int] = Field(
        default_factory=list, alias="indeterminateRootCategoryIdList"
    )
    limit_one_month: bool = Field(default=False, alias="limitOneMonth")
    min_volume: int = Field(default=0, alias="minVolume")
    max_volume: int = Field(default=99999999, alias="maxVolume")
    min_coupon_cprice: int = Field(default=0, alias="minCouponCprice")
    max_coupon_cprice: int = Field(default=999999, alias="maxCouponCprice")
    property_list: list[dict] | None = Field(default=None, alias="propertyList")
    query_title: str | None = Field(default=None, alias="queryTitle")
