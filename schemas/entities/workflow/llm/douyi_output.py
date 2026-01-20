# -*- coding: utf-8 -*-
from typing import Any, Optional

from pydantic import AliasChoices, BaseModel, Field, field_validator


class DouyiCategoryParseResult(BaseModel):
    """抖衣类目解析结果"""
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


class DouyiTimeParseResult(BaseModel):
    """抖衣时间解析结果"""
    class Config:
        populate_by_name = True

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


class DouyiNumericParseResult(BaseModel):
    """抖衣数值解析结果"""
    class Config:
        populate_by_name = True

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
    limit: Optional[int] = Field(
        default=None, alias="limit", description="返回数据条数，最大6000"
    )


class DouyiSalesFlagParseResult(BaseModel):
    """抖衣销售方式/监控解析结果"""
    class Config:
        populate_by_name = True

    sale_style: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("saleStyle", "sale_style"),
        serialization_alias="saleStyle",
        description="销售方式/风格",
    )
    is_monitor_shop: Optional[int] = Field(
        default=0, alias="isMonitorShop", description="是否只看监控店铺，1为是，0为否"
    )
    is_monitor_streamer: Optional[int] = Field(
        default=0, alias="isMonitorStreamer", description="是否只看监控达人，1为是，0为否"
    )
    has_live_sale: Optional[int] = Field(default=None, alias="hasLiveSale", description="是否本期有直播销售")
    has_video_sale: Optional[int] = Field(default=None, alias="hasVideoSale", description="是否本期有作品销售")
    has_card_sale: Optional[int] = Field(default=None, alias="hasCardSale", description="是否本期有商品卡销售")
    only_live_sale: Optional[int] = Field(default=None, alias="onlyLiveSale", description="仅看直播销售")
    only_video_sale: Optional[int] = Field(default=None, alias="onlyVideoSale", description="仅看作品销售")
    only_card_sale: Optional[int] = Field(default=None, alias="onlyCardSale", description="仅看商品卡销售")


class DouyiSortIntentParseResult(BaseModel):
    """抖衣排序意图解析结果"""
    class Config:
        populate_by_name = True

    sort_field: Optional[str] = Field(default="默认", alias="sortField", description="排序字段")
    type: Optional[str] = Field(default=None, alias="type", description="类型标识（如热销/新品等）")


class DouyiPropertiesParseResult(BaseModel):
    """抖衣属性词解析结果"""
    class Config:
        populate_by_name = True

    properties: Optional[str] = Field(default=None, alias="properties", description="属性词文本")


class DouyiMiscParseResult(BaseModel):
    """抖衣标题/画像解析结果"""
    class Config:
        populate_by_name = True

    brand: Optional[str] = Field(default=None, alias="brand", description="品牌关键词")
    user_data: Optional[int] = Field(
        default=0,
        validation_alias=AliasChoices("user_data", "userData"),
        serialization_alias="userData",
        description="是否使用用户画像，1为是，0为否",
    )
    title: Optional[str] = Field(default=None, alias="title", description="选品任务标题")


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


class DouyiPropertyLlmCleanResult(BaseModel):
    """抖衣属性标签召回清洗结果"""
    clean_tag_list: list[str] = Field(default_factory=list, description="清洗后保留的属性标签列表")


class DouyiPropertyListParseResult(BaseModel):
    """抖衣属性封装结果（n8n propertyList）"""
    class Config:
        populate_by_name = True

    property_list: list[list[str]] = Field(
        default_factory=list,
        alias="propertyList",
        description="属性列表，二维数组",
    )
