# -*- coding: utf-8 -*-
from typing import Optional

from pydantic import BaseModel, Field


class AbroadGoodsCategoryParseResult(BaseModel):
    """海外探款商品类目解析结果"""
    class Config:
        populate_by_name = True

    category_id_list: list[list[str]] = Field(default_factory=list, alias="categoryIdList")


class AbroadGoodsAttributeParseResult(BaseModel):
    """海外探款商品属性解析结果"""
    class Config:
        populate_by_name = True

    feature: Optional[str] = Field(default=None, alias="feature", description="站点特色")
    label: Optional[str] = Field(default=None, alias="label", description="设计细节标签路径")
    style: Optional[str] = Field(default=None, alias="style", description="风格")
    color: Optional[str] = Field(default=None, alias="color", description="颜色")
    body_type: Optional[list[str] | str] = Field(
        default=None, alias="bodyType", description="适用体型"
    )


class AbroadGoodsBrandParseResult(BaseModel):
    """海外探款商品品牌解析结果"""
    class Config:
        populate_by_name = True

    brand: Optional[str] = Field(default=None, alias="brand", description="品牌")


class AbroadGoodsTimeParseResult(BaseModel):
    """海外探款商品时间解析结果"""
    class Config:
        populate_by_name = True

    put_on_sale_start_date: Optional[str] = Field(
        default=None, alias="putOnSaleStartDate", description="上架开始日期"
    )
    put_on_sale_end_date: Optional[str] = Field(
        default=None, alias="putOnSaleEndDate", description="上架结束日期"
    )
    start_date: Optional[str] = Field(default=None, alias="startDate", description="统计开始日期")
    end_date: Optional[str] = Field(default=None, alias="endDate", description="统计结束日期")
    on_sale_flag: Optional[int] = Field(default=None, alias="onSaleFlag", description="是否在售")


class AbroadGoodsNumericParseResult(BaseModel):
    """海外探款商品数值解析结果"""
    class Config:
        populate_by_name = True

    min_sale_volume_total: Optional[int] = Field(
        default=None, alias="minSaleVolumeTotal", description="总销量下界"
    )
    max_sale_volume_total: Optional[int] = Field(
        default=None, alias="maxSaleVolumeTotal", description="总销量上界"
    )
    min_sprice: Optional[int] = Field(default=None, alias="minSprice", description="最低价格（元）")
    max_sprice: Optional[int] = Field(default=None, alias="maxSprice", description="最高价格（元）")
    limit: Optional[int] = Field(default=None, alias="limit", description="返回数据条数")


class AbroadGoodsPlatformRouteParseResult(BaseModel):
    """海外探款商品平台与路由解析结果"""
    class Config:
        populate_by_name = True

    platform: Optional[str] = Field(default=None, alias="platform", description="平台名称")
    platform_type_list: Optional[str] = Field(
        default=None, alias="platformTypeList", description="平台类型列表"
    )
    type: Optional[str] = Field(default=None, alias="type", description="平台类型/查询类型")
    zone_type: Optional[str] = Field(default=None, alias="zoneType", description="专区类型")
    flag: Optional[int] = Field(default=None, alias="flag", description="查询模式")
    user_data: Optional[int] = Field(default=None, alias="userData", description="个人偏好模式")
    new_type: Optional[str] = Field(default=None, alias="newType", description="新品类型")


class AbroadGoodsRegionParseResult(BaseModel):
    """海外探款商品地域解析结果"""
    class Config:
        populate_by_name = True

    region_ids: Optional[str] = Field(default=None, alias="regionIds", description="地区ID")
    country_list: Optional[str] = Field(default=None, alias="countryList", description="国家列表")


class AbroadGoodsTextTitleParseResult(BaseModel):
    """海外探款商品文本与标题解析结果"""
    class Config:
        populate_by_name = True

    text: Optional[str] = Field(default=None, alias="text", description="补充搜索词")
    title: Optional[str] = Field(default=None, alias="title", description="任务标题")
    sort_field: Optional[str] = Field(default=None, alias="sortField", description="排序字段")


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
    put_on_sale_start_date: Optional[str] = Field(
        default=None, alias="putOnSaleStartDate", description="上架开始日期，格式 yyyy-MM-dd"
    )
    put_on_sale_end_date: Optional[str] = Field(
        default=None, alias="putOnSaleEndDate", description="上架结束日期，格式 yyyy-MM-dd"
    )
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


class AbroadGoodsStyleResult(BaseModel):
    """海外探款商品风格解析结果"""
    style_list: list[str] = Field(default_factory=list, description="风格标签列表")


class AbroadGoodsBrandResult(BaseModel):
    clean_tag_list: list[str] = Field(default_factory=list, description="清洗后保留的品牌标签列表")
