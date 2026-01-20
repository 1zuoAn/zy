# -*- coding: utf-8 -*-
"""
知衣/抖衣 API 数据模型定义

包含请求参数和响应实体类。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, Field, field_validator, ConfigDict

from app.schemas.entities.workflow.llm_output import DouyiParseParam, ZhiyiParseParam


class ZhiyiSearchRequest(BaseModel):
    """知衣商品搜索请求参数 - 对齐n8n格式"""

    class Config:
        populate_by_name = True

    # 品类 - 对齐n8n: rootCategoryIdList (数组)
    root_category_id_list: Optional[List[int]] = Field(default=None, alias="rootCategoryIdList")
    category_id_list: Optional[List[int]] = Field(default=None, alias="categoryIdList")

    # 销量范围 - 对齐n8n: minVolume/maxVolume
    min_volume: Optional[int] = Field(default=None, alias="minVolume")
    max_volume: Optional[int] = Field(default=None, alias="maxVolume")

    # 价格范围（单位：分）- 对齐n8n: minCouponCprice/maxCouponCprice
    min_coupon_cprice: Optional[int] = Field(default=None, alias="minCouponCprice")
    max_coupon_cprice: Optional[int] = Field(default=None, alias="maxCouponCprice")

    # 统计时间范围
    start_date: Optional[str] = Field(default=None, alias="startDate")
    end_date: Optional[str] = Field(default=None, alias="endDate")

    # 上架时间范围
    sale_start_date: Optional[str] = Field(default=None, alias="saleStartDate")
    sale_end_date: Optional[str] = Field(default=None, alias="saleEndDate")

    # 属性与搜索 - 对齐n8n: propertyList (对象数组)
    property_list: Optional[List[Dict[str, Any]]] = Field(default=None, alias="propertyList")
    query_title: Optional[str] = Field(default=None, alias="queryTitle")
    brand: Optional[str] = Field(default=None, alias="brand")
    style_list: Optional[List[str]] = Field(default=None, alias="styleList")

    # 监控分组 - 对齐 n8n: groupIdList（默认 []；监控店铺时为 [-3, -4]）
    group_id_list: List[int] = Field(default_factory=list, alias="groupIdList")

    # 店铺筛选 - 对齐n8n: shopLabelList
    shop_type: Optional[str] = Field(default=None, alias="shopType")
    shop_label_list: Optional[List[str]] = Field(default=None, alias="shopLabelList")
    shop_id: Optional[int] = Field(default=None, alias="shopId")

    # 模式标识
    flag: int = Field(default=2, alias="flag")
    user_data: int = Field(default=0, alias="userData")

    # 排序
    sort_field: Optional[str] = Field(default=None, alias="sortField")
    sort_type: str = Field(default="desc", alias="sortType")

    # 分页
    page_no: int = Field(default=1, alias="pageNo")
    page_size: int = Field(default=10, alias="pageSize")
    limit: int = Field(default=6000, alias="limit")

    @classmethod
    def from_parse_param(
        cls,
        param: ZhiyiParseParam,
        style_list: Optional[List[str]] = None,
        property_list: Optional[List[Dict[str, Any]]] = None,
    ) -> "ZhiyiSearchRequest":
        """从 LLM 解析结果转换为请求参数 - 对齐n8n格式"""
        # 构建 rootCategoryIdList (数组格式)
        root_category_id_list = [param.root_category_id] if param.root_category_id else None

        return cls(
            rootCategoryIdList=root_category_id_list,
            categoryIdList=param.category_id if param.category_id else None,
            minVolume=param.low_volume if param.low_volume > 0 else None,
            maxVolume=param.high_volume if param.high_volume < 99999999 else None,
            # 对齐 n8n：low_price/high_price 直接映射到 minCouponCprice/maxCouponCprice
            minCouponCprice=param.low_price if param.low_price > 0 else 0,
            maxCouponCprice=param.high_price if param.high_price < 999999 else 999999,
            startDate=param.start_date,
            endDate=param.end_date,
            saleStartDate=param.sale_start_date if param.sale_start_date else None,
            saleEndDate=param.sale_end_date if param.sale_end_date else None,
            propertyList=property_list,
            queryTitle=param.query_title if param.query_title else None,
            brand=param.brand if param.brand else None,
            styleList=style_list,
            groupIdList=[-3, -4] if getattr(param, "flag", 2) == 1 else [],
            shopType=param.shop_type if param.shop_type != "null" else None,
            shopLabelList=param.shop_switch if param.shop_switch else None,
            flag=param.flag,
            userData=param.user_data,
            sortField=param.sort_field if param.sort_field != "默认" else None,
            limit=param.limit or 6000,
        )

    def to_simplified(self) -> "ZhiyiSearchRequest":
        """生成精简版请求（用于兜底查询）- 对齐n8n格式"""
        return ZhiyiSearchRequest(
            rootCategoryIdList=self.root_category_id_list,
            categoryIdList=self.category_id_list,
            startDate=self.start_date,
            endDate=self.end_date,
            groupIdList=self.group_id_list,
            flag=self.flag,
            sortField=self.sort_field,
            limit=self.limit,
        )


class ZhiyiGoodsEntity(BaseModel):
    """知衣商品实体"""

    class Config:
        populate_by_name = True

    goods_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("goodsId", "itemId"),
        serialization_alias="goodsId",
        description="商品ID（兼容 goodsId/itemId）",
    )
    spu_id: Optional[str] = Field(default=None, alias="spuId")
    title: Optional[str] = Field(default=None, alias="title")
    pic_url: Optional[str] = Field(default=None, alias="picUrl")
    price: Optional[int] = Field(default=None, alias="price")
    sale_volume: Optional[int] = Field(default=None, alias="saleVolume")
    sale_amount: Optional[int] = Field(default=None, alias="saleAmount")
    first_record_date: Optional[str] = Field(default=None, alias="firstRecordDate")
    shop_name: Optional[str] = Field(default=None, alias="shopName")
    brand_name: Optional[str] = Field(default=None, alias="brandName")

    @field_validator("goods_id", mode="before")
    @classmethod
    def _coerce_goods_id(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        return str(value)


class DouyiSearchRequest(BaseModel):
    """抖衣商品搜索请求参数"""

    class Config:
        populate_by_name = True

    # 品类
    root_category_id: Optional[int] = Field(default=None, alias="rootCategoryId")
    category_id_list: Optional[List[int]] = Field(default=None, alias="categoryIdList")

    # 属性
    hot_properties: Optional[List[List[str]]] = Field(default=None, alias="hotProperties")

    # 价格（单位：分）
    min_price: Optional[int] = Field(default=None, alias="minPrice")
    max_price: Optional[int] = Field(default=None, alias="maxPrice")

    # 上架时间
    min_first_record_date: Optional[str] = Field(default=None, alias="minFirstRecordDate")
    max_first_record_date: Optional[str] = Field(default=None, alias="maxFirstRecordDate")
    first_record_date_type: Optional[str] = Field(default="custom", alias="firstRecordDateType")

    # 年份季节
    year_season: Optional[str] = Field(default=None, alias="yearSeason")

    # 监控标记
    is_monitor_shop: Optional[int] = Field(default=0, alias="isMonitorShop")
    is_monitor_streamer: Optional[int] = Field(default=0, alias="isMonitorStreamer")

    # 销售方式
    has_live_sale: Optional[int] = Field(default=None, alias="hasLiveSale", description="是否本期有直播销售")
    has_video_sale: Optional[int] = Field(default=None, alias="hasVideoSale", description="是否本期有作品销售")
    has_card_sale: Optional[int] = Field(default=None, alias="hasCardSale", description="是否本期有商品卡销售")
    only_live_sale: Optional[int] = Field(default=None, alias="onlyLiveSale", description="仅看直播销售")
    only_video_sale: Optional[int] = Field(default=None, alias="onlyVideoSale", description="仅看作品销售")
    only_card_sale: Optional[int] = Field(default=None, alias="onlyCardSale", description="仅看商品卡销售")

    # 排序
    sort_field: Optional[str] = Field(default=None, alias="sortField")
    sort_type: Optional[str] = Field(default="desc", alias="sortType")

    # 排序统计时间范围（n8n: sortStartDate/sortEndDate）
    sort_start_date: Optional[str] = Field(default=None, alias="sortStartDate")
    sort_end_date: Optional[str] = Field(default=None, alias="sortEndDate")
    sort_date_type: Optional[str] = Field(default="custom", alias="sortDateType")
    sort_recent_type: Optional[str] = Field(default=None, alias="sortRecentType")

    # 分页
    page_no: Optional[int] = Field(default=1, alias="pageNo")
    page_size: Optional[int] = Field(default=40, alias="pageSize")
    limit: Optional[int] = Field(default=6000, alias="limit")

    # 查询控制（n8n: isExport/queryType/keyword）
    is_export: Optional[int] = Field(default=1, alias="isExport")
    query_type: Optional[str] = Field(default="GoodsLibraryAll", alias="queryType")
    keyword: Optional[str] = Field(default="", alias="keyword")

    # 其他可选字段
    goods_type: Optional[str] = Field(default=None, alias="goodsType")
    gender: Optional[str] = Field(default=None, alias="gender")
    sale_style: Optional[str] = Field(default=None, alias="saleStyle")
    brand: Optional[str] = Field(default=None, alias="brand", description="品牌名称")
    shop_id: Optional[str] = Field(default=None, alias="shopId", description="店铺ID")

    @classmethod
    def from_parse_param(cls, param: DouyiParseParam) -> "DouyiSearchRequest":
        """从 LLM 解析结果转换为请求参数"""
        # 解析品类ID列表
        category_id_list = None
        if param.category_id_list:
            category_id_list = [int(x.strip()) for x in param.category_id_list.split(",") if x.strip()]
        root_category_id = param.root_category_id
        if not root_category_id and category_id_list:
            root_category_id = category_id_list[0]

        return cls(
            rootCategoryId=root_category_id,
            categoryIdList=category_id_list,
            minPrice=param.min_price * 100 if param.min_price is not None else None,  # 元转分
            maxPrice=param.max_price * 100 if param.max_price is not None else None,  # 元转分
            minFirstRecordDate=param.put_on_sale_start_date,
            maxFirstRecordDate=param.put_on_sale_end_date,
            sortStartDate=getattr(param, "start_date", None),
            sortEndDate=getattr(param, "end_date", None),
            yearSeason=param.year_season,
            hasCardSale=param.has_card_sale,
            hasLiveSale=param.has_live_sale,
            hasVideoSale=param.has_video_sale,
            onlyLiveSale=param.only_live_sale,
            onlyVideoSale=param.only_video_sale,
            onlyCardSale=param.only_card_sale,
            isMonitorShop=param.is_monitor_shop or 0,
            isMonitorStreamer=param.is_monitor_streamer or 0,
            sortField=param.sort_field if param.sort_field != "默认" else None,
            limit=param.limit or 6000,
            saleStyle=param.sale_style,
            brand=param.brand if param.brand else None,
            shopId=None,  # 店铺ID在调用前单独设置
        )


class DouyiGoodsEntity(BaseModel):
    """抖衣商品实体"""

    class Config:
        populate_by_name = True

    goods_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("goodsId", "itemId"),
        serialization_alias="goodsId",
        description="商品ID（兼容 goodsId/itemId）",
    )
    title: Optional[str] = Field(default=None, alias="title")
    pic_url: Optional[str] = Field(default=None, alias="picUrl")
    price: Optional[int] = Field(default=None, alias="price")
    sale_volume_30day: Optional[int] = Field(default=None, alias="saleVolume30day")
    sale_amount_30day: Optional[int] = Field(default=None, alias="saleAmount30day")
    first_record_date: Optional[str] = Field(default=None, alias="firstRecordDate")
    shop_name: Optional[str] = Field(default=None, alias="shopName")


# ========== 深度思考工作流请求模型 ==========

class ZhiyiSaleTrendRequest(BaseModel):
    """销售趋势请求 - 对应 /v1-6-2/trend/sale-trend"""
    model_config = ConfigDict(populate_by_name=True)

    shop_id: int = Field(alias="shopId")
    distribution: int = Field(description="1=销量, 3=销售额")
    granularity: int = Field(description="2=周, 3=月")
    sb_type: str = Field(default="all", alias="sbType")
    has_last_year: bool = Field(default=False, alias="hasLastYear")
    start_date: str = Field(alias="startDate")
    end_date: str = Field(alias="endDate")


class ZhiyiPriceRangeTrendRequest(BaseModel):
    """价格带趋势请求 - 对应 /monitor-shop/v2-3-1/trend/price-range-trend"""
    model_config = ConfigDict(populate_by_name=True)

    root_category_id: int = Field(alias="rootCategoryId")
    category_id: Optional[int] = Field(default=None, alias="categoryId")
    start_date: str = Field(alias="startDate")
    end_date: str = Field(alias="endDate")
    query_type: str = Field(default="monitor", alias="queryType")
    shop_id: int = Field(alias="shopId")
    entrance: str = Field(default="1")
    price_type: str = Field(default="1", alias="priceType")
    distribution: int = Field(default=1)
    granularity: int = Field(description="2=周, 3=月")


class ZhiyiPropertyTrendRequest(BaseModel):
    """属性趋势请求 - 对应 /v1-6-0/monitor/trend/property-trend"""
    model_config = ConfigDict(populate_by_name=True)

    root_category_id: int = Field(alias="rootCategoryId")
    category_id: Optional[str] = Field(default=None, alias="categoryId")
    start_date: str = Field(alias="startDate")
    end_date: str = Field(alias="endDate")
    query_type: str = Field(default="monitor", alias="queryType")
    shop_id: int = Field(alias="shopId")
    entrance: str = Field(default="1")
    price_type: str = Field(default="1", alias="priceType")
    distribution: int = Field(default=1)
    granularity: int = Field(description="2=周, 3=月")
    property_type: int = Field(default=3, alias="propertyType")
    property_name: str = Field(alias="propertyName", description="属性名称，如'颜色'")


class ZhiyiCategoryTrendRequest(BaseModel):
    """品类趋势请求 - 对应 /v1-6-0/monitor/trend/category-trend"""
    model_config = ConfigDict(populate_by_name=True)

    start_date: str = Field(alias="startDate")
    end_date: str = Field(alias="endDate")
    shop_id: int = Field(alias="shopId")
    entrance: str = Field(default="1")
    query_type: str = Field(default="monitor", alias="queryType")
    root_category_id: Optional[int] = Field(default=None, alias="rootCategoryId")
    category_id: Optional[int] = Field(default=None, alias="categoryId")
    distribution: int = Field(default=1)
    terms_agg_field: str = Field(default="category_id", alias="termsAggField")
    granularity: int = Field(description="2=周, 3=月")
    limit: int = Field(default=99)


class ZhiyiShopHotItemRequest(BaseModel):
    """店铺热销商品请求 - 对应 /v2-0-x/item/shop/hot-item-list"""
    model_config = ConfigDict(populate_by_name=True)

    page_size: int = Field(default=10, alias="pageSize")
    page_no: int = Field(default=1, alias="pageNo")
    limit: int = Field(default=10)
    shop_id: int = Field(alias="shopId")
    sort_field: str = Field(default="aggSaleVolume", alias="sortField")
    sort_type: str = Field(default="desc", alias="sortType")
    root_category_id_list: List[int] = Field(alias="rootCategoryIdList")
    category_id_list: Optional[List[int]] = Field(default=None, alias="categoryIdList")
    start_date: str = Field(alias="startDate")
    end_date: str = Field(alias="endDate")
    top: int = Field(default=10)
    type: int = Field(default=1)
    indeterminate_root_category_id_list: List[int] = Field(
        default_factory=list, alias="indeterminateRootCategoryIdList"
    )


class ZhiyiPropertyTopRequest(BaseModel):
    """属性列表请求 - 对应 /v1-6-0/item/taobao-item-property-top"""
    model_config = ConfigDict(populate_by_name=True)

    root_category_id: int = Field(alias="rootCategoryId")
    category_id: Optional[int] = Field(default=None, alias="categoryId")
    entrance: int = Field(default=2)
    root_category_id_list: Optional[int] = Field(default=None, alias="rootCategoryIdList")


# ========== 知衣大盘分析(HYDC)请求模型 ==========

class ZhiyiHydcTrendRequest(BaseModel):
    """大盘销售趋势请求 - 对应 /v1-6-2/hydc/trend/sale-trend"""
    model_config = ConfigDict(populate_by_name=True)

    root_category_id: int = Field(alias="rootCategoryId")
    category_id: Optional[int] = Field(default=None, alias="categoryId")
    distribution: int = Field(description="1=销量, 3=销售额")
    granularity: int = Field(description="2=周, 3=月")
    sb_type: str = Field(default="all", alias="sbType")
    has_last_year: bool = Field(default=False, alias="hasLastYear")
    start_date: str = Field(alias="startDate")
    end_date: str = Field(alias="endDate")


class ZhiyiHydcPriceRangeRequest(BaseModel):
    """大盘价格带请求 - 对应 /v2-3-1/hydc/trend/price-range-trend"""
    model_config = ConfigDict(populate_by_name=True)

    root_category_id: int = Field(alias="rootCategoryId")
    category_id: Optional[int] = Field(default=None, alias="categoryId")
    start_date: str = Field(alias="startDate")
    end_date: str = Field(alias="endDate")
    query_type: str = Field(default="hydc", alias="queryType")
    entrance: str = Field(default="1")
    price_type: str = Field(default="1", alias="priceType")
    distribution: int = Field(default=1)
    granularity: int = Field(description="2=周, 3=月")


class ZhiyiHydcPropertyRequest(BaseModel):
    """大盘属性/颜色请求 - 对应 /v1-6-0/hydc/trend/property-trend"""
    model_config = ConfigDict(populate_by_name=True)

    root_category_id: int = Field(alias="rootCategoryId")
    category_id: Optional[int] = Field(default=None, alias="categoryId")
    start_date: str = Field(alias="startDate")
    end_date: str = Field(alias="endDate")
    query_type: str = Field(default="hydc", alias="queryType")
    entrance: str = Field(default="1")
    price_type: str = Field(default="1", alias="priceType")
    distribution: int = Field(default=1)
    granularity: int = Field(description="2=周, 3=月")
    property_type: int = Field(default=3, alias="propertyType")
    property_name: str = Field(alias="propertyName", description="属性名称，如'颜色'")


class ZhiyiHydcBrandRequest(BaseModel):
    """大盘品牌请求 - 对应 /v1-6-0/hydc/trend/brand-trend"""
    model_config = ConfigDict(populate_by_name=True)

    root_category_id: int = Field(alias="rootCategoryId")
    category_id: Optional[int] = Field(default=None, alias="categoryId")
    start_date: str = Field(alias="startDate")
    end_date: str = Field(alias="endDate")
    distribution: int = Field(default=1)
    granularity: int = Field(description="2=周, 3=月")
    query_type: str = Field(default="hydc", alias="queryType")
    entrance: str = Field(default="1")


class ZhiyiHydcTopItemRequest(BaseModel):
    """大盘热销商品请求 - 对应 /v2-0-x/hydc/item/hot-item-list"""
    model_config = ConfigDict(populate_by_name=True)

    page_size: int = Field(default=10, alias="pageSize")
    page_no: int = Field(default=1, alias="pageNo")
    limit: int = Field(default=10)
    sort_field: str = Field(default="aggSaleVolume", alias="sortField")
    sort_type: str = Field(default="desc", alias="sortType")
    root_category_id_list: List[int] = Field(alias="rootCategoryIdList")
    category_id_list: Optional[List[int]] = Field(default=None, alias="categoryIdList")
    start_date: str = Field(alias="startDate")
    end_date: str = Field(alias="endDate")
    top: int = Field(default=10)
    type: int = Field(default=1)
    indeterminate_root_category_id_list: List[int] = Field(
        default_factory=list, alias="indeterminateRootCategoryIdList"
    )


class ZhiyiHydcCategoryRequest(BaseModel):
    """大盘品类趋势请求 - 对应 /v1-6-0/hydc/trend/category-trend"""
    model_config = ConfigDict(populate_by_name=True)

    start_date: str = Field(alias="startDate")
    end_date: str = Field(alias="endDate")
    entrance: str = Field(default="1")
    query_type: str = Field(default="hydc", alias="queryType")
    root_category_id: Optional[int] = Field(default=None, alias="rootCategoryId")
    category_id: Optional[int] = Field(default=None, alias="categoryId")
    distribution: int = Field(default=1)
    terms_agg_field: str = Field(default="category_id", alias="termsAggField")
    granularity: int = Field(description="2=周, 3=月")
    limit: int = Field(default=99)


# ========== 深度思考工作流原始响应模型 ==========

class SaleTrendTrendDTO(BaseModel):
    """销售趋势-趋势项"""
    model_config = ConfigDict(populate_by_name=True)

    granularity_date: Optional[str] = Field(default=None, alias="granularityDate")
    insert_date: Optional[str] = Field(default=None, alias="insertDate")
    start_time: Optional[str] = Field(default=None, alias="startTime")
    end_time: Optional[str] = Field(default=None, alias="endTime")
    daily_sum_day_sales_volume: Optional[int] = Field(default=None, alias="dailySumDaySalesVolume")
    daily_all_sales_volume: Optional[int] = Field(default=None, alias="dailyAllSalesVolume")
    daily_sum_day_sale: Optional[int] = Field(default=None, alias="dailySumDaySale")
    daily_new_item_sales_volume: Optional[int] = Field(default=None, alias="dailyNewItemSalesVolume")
    daily_shelves_count: Optional[int] = Field(default=None, alias="dailyShelvesCount")
    onsale_sku_num: Optional[int] = Field(default=None, alias="onsaleSkuNum")
    onsale_stock: Optional[int] = Field(default=None, alias="onsaleStock")
    new_sku_num: Optional[int] = Field(default=None, alias="newSkuNum")


class SaleTrendRawResult(BaseModel):
    """销售趋势原始结果"""
    model_config = ConfigDict(populate_by_name=True)

    days: Optional[int] = Field(default=None, alias="days")
    sum_sales_volume: Optional[int] = Field(default=None, alias="sumSalesVolume")
    all_sales_volume: Optional[int] = Field(default=None, alias="allSalesVolume")
    sum_sale: Optional[int] = Field(default=None, alias="sumSale")
    sum_shelves_count: Optional[int] = Field(default=None, alias="sumShelvesCount")
    sum_new_item_sales_volume: Optional[int] = Field(default=None, alias="sumNewItemSalesVolume")
    onsale_sku_num: Optional[int] = Field(default=None, alias="onsaleSkuNum")
    onsale_stock: Optional[int] = Field(default=None, alias="onsaleStock")
    new_sku_num: Optional[int] = Field(default=None, alias="newSkuNum")
    new_stock: Optional[int] = Field(default=None, alias="newStock")
    top3_category_list: Optional[List[str]] = Field(default=None, alias="top3CategoryList")
    has_promotion: Optional[bool] = Field(default=None, alias="hasPromotion")
    has_shuang11_presale: Optional[bool] = Field(default=None, alias="hasShuang11Presale")
    trend_dtos: List[SaleTrendTrendDTO] = Field(default_factory=list, alias="trendDTOS")


class SaleTrendRawResponse(BaseModel):
    """销售趋势原始响应"""
    success: bool = True
    result: SaleTrendRawResult


class PriceRangeSubItem(BaseModel):
    """价格带-子项"""
    model_config = ConfigDict(populate_by_name=True)

    level: Optional[str] = Field(default=None, alias="level")
    left_price: Optional[int] = Field(default=None, alias="leftPrice")
    right_price: Optional[int] = Field(default=None, alias="rightPrice")
    sum_day_sales_volume_by_range: Optional[int] = Field(default=None, alias="sumDaySalesVolumeByRange")
    sum_day_sale_by_range: Optional[int] = Field(default=None, alias="sumDaySaleByRange")
    all_sales_volume_by_range: Optional[int] = Field(default=None, alias="allSalesVolumeByRange")
    sum_day_sales_volume_by_range_rate: Optional[float] = Field(default=None, alias="sumDaySalesVolumeByRangeRate")
    sale_date: Optional[str] = Field(default=None, alias="saleDate")
    sub_list: Optional[List[PriceRangeSubItem]] = Field(default=None, alias="subList")


class PriceRangeRawResponse(BaseModel):
    """价格带原始响应"""
    success: bool = True
    result: List[PriceRangeSubItem]


class PropertyTrendTrendDTO(BaseModel):
    """属性趋势-趋势项"""
    model_config = ConfigDict(populate_by_name=True)

    granularity_date: Optional[str] = Field(default=None, alias="granularityDate")
    insert_date: Optional[str] = Field(default=None, alias="insertDate")
    start_time: Optional[str] = Field(default=None, alias="startTime")
    end_time: Optional[str] = Field(default=None, alias="endTime")
    daily_sum_day_sales_volume: Optional[int] = Field(default=None, alias="dailySumDaySalesVolume")
    daily_all_sales_volume: Optional[int] = Field(default=None, alias="dailyAllSalesVolume")
    daily_sum_day_sale: Optional[int] = Field(default=None, alias="dailySumDaySale")
    daily_new_item_sales_volume: Optional[int] = Field(default=None, alias="dailyNewItemSalesVolume")
    daily_shelves_count: Optional[int] = Field(default=None, alias="dailyShelvesCount")
    onsale_sku_num: Optional[int] = Field(default=None, alias="onsaleSkuNum")
    onsale_stock: Optional[int] = Field(default=None, alias="onsaleStock")
    new_sku_num: Optional[int] = Field(default=None, alias="newSkuNum")


class PropertyTrendRawItem(BaseModel):
    """属性趋势原始项"""
    model_config = ConfigDict(populate_by_name=True)

    property_name: Optional[str] = Field(default=None, alias="propertyName")
    property_value: Optional[str] = Field(default=None, alias="propertyValue")
    color_value: Optional[str] = Field(default=None, alias="colorValue")
    other_flag: bool = Field(default=False, alias="otherFlag")
    sum_day_sales_volume_by_property: Optional[int] = Field(default=None, alias="sumDaySalesVolumeByProperty")
    all_sales_volume_by_property: Optional[int] = Field(default=None, alias="allSalesVolumeByProperty")
    sum_day_sale_by_property: Optional[int] = Field(default=None, alias="sumDaySaleByProperty")
    sum_day_sales_volume_by_property_rate: Optional[float] = Field(default=None, alias="sumDaySalesVolumeByPropertyRate")
    all_sales_volume_by_property_rate: Optional[float] = Field(default=None, alias="allSalesVolumeByPropertyRate")
    sum_day_sale_by_property_rate: Optional[float] = Field(default=None, alias="sumDaySaleByPropertyRate")
    trend_dtos: List[PropertyTrendTrendDTO] = Field(default_factory=list, alias="trendDTOS")


class PropertyTrendRawResponse(BaseModel):
    """属性趋势原始响应"""
    success: bool = True
    result: List[PropertyTrendRawItem]


class ShopHotItemRawItem(BaseModel):
    """热销商品原始项"""
    model_config = ConfigDict(populate_by_name=True)

    item_id: Optional[int] = Field(default=None, alias="itemId")
    brand: Optional[str] = Field(default=None, alias="brand")
    title: Optional[str] = Field(default=None, alias="title")
    pic_url: Optional[str] = Field(default=None, alias="picUrl")
    category_name: Optional[str] = Field(default=None, alias="categoryName")
    category_detail: Optional[str] = Field(default=None, alias="categoryDetail")
    total_sale_amount: Optional[int] = Field(default=None, alias="totalSaleAmount")
    total_sale_volume: Optional[int] = Field(default=None, alias="totalSaleVolume")
    agg_sale_volume: Optional[int] = Field(default=None, alias="aggSaleVolume")
    agg_sale_amount: Optional[int] = Field(default=None, alias="aggSaleAmount")
    max_s_price: Optional[int] = Field(default=None, alias="maxSPrice")
    min_price: Optional[str] = Field(default=None, alias="minPrice")


class ShopHotItemRawResult(BaseModel):
    """热销商品原始结果"""
    model_config = ConfigDict(populate_by_name=True)

    start: Optional[int] = Field(default=0, alias="start")
    page_size: Optional[int] = Field(default=10, alias="pageSize")
    result_count: Optional[int] = Field(default=0, alias="resultCount")
    result_list: List[ShopHotItemRawItem] = Field(default_factory=list, alias="resultList")


class ShopHotItemRawResponse(BaseModel):
    """热销商品原始响应"""
    success: bool = True
    result: ShopHotItemRawResult


class CategoryTrendTrendDTO(BaseModel):
    """品类趋势-趋势项"""
    model_config = ConfigDict(populate_by_name=True)

    granularity_date: Optional[str] = Field(default=None, alias="granularityDate")
    insert_date: Optional[str] = Field(default=None, alias="insertDate")
    insert_time: Optional[str] = Field(default=None, alias="insertTime")
    start_time: Optional[str] = Field(default=None, alias="startTime")
    end_time: Optional[str] = Field(default=None, alias="endTime")
    daily_sum_day_sales_volume: Optional[int] = Field(default=None, alias="dailySumDaySalesVolume")
    daily_all_sales_volume: Optional[int] = Field(default=None, alias="dailyAllSalesVolume")
    daily_sum_day_sale: Optional[int] = Field(default=None, alias="dailySumDaySale")
    daily_new_item_sales_volume: Optional[int] = Field(default=None, alias="dailyNewItemSalesVolume")
    daily_shelves_count: Optional[int] = Field(default=None, alias="dailyShelvesCount")
    onsale_sku_num: Optional[int] = Field(default=None, alias="onsaleSkuNum")
    onsale_stock: Optional[int] = Field(default=None, alias="onsaleStock")
    new_sku_num: Optional[int] = Field(default=None, alias="newSkuNum")
    percentage: Optional[float] = Field(default=None, alias="percentage")


class CategoryTrendRawItem(BaseModel):
    """品类趋势原始项"""
    model_config = ConfigDict(populate_by_name=True)

    shop_id: Optional[int] = Field(default=None, alias="shopId")
    root_category_id: Optional[int] = Field(default=None, alias="rootCategoryId")
    root_category_name: Optional[str] = Field(default=None, alias="rootCategoryName")
    category_id: Optional[int] = Field(default=None, alias="categoryId")
    category_name: Optional[str] = Field(default=None, alias="categoryName")
    first_category_name: Optional[str] = Field(default=None, alias="firstCategoryName")
    second_category_name: Optional[str] = Field(default=None, alias="secondCategoryName")
    all_sales_volume_by_category: Optional[int] = Field(default=None, alias="allSalesVolumeByCategory")
    sum_day_sales_volume_by_category: Optional[int] = Field(default=None, alias="sumDaySalesVolumeByCategory")
    sum_day_sale_by_category: Optional[int] = Field(default=None, alias="sumDaySaleByCategory")
    new_item_sales_volume_by_category: Optional[int] = Field(default=None, alias="newItemSalesVolumeByCategory")
    shelves_count_by_category: Optional[int] = Field(default=None, alias="shelvesCountByCategory")
    all_sales_volume_by_category_rate: Optional[float] = Field(default=None, alias="allSalesVolumeByCategoryRate")
    sum_day_sales_volume_by_category_rate: Optional[float] = Field(default=None, alias="sumDaySalesVolumeByCategoryRate")
    sum_day_sale_by_category_rate: Optional[float] = Field(default=None, alias="sumDaySaleByCategoryRate")
    onsale_sku_num: Optional[int] = Field(default=None, alias="onsaleSkuNum")
    onsale_stock: Optional[int] = Field(default=None, alias="onsaleStock")
    new_sku_num: Optional[int] = Field(default=None, alias="newSkuNum")
    trend_dtos: List[CategoryTrendTrendDTO] = Field(default_factory=list, alias="trendDTOS")


class CategoryTrendRawResponse(BaseModel):
    """品类趋势原始响应"""
    success: bool = True
    result: List[CategoryTrendRawItem]


class PropertyTopRawItem(BaseModel):
    """属性列表原始项"""
    model_config = ConfigDict(populate_by_name=True)

    property_name: Optional[str] = Field(default=None, alias="propertyName")


class PropertyTopRawResponse(BaseModel):
    """属性列表原始响应"""
    success: bool = True
    result: List[PropertyTopRawItem]


# ========== 深度思考工作流清洗后响应模型 ==========

class SaleTrendSlimItem(BaseModel):
    """趋势数据精简项"""
    model_config = ConfigDict(populate_by_name=True)

    granularity_date: Optional[str] = Field(default=None, alias="granularityDate")
    daily_sum_day_sales_volume: int = Field(default=0, alias="dailySumDaySalesVolume")
    daily_sum_day_sale: int = Field(default=0, alias="dailySumDaySale")
    daily_shelves_count: int = Field(default=0, alias="dailyShelvesCount")
    daily_new_item_sales_volume: int = Field(default=0, alias="dailyNewItemSalesVolume")


class SaleTrendSlimResult(BaseModel):
    """销售趋势精简结果"""
    model_config = ConfigDict(populate_by_name=True)

    sum_sales_volume: Optional[int] = Field(default=None, alias="sumSalesVolume")
    sum_sale: Optional[int] = Field(default=None, alias="sumSale")
    all_sales_volume: Optional[int] = Field(default=None, alias="allSalesVolume")
    sum_shelves_count: Optional[int] = Field(default=None, alias="sumShelvesCount")
    sum_new_item_sales_volume: Optional[int] = Field(default=None, alias="sumNewItemSalesVolume")
    onsale_sku_num: Optional[int] = Field(default=None, alias="onsaleSkuNum")
    onsale_stock: Optional[int] = Field(default=None, alias="onsaleStock")
    new_sku_num: Optional[int] = Field(default=None, alias="newSkuNum")
    top3_category_list: Optional[List[str]] = Field(default=None, alias="top3CategoryList")
    has_promotion: Optional[bool] = Field(default=None, alias="hasPromotion")
    has_shuang11_presale: Optional[bool] = Field(default=None, alias="hasShuang11Presale")
    trend_dtos: List[SaleTrendSlimItem] = Field(default_factory=list, alias="trendDTOS")


class SaleTrendCleanResponse(BaseModel):
    """销售趋势清洗后响应"""
    success: bool = True
    result: SaleTrendSlimResult


class PriceRangeSlimItem(BaseModel):
    """价格带精简项"""
    model_config = ConfigDict(populate_by_name=True)

    left_price: int = Field(alias="leftPrice")
    right_price: int = Field(alias="rightPrice")
    sales_volume: int = Field(default=0, alias="salesVolume")
    rate: Optional[float] = Field(default=None, alias="rate")


class PriceRangeSlimResult(BaseModel):
    """价格带精简结果"""
    model_config = ConfigDict(populate_by_name=True)

    price_slim: List[PriceRangeSlimItem] = Field(default_factory=list)


class PriceRangeCleanResponse(BaseModel):
    """价格带清洗后响应"""
    success: bool = True
    result: PriceRangeSlimResult


class ColorSlimItem(BaseModel):
    """颜色/属性精简项"""
    model_config = ConfigDict(populate_by_name=True)

    property_value: Optional[str] = Field(default=None, alias="propertyValue")
    sales_volume: int = Field(default=0, alias="salesVolume")
    sales_amount: Optional[int] = Field(default=None, alias="salesAmount")
    rate: Optional[float] = Field(default=None, alias="rate")
    other_flag: bool = Field(default=False, alias="otherFlag")


class ColorSlimResult(BaseModel):
    """颜色精简结果"""
    model_config = ConfigDict(populate_by_name=True)

    color_slim: List[ColorSlimItem] = Field(default_factory=list)


class ColorCleanResponse(BaseModel):
    """颜色清洗后响应"""
    success: bool = True
    result: ColorSlimResult


class Top10ItemSlim(BaseModel):
    """Top10商品精简项"""
    model_config = ConfigDict(populate_by_name=True)

    rank: int
    item_id: Optional[str] = Field(default=None, alias="itemId")
    title: Optional[str] = Field(default=None)
    pic_url: Optional[str] = Field(default=None, alias="picUrl")
    category_name: Optional[str] = Field(default=None, alias="categoryName")
    category_detail: Optional[str] = Field(default=None, alias="categoryDetail")
    sales_volume: int = Field(default=0, alias="salesVolume")
    sales_amount: int = Field(default=0, alias="salesAmount")
    min_price: Optional[int] = Field(default=None, alias="minPrice")
    max_s_price: Optional[int] = Field(default=None, alias="maxSPrice")


class Top10ItemsSlimResult(BaseModel):
    """Top10商品精简结果"""
    model_config = ConfigDict(populate_by_name=True)

    start: int = Field(default=0)
    page_size: int = Field(default=10, alias="pageSize")
    result_count: int = Field(default=0, alias="resultCount")
    top10_slim: List[Top10ItemSlim] = Field(default_factory=list)


class Top10ItemsCleanResponse(BaseModel):
    """Top10商品清洗后响应"""
    success: bool = True
    result: Top10ItemsSlimResult


class CategoryTrendSlimItem(BaseModel):
    """品类趋势精简项"""
    model_config = ConfigDict(populate_by_name=True)

    rank: int
    category_id: Optional[str] = Field(default=None, alias="categoryId")
    first_category_name: Optional[str] = Field(default=None, alias="firstCategoryName")
    second_category_name: Optional[str] = Field(default=None, alias="secondCategoryName")
    category_path: Optional[str] = Field(default=None, alias="categoryPath")
    all_sales_volume: int = Field(default=0, alias="allSalesVolume")
    sum_day_sales_volume: int = Field(default=0, alias="sumDaySalesVolume")
    sum_day_sales_amount: int = Field(default=0, alias="sumDaySalesAmount")
    new_item_sales_volume: int = Field(default=0, alias="newItemSalesVolume")
    shelves_count: Optional[int] = Field(default=None, alias="shelvesCount")
    all_sales_volume_rate: Optional[float] = Field(default=None, alias="allSalesVolumeRate")
    sum_day_sales_volume_rate: Optional[float] = Field(default=None, alias="sumDaySalesVolumeRate")
    sum_day_sales_amount_rate: Optional[float] = Field(default=None, alias="sumDaySalesAmountRate")
    onsale_sku_num: Optional[int] = Field(default=None, alias="onsaleSkuNum")
    onsale_stock: Optional[int] = Field(default=None, alias="onsaleStock")
    new_sku_num: Optional[int] = Field(default=None, alias="newSkuNum")
    trend: List[Dict[str, Any]] = Field(default_factory=list)


class CategoryTrendSlimResult(BaseModel):
    """品类趋势精简结果"""
    model_config = ConfigDict(populate_by_name=True)

    result_count: int = Field(default=0, alias="resultCount")
    top10_slim: List[CategoryTrendSlimItem] = Field(default_factory=list)


class CategoryTrendCleanResponse(BaseModel):
    """品类趋势清洗后响应"""
    success: bool = True
    result: CategoryTrendSlimResult


# ========== 大盘品牌数据模型 ==========

class BrandRawItem(BaseModel):
    """品牌原始项"""
    model_config = ConfigDict(populate_by_name=True)

    brand_name: Optional[str] = Field(default=None, alias="brandName")
    sum_day_sales_volume_by_brand: Optional[int] = Field(default=None, alias="sumDaySalesVolumeByBrand")
    sum_day_sale_by_brand: Optional[int] = Field(default=None, alias="sumDaySaleByBrand")
    sum_day_sales_volume_by_brand_rate: Optional[float] = Field(default=None, alias="sumDaySalesVolumeByBrandRate")
    other_flag: bool = Field(default=False, alias="otherFlag")


class BrandRawResponse(BaseModel):
    """品牌原始响应"""
    success: bool = True
    result: List[BrandRawItem]


class BrandSlimItem(BaseModel):
    """品牌精简项"""
    model_config = ConfigDict(populate_by_name=True)

    brand_name: Optional[str] = Field(default=None, alias="brandName")
    sales_volume: int = Field(default=0, alias="salesVolume")
    sales_amount: Optional[int] = Field(default=None, alias="salesAmount")
    rate: Optional[float] = None
    other_flag: bool = Field(default=False, alias="otherFlag")


class BrandSlimResult(BaseModel):
    """品牌精简结果"""
    model_config = ConfigDict(populate_by_name=True)

    brand_slim: List[BrandSlimItem] = Field(default_factory=list)


class BrandCleanResponse(BaseModel):
    """品牌清洗后响应"""
    success: bool = True
    result: BrandSlimResult


# ========== 抖衣深度思考工作流请求模型 ==========

class DouyiTrendAnalysisRequest(BaseModel):
    """抖衣趋势分析请求 - 对应 /v1-2-4/douyin/item-analysis/trend-analysis"""
    model_config = ConfigDict(populate_by_name=True)

    # 查询类型：categoryAnalysis(品类分析), priceAnalysis(价格分析), propertyAnalysis(属性分析)
    query_type: str = Field(alias="queryType", description="查询类型")
    root_category_id: int = Field(alias="rootCategoryId", description="根类目ID")
    category_id_list: List[int] | None = Field(default=None, alias="categoryIdList", description="类目ID列表")
    start_date: str = Field(alias="startDate", description="开始日期")
    end_date: str = Field(alias="endDate", description="结束日期")
    trend_type: str = Field(alias="trendType", description="趋势类型：weekly/monthly")

    # 带货类型
    front_title_tab_value: str = Field(default="windowGoods", alias="frontTitleTabValue",
                                        description="带货类型：windowGoods/liveGoods/videoGoods/cardGoods")
    relate_type: str = Field(default="windowGoods", alias="relateType", description="关联类型")
    target: str = Field(default="saleVolume", alias="target", description="目标指标：saleVolume/saleAmount")

    # 价格筛选
    min_price: int | None = Field(default=None, alias="minPrice")
    max_price: int | None = Field(default=None, alias="maxPrice")
    custom_price_range: List[Dict] | None = Field(default=None, alias="customPriceRange", description="自定义价格范围")
    price_type: str = Field(default="goodsPrice", alias="priceType")
    price_range_list: List[Dict[str, int]] | None = Field(
        default=None,
        alias="priceRangeList",
        description="价格带列表，格式: [{'minPrice':0,'maxPrice':5000}, ...]，单位分"
    )

    # 属性筛选
    front_property: List[str] | None = Field(default=None, alias="frontProperty", description="前端属性名列表")
    properties: List[str] | None = Field(default=None, alias="properties", description="属性值列表，格式：属性名:属性值")
    brand_list: List[str] | None = Field(default=None, alias="brandList")
    delivery_address: str | None = Field(default=None, alias="deliveryAddress")
    goods_type_list: List[str] | None = Field(default=None, alias="goodsTypeList")

    # 其他筛选
    front_date_picker_status: str = Field(default="recent", alias="frontDatePickerStatus")
    other_sale: int | None = Field(default=None, alias="otherSale")
    self_sale: int | None = Field(default=None, alias="selfSale")

    # 达人筛选
    min_live_like_num: int | None = Field(default=None, alias="minLiveLikeNum")
    max_live_like_num: int | None = Field(default=None, alias="maxLiveLikeNum")
    min_live_peak_audi_num: int | None = Field(default=None, alias="minLivePeakAudiNum")
    max_live_peak_audi_num: int | None = Field(default=None, alias="maxLivePeakAudiNum")
    max_live_audi_num: int | None = Field(default=None, alias="maxLiveAudiNum")
    min_live_audi_num: int | None = Field(default=None, alias="minLiveAudiNum")
    min_fans_num: int | None = Field(default=None, alias="minFansNum")
    max_fans_num: int | None = Field(default=None, alias="maxFansNum")
    min_live_sale_volume_30day: int | None = Field(default=None, alias="minLiveSaleVolume30day")
    max_live_sale_volume_30day: int | None = Field(default=None, alias="maxLiveSaleVolume30day")
    hot_sale_root_category_30day: str | None = Field(default=None, alias="hotSaleRootCategory30day")
    hot_sale_category_30day: str | None = Field(default=None, alias="hotSaleCategory30day")


class DouyiTopItemsRequest(BaseModel):
    """抖衣Top商品请求 - 对应 /v1-2-4/douyin/item-analysis/item-analysis-top-list"""
    model_config = ConfigDict(populate_by_name=True)

    relate_type: str = Field(default="windowGoods", alias="relateType")
    start_date: str = Field(alias="startDate")
    end_date: str = Field(alias="endDate")
    root_category_id: int = Field(alias="rootCategoryId")
    category_id_list: List[int] | None = Field(default=None, alias="categoryIdList")
    properties: List[str] | None = Field(default=None, alias="properties")
    brand_list: List[str] | None = Field(default=None, alias="brandList")
    delivery_address: str | None = Field(default=None, alias="deliveryAddress")
    goods_type_list: List[str] | None = Field(default=None, alias="goodsTypeList")
    target: str = Field(default="cardSaleVolume", alias="target")
    query_type: str = Field(default="itemAnalysis", alias="queryType")
    limit: int = Field(default=10)
    page_no: int = Field(default=1, alias="pageNo")
    page_size: int = Field(default=20, alias="pageSize")

    # 价格筛选
    min_price: int | None = Field(default=None, alias="minPrice")
    max_price: int | None = Field(default=None, alias="maxPrice")


class DouyiPropertySelectorRequest(BaseModel):
    """抖衣属性选择器请求 - 对应 /douyin-common/item/item-property-selector-list"""
    model_config = ConfigDict(populate_by_name=True)

    root_category_id: int = Field(alias="rootCategoryId")
    category_id: int | None = Field(default=None, alias="categoryId")


# ========== 抖衣深度思考工作流原始响应模型 ==========

class DouyiTrendSubListItem(BaseModel):
    """抖衣趋势分析-子列表项（时间周期数据）"""
    model_config = ConfigDict(populate_by_name=True)

    key: str | None = Field(default=None, description="时间标识，如 '2025-01' 或 '2025-49'")
    name: str | None = Field(default=None, description="时间范围名称，如 '2025-11-13~2025-11-16'")
    sum: int | None = Field(default=None, description="总销量/总销售额")
    live_sum: int | None = Field(default=None, alias="liveSum", description="直播销量")
    video_sum: int | None = Field(default=None, alias="videoSum", description="作品销量")
    card_sum: int | None = Field(default=None, alias="cardSum", description="商品卡销量")
    percentage: float | None = Field(default=None, description="占比")
    # 同比数据
    yoy_sum: int | None = Field(default=None, alias="yoySum", description="同比销量")
    yoy_live_sum: int | None = Field(default=None, alias="yoyLiveSum", description="同比直播销量")
    yoy_video_sum: int | None = Field(default=None, alias="yoyVideoSum", description="同比作品销量")
    yoy_card_sum: int | None = Field(default=None, alias="yoyCardSum", description="同比商品卡销量")


class DouyiTrendSubItem(BaseModel):
    """抖衣趋势分析-品类/属性/价格带项"""
    model_config = ConfigDict(populate_by_name=True)

    key: str | None = Field(default=None, description="标识，如品类ID或价格范围")
    name: str | None = Field(default=None, description="名称")
    sum: int | None = Field(default=None, description="总计")
    percentage: float | None = Field(default=None, description="占比")
    sub_list: List[DouyiTrendSubListItem] | None = Field(default=None, alias="subList", description="时间周期数据")
    # 分渠道销量
    live_sum: int | None = Field(default=None, alias="liveSum", description="直播销量")
    video_sum: int | None = Field(default=None, alias="videoSum", description="作品销量")
    card_sum: int | None = Field(default=None, alias="cardSum", description="商品卡销量")
    # 同比和环比
    yoy_sum: int | None = Field(default=None, alias="yoySum", description="同比总量")
    mom_ratio: float | None = Field(default=None, alias="momRatio", description="环比")
    yoy_ratio: float | None = Field(default=None, alias="yoyRatio", description="同比")


class DouyiTrendAnalysisRawResult(BaseModel):
    """抖衣趋势分析原始结果"""
    model_config = ConfigDict(populate_by_name=True)

    key: str | None = Field(default=None, description="标识")
    name: str | None = Field(default=None, description="名称")
    sum: int | None = Field(default=None, description="总计")
    sub_list: List[DouyiTrendSubItem] | None = Field(default=None, alias="subList", description="分类数据列表")
    # 分渠道销量
    live_sum: int | None = Field(default=None, alias="liveSum", description="直播销量")
    video_sum: int | None = Field(default=None, alias="videoSum", description="作品销量")
    card_sum: int | None = Field(default=None, alias="cardSum", description="商品卡销量")
    # 同比和环比
    mom_ratio: float | None = Field(default=None, alias="momRatio", description="环比")
    yoy_ratio: float | None = Field(default=None, alias="yoyRatio", description="同比")
    percentage: float | None = Field(default=None, description="占比")


class DouyiTrendAnalysisRawResponse(BaseModel):
    """抖衣趋势分析原始响应"""
    success: bool = True
    result: DouyiTrendAnalysisRawResult | None = None


class DouyiGoodsNested(BaseModel):
    """抖音商品嵌套数据"""
    model_config = ConfigDict(populate_by_name=True)

    relate_live_num: int | None = Field(default=None, alias="relateLiveNum", description="关联直播数")
    relate_product_num: int | None = Field(default=None, alias="relateProductNum", description="关联商品数")
    comment_num_30day: int | None = Field(default=None, alias="commentNum30day", description="30天评论数")
    total_view_num: int | None = Field(default=None, alias="totalViewNum", description="总浏览数")


class DouyiTopItemRawItem(BaseModel):
    """抖衣Top商品原始项"""
    model_config = ConfigDict(populate_by_name=True)

    # 基础信息
    item_id: str | None = Field(default=None, alias="itemId", description="商品ID")
    title: str | None = Field(default=None, description="商品标题")
    goods_title: str | None = Field(default=None, alias="goodsTitle", description="商品标题（别名）")
    pic_url: str | None = Field(default=None, alias="picUrl", description="商品主图")
    pic_url_list: List[str] | None = Field(default=None, alias="picUrlList", description="商品图片列表")
    image_entity_extend: Any | None = Field(default=None, alias="imageEntityExtend", description="图片扩展信息")

    # 类目信息
    root_category_id: int | None = Field(default=None, alias="rootCategoryId", description="根类目ID")
    root_category_name: str | None = Field(default=None, alias="rootCategoryName", description="根类目名称")
    category_id: int | None = Field(default=None, alias="categoryId", description="类目ID")
    category_name: str | None = Field(default=None, alias="categoryName", description="类目名称")

    # 店铺信息
    shop_id: int | None = Field(default=None, alias="shopId", description="店铺ID")
    shop_name: str | None = Field(default=None, alias="shopName", description="店铺名称")

    # 品牌和属性
    brand: str | None = Field(default=None, description="品牌")
    properties: List[str] | None = Field(default=None, description="属性列表，如 ['袖型:常规', '袖长:长袖']")

    # 价格信息
    goods_price: int | None = Field(default=None, alias="goodsPrice", description="商品价格（分）")
    cprice: int | None = Field(default=None, alias="cprice", description="C端价格（分）")
    period_latest_c_price: int | None = Field(default=None, alias="periodLatestCPrice", description="周期最新C价（分）")
    period_latest_s_price: int | None = Field(default=None, alias="periodLatestSPrice", description="周期最新S价（分）")
    min_price: int | None = Field(default=None, alias="minPrice", description="最低价（分）")
    max_price: int | None = Field(default=None, alias="maxPrice", description="最高价（分）")
    sprice: int | None = Field(default=None, alias="sprice", description="划线价（分）")
    sku_min_price: int | None = Field(default=None, alias="skuMinPrice", description="SKU最低价")
    sku_max_price: int | None = Field(default=None, alias="skuMaxPrice", description="SKU最高价")

    # 总销量/销售额
    sale_volume: int | None = Field(default=None, alias="saleVolume", description="总销量")
    sale_amount: int | None = Field(default=None, alias="saleAmount", description="总销售额（分）")
    total_sale_volume: int | None = Field(default=None, alias="totalSaleVolume", description="累计总销量")
    total_sale_amount: int | None = Field(default=None, alias="totalSaleAmount", description="累计总销售额（分）")

    # 互动数据
    total_comment_num: int | None = Field(default=None, alias="totalCommentNum", description="总评论数")
    total_view_num: int | None = Field(default=None, alias="totalViewNum", description="总浏览数")

    # 周期销量指标
    new_sale_volume: int | None = Field(default=None, alias="newSaleVolume", description="周期总销量")
    new_live_sale_volume: int | None = Field(default=None, alias="newLiveSaleVolume", description="周期直播销量")
    new_video_sale_volume: int | None = Field(default=None, alias="newVideoSaleVolume", description="周期作品销量")
    new_card_sale_volume: int | None = Field(default=None, alias="newCardSaleVolume", description="周期商品卡销量")

    # 周期销售额指标
    new_sale_amount: int | None = Field(default=None, alias="newSaleAmount", description="周期总销售额（分）")
    new_live_sale_amount: int | None = Field(default=None, alias="newLiveSaleAmount", description="周期直播销售额")
    new_video_sale_amount: int | None = Field(default=None, alias="newVideoSaleAmount", description="周期作品销售额")
    new_card_sale_amount: int | None = Field(default=None, alias="newCardSaleAmount", description="周期商品卡销售额")

    # 近期销量统计
    sale_volume_7day: int | None = Field(default=None, alias="saleVolume7day", description="近7天销量")
    sale_volume_30day: int | None = Field(default=None, alias="saleVolume30day", description="近30天销量")
    sale_amount_7day: int | None = Field(default=None, alias="saleAmount7day", description="近7天销售额")
    sale_amount_30day: int | None = Field(default=None, alias="saleAmount30day", description="近30天销售额")

    # 时间相关
    first_record_time: str | None = Field(default=None, alias="firstRecordTime", description="首次记录时间")
    update_time: str | None = Field(default=None, alias="updateTime", description="更新时间")

    # 抖音商品嵌套数据
    douyin_goods: DouyiGoodsNested | None = Field(default=None, alias="douyinGoods", description="抖音商品数据")


class DouyiTopItemsRawResult(BaseModel):
    """抖衣Top商品原始结果"""
    model_config = ConfigDict(populate_by_name=True)

    start: int | None = Field(default=0)
    page_size: int | None = Field(default=10, alias="pageSize")
    result_count: int | None = Field(default=0, alias="resultCount")
    result_list: List[DouyiTopItemRawItem] = Field(default_factory=list, alias="resultList")


class DouyiTopItemsRawResponse(BaseModel):
    """抖衣Top商品原始响应"""
    success: bool = True
    result: DouyiTopItemsRawResult | None = None


class DouyiPropertyValueItem(BaseModel):
    """属性值对象"""
    model_config = ConfigDict(populate_by_name=True)

    property_value_name: str | None = Field(default=None, alias="propertyValueName", description="属性值名称")


class DouyiPropertySelectorRawItem(BaseModel):
    """抖衣属性选择器原始项"""
    model_config = ConfigDict(populate_by_name=True)

    property_name: str | None = Field(default=None, alias="propertyName", description="属性名称")
    property_value_list: List[DouyiPropertyValueItem] | None = Field(
        default=None, alias="propertyValueList", description="属性值对象列表"
    )


class DouyiPropertySelectorRawResponse(BaseModel):
    """抖衣属性选择器原始响应"""
    success: bool = True
    result: List[DouyiPropertySelectorRawItem] = Field(default_factory=list)


__all__ = [
    "ZhiyiSearchRequest",
    "ZhiyiGoodsEntity",
    "DouyiSearchRequest",
    "DouyiGoodsEntity",
    # 深度思考工作流请求模型
    "ZhiyiSaleTrendRequest",
    "ZhiyiPriceRangeTrendRequest",
    "ZhiyiPropertyTrendRequest",
    "ZhiyiCategoryTrendRequest",
    "ZhiyiShopHotItemRequest",
    "ZhiyiPropertyTopRequest",
    # 知衣大盘分析(HYDC)请求模型
    "ZhiyiHydcTrendRequest",
    "ZhiyiHydcPriceRangeRequest",
    "ZhiyiHydcPropertyRequest",
    "ZhiyiHydcBrandRequest",
    "ZhiyiHydcTopItemRequest",
    "ZhiyiHydcCategoryRequest",
    # 深度思考工作流原始响应模型
    "SaleTrendTrendDTO",
    "SaleTrendRawResult",
    "SaleTrendRawResponse",
    "PriceRangeSubItem",
    "PriceRangeRawResponse",
    "PropertyTrendTrendDTO",
    "PropertyTrendRawItem",
    "PropertyTrendRawResponse",
    "ShopHotItemRawItem",
    "ShopHotItemRawResult",
    "ShopHotItemRawResponse",
    "CategoryTrendTrendDTO",
    "CategoryTrendRawItem",
    "CategoryTrendRawResponse",
    "PropertyTopRawItem",
    "PropertyTopRawResponse",
    # 大盘品牌数据模型
    "BrandRawItem",
    "BrandRawResponse",
    "BrandSlimItem",
    "BrandSlimResult",
    "BrandCleanResponse",
    # 深度思考工作流清洗后响应模型
    "SaleTrendSlimItem",
    "SaleTrendSlimResult",
    "SaleTrendCleanResponse",
    "PriceRangeSlimItem",
    "PriceRangeSlimResult",
    "PriceRangeCleanResponse",
    "ColorSlimItem",
    "ColorSlimResult",
    "ColorCleanResponse",
    "Top10ItemSlim",
    "Top10ItemsSlimResult",
    "Top10ItemsCleanResponse",
    "CategoryTrendSlimItem",
    "CategoryTrendSlimResult",
    "CategoryTrendCleanResponse",
    # 抖衣深度思考工作流请求模型
    "DouyiTrendAnalysisRequest",
    "DouyiTopItemsRequest",
    "DouyiPropertySelectorRequest",
    # 抖衣深度思考工作流原始响应模型
    "DouyiTrendSubListItem",
    "DouyiTrendSubItem",
    "DouyiTrendAnalysisRawResult",
    "DouyiTrendAnalysisRawResponse",
    "DouyiTopItemRawItem",
    "DouyiGoodsNested",
    "DouyiTopItemsRawResult",
    "DouyiTopItemsRawResponse",
    "DouyiPropertyValueItem",
    "DouyiPropertySelectorRawItem",
    "DouyiPropertySelectorRawResponse",
]
