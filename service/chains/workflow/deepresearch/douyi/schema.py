# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2026/1/15 15:05
# @File     : schema.py
from __future__ import annotations
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class DouyiMainParseParam(BaseModel):
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
    date_type: str | None = Field(default=None, description="日期类型,例如 'weekly' / 'monthly'")
    table_type: Literal["1", "2", "3"] = Field(default="1",
        description="图表类型：'1'-品类分析（默认）、'2'-属性分析（长袖/长裙/季节/场景/尺码/面料/品牌/颜色等）、'3'-价格带分析（价格带相关/价格区间等）。注意：若categoryId为null，则不可为'2'"
    )
    frontTitleTabValue: Literal["windowGoods", "liveGoods", "videoGoods", "cardGoods"] = Field(
        default="windowGoods",
        description="在windowGoods、liveGoods、videoGoods、cardGoods中选择"
    )

class DouyiTableType(Enum):
    CATEGORY_ANALYSIS = ("1", "品类分析")
    PROPERTY_ANALYSIS = ("2", "属性分析")
    PRICE_ANALYSIS = ("3", "价格带分析")

    def __init__(self, code: str, desc: str):
        self.code = code
        self.desc = desc

    @classmethod
    def from_code(cls, code: str) -> "DouyiTableType":
        """根据 code 获取对应的枚举值"""
        for item in cls:
            if item.code == code:
                return item
        return cls.CATEGORY_ANALYSIS  # 默认返回品类分析


class DouyiGoodsRelateType(Enum):
    WINDOW_GOODS = ("windowGoods", "全部商品")
    LIVE_GOODS = ("liveGoods", "直播带货")
    VIDEO_GOODS = ("videoGoods", "作品带货")
    CARD_GOODS = ("cardGoods", "商品卡")

    def __init__(self, code: str, desc: str):
        self.code = code
        self.desc = desc

    @classmethod
    def from_code(cls, code: str) -> "DouyiGoodsRelateType":
        """根据 code 获取对应的枚举值"""
        for item in cls:
            if item.code == code:
                return item
        return cls.WINDOW_GOODS



class DouyiCategoryFormatItem(BaseModel):
    root_category_id: str = Field(default=None)
    root_category_id_name: str = Field(default=None)
    category_id: str = Field(default=None)
    category_name: str = Field(default=None)

class DouyiParsedCategory(BaseModel):
    category_list: list[DouyiCategoryFormatItem] = Field(default_factory=list, description="处理后的类目列表")


# ========== 抖衣趋势分析清洗后数据模型 ==========

class DouyiTrendSlimItem(BaseModel):
    """抖衣趋势数据精简项"""
    model_config = ConfigDict(populate_by_name=True)

    granularity_date: str | None = Field(default=None, alias="granularityDate", description="日期/周次")
    daily_sum_day_sales_volume: int = Field(default=0, alias="dailySumDaySalesVolume", description="销量")
    daily_sum_day_sale: int = Field(default=0, alias="dailySumDaySale", description="销售额（分）")
    daily_live_sum: int = Field(default=0, alias="dailyLiveSum", description="直播销量")
    daily_video_sum: int = Field(default=0, alias="dailyVideoSum", description="作品销量")
    daily_card_sum: int = Field(default=0, alias="dailyCardSum", description="商品卡销量")


class DouyiTrendSlimResult(BaseModel):
    """抖衣趋势数据精简结果"""
    model_config = ConfigDict(populate_by_name=True)

    sum_sales_volume: int | None = Field(default=None, alias="sumSalesVolume", description="总销量")
    sum_sale: int | None = Field(default=None, alias="sumSale", description="总销售额")
    top3_category_list: list[str] | None = Field(default=None, alias="top3CategoryList", description="Top3品类")
    has_shuang11_presale: bool = Field(default=False, alias="hasShuang11Presale")
    trend_dtos: list[DouyiTrendSlimItem] = Field(default_factory=list, alias="trendDTOS")


class DouyiTrendCleanResponse(BaseModel):
    """抖衣趋势数据清洗后响应"""
    success: bool = True
    result: DouyiTrendSlimResult | None = None


class DouyiPriceSlimItem(BaseModel):
    """抖衣价格带精简项"""
    model_config = ConfigDict(populate_by_name=True)

    left_price: int = Field(alias="leftPrice", description="价格下限（分）")
    right_price: int | None = Field(default=None, alias="rightPrice", description="价格上限（分），可为空表示无上限")
    sales_volume: int = Field(default=0, alias="salesVolume", description="销量")
    rate: float | None = Field(default=None, alias="rate", description="占比")


class DouyiPriceSlimResult(BaseModel):
    """抖衣价格带精简结果"""
    model_config = ConfigDict(populate_by_name=True)

    price_slim: list[DouyiPriceSlimItem] = Field(default_factory=list)


class DouyiPriceCleanResponse(BaseModel):
    """抖衣价格带清洗后响应"""
    success: bool = True
    result: DouyiPriceSlimResult | None = None


class DouyiPropertySlimItem(BaseModel):
    """抖衣属性分布精简项"""
    model_config = ConfigDict(populate_by_name=True)

    property_name: str | None = Field(default=None, alias="propertyName", description="属性名称，如'面料'")
    property_value: str | None = Field(default=None, alias="propertyValue", description="属性值，如'聚酯纤维'")
    sales_volume: int = Field(default=0, alias="salesVolume", description="销量")
    sales_amount: int | None = Field(default=None, alias="salesAmount", description="销售额")
    rate: float | None = Field(default=None, alias="rate", description="占比")
    other_flag: bool = Field(default=False, alias="otherFlag", description="是否为'其他'合并项")


class DouyiPropertySlimResult(BaseModel):
    """抖衣属性分布精简结果"""
    model_config = ConfigDict(populate_by_name=True)

    property_slim: list[DouyiPropertySlimItem] = Field(default_factory=list)


class DouyiPropertyCleanResponse(BaseModel):
    """抖衣属性分布清洗后响应"""
    success: bool = True
    result: DouyiPropertySlimResult | None = None


class DouyiTopGoodsSlimItem(BaseModel):
    """抖衣Top商品精简项"""
    model_config = ConfigDict(populate_by_name=True)

    rank: int = Field(description="排名")
    item_id: str | None = Field(default=None, alias="itemId", description="商品ID")
    title: str | None = Field(default=None, description="商品标题")
    goods_title: str | None = Field(default=None, alias="goodsTitle", description="商品标题（别名）")
    pic_url: str | None = Field(default=None, alias="picUrl", description="商品图片URL")
    image_entity_extend: str | None = Field(default=None, alias="imageEntityExtend", description="图片扩展信息")
    category_name: str | None = Field(default=None, alias="categoryName", description="品类名称")
    category_detail: str | None = Field(default=None, alias="categoryDetail", description="品类路径")

    # 价格信息
    coupon_cprice: int | None = Field(default=None, alias="couponCprice", description="优惠C价（分）")
    cprice: int | None = Field(default=None, alias="cprice", description="C端价格（分）")
    goods_price: int | None = Field(default=None, alias="goodsPrice", description="商品价格（分）")
    min_price: str | None = Field(default=None, alias="minPrice", description="最低价")
    max_s_price: int | None = Field(default=None, alias="maxSPrice", description="最高价")

    # 店铺信息
    shop_id: int | None = Field(default=None, alias="shopId", description="店铺ID")
    shop_name: str | None = Field(default=None, alias="shopName", description="店铺名称")
    brand: str | None = Field(default=None, description="品牌")

    # 时间信息
    first_record_time: str | None = Field(default=None, alias="firstRecordTime", description="首次记录时间")

    # 销量数据
    total_sale_volume: int | None = Field(default=None, alias="totalSaleVolume", description="累计总销量")
    sale_volume_30day: int | None = Field(default=None, alias="saleVolume30day", description="30天销量")
    new_sale_volume: int | None = Field(default=None, alias="newSaleVolume", description="周期销量")
    sales_volume: int = Field(default=0, alias="salesVolume", description="销量")
    sales_amount: int = Field(default=0, alias="salesAmount", description="销售额（分）")
    new_sale_amount: int | None = Field(default=None, alias="newSaleAmount", description="周期销售额（分）")

    # 互动数据
    comment_count: int | None = Field(default=None, alias="commentCount", description="评论数")
    comment_num_30day: int | None = Field(default=None, alias="commentNum30day", description="30天评论数")
    relate_live_num: int | None = Field(default=None, alias="relateLiveNum", description="关联直播数")
    relate_product_num: int | None = Field(default=None, alias="relateProductNum", description="关联商品数")


class DouyiTopGoodsSlimResult(BaseModel):
    """抖衣Top商品精简结果"""
    model_config = ConfigDict(populate_by_name=True)

    start: int = Field(default=0)
    page_size: int = Field(default=10, alias="pageSize")
    result_count: int = Field(default=0, alias="resultCount")
    top10_slim: list[DouyiTopGoodsSlimItem] = Field(default_factory=list)


class DouyiTopGoodsCleanResponse(BaseModel):
    """抖衣Top商品清洗后响应"""
    success: bool = True
    result: DouyiTopGoodsSlimResult | None = None


# ========== 聚合数据模型（用于报告生成） ==========

class DouyiAggregatedData(BaseModel):
    """抖衣聚合分析数据 - 用于报告生成"""
    model_config = ConfigDict(populate_by_name=True)

    # 分析维度标识
    table_type: str = Field(description="分析类型：1-品类，2-属性，3-价格带")

    # 趋势数据
    sale_volume_data: DouyiTrendCleanResponse | None = Field(default=None, alias="shopSalesVolume", description="销量趋势")
    sale_amount_data: DouyiTrendCleanResponse | None = Field(default=None, alias="shopSalesAmount", description="销售额趋势")

    # 价格带数据
    price_data: DouyiPriceCleanResponse | None = Field(default=None, alias="priceData", description="价格带分布")

    # 属性数据
    property_data: DouyiPropertyCleanResponse | None = Field(default=None, alias="propertyData", description="属性分布")

    # Top商品数据
    top10_products: DouyiTopGoodsCleanResponse | None = Field(default=None, alias="top10Products", description="Top10商品")