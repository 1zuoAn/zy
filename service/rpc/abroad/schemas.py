# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2025/01/15
# @File     : schemas.py
"""
海外探款 API - 数据模型定义

从 abroad_api.py 分离出的 Schema 类
"""
from __future__ import annotations

import re
from typing import Optional, Union

from pydantic import BaseModel, Field, ConfigDict

from app.schemas.entities.workflow.llm_output import AbroadGoodsParseParam, AbroadInsParseParam


# ============================================================
# 海外探款商品 - 请求模型
# ============================================================


class AbroadGoodsSearchRequest(BaseModel):
    """海外探款商品搜索请求参数"""
    model_config = ConfigDict(populate_by_name=True)

    page_no: int = Field(default=1, alias="pageNo")
    page_size: int = Field(default=40, alias="pageSize")

    # === 必需参数（来自 n8n 工作流分析）===
    goods_zone_type: str = Field(default="ALL", alias="goodsZoneType")
    on_sale_flag: int = Field(default=1, alias="onSaleFlag")
    smart_unique: bool = Field(default=False, alias="smartUnique")
    design_element_industry: str = Field(default="CLOTH", alias="designElementIndustry")

    # 排序 - n8n 中是数字类型 sortType (1=最新上架, 8=近30天热销)
    sort_type: Optional[int] = Field(default=None, alias="sortType")

    # 销量范围（近30天）
    min_sale_volume_30day: Optional[int] = Field(default=None, alias="minSaleVolume30Day")
    max_sale_volume_30day: Optional[int] = Field(default=None, alias="maxSaleVolume30Day")

    # 价格范围（单位：分，需要乘以100）
    min_sprice: Optional[int] = Field(default=None, alias="minSprice")
    max_sprice: Optional[int] = Field(default=None, alias="maxSprice")

    # 时间范围 - n8n 中使用 startDate/endDate
    start_date: Optional[str] = Field(default=None, alias="startDate")
    end_date: Optional[str] = Field(default=None, alias="endDate")
    put_on_sale_start_date: Optional[str] = Field(default=None, alias="putOnSaleStartDate")
    put_on_sale_end_date: Optional[str] = Field(default=None, alias="putOnSaleEndDate")

    # 品类 - 二维数组格式，如 [["1", "344", "377"]]
    category_id_list: Optional[list[list[str]]] = Field(default=None, alias="categoryIdList")

    # 地区
    region_id: Optional[str] = Field(default=None, alias="regionId")
    country_list: Optional[list[str]] = Field(default=None, alias="countryList")

    # 平台 - 后端 API 实际需要整数数组，但兼容字符串
    platform_type_list: Optional[list[Union[int, str]]] = Field(default=None, alias="platformTypeList")

    # 属性筛选 - n8n 中都是数组类型
    feature: Optional[list[str]] = Field(default=None, alias="feature")
    style: Optional[list[str]] = Field(default=None, alias="style")
    label: Optional[list[list[str]]] = Field(default=None, alias="label")
    color: Optional[list[str]] = Field(default=None, alias="color")
    brand: Optional[list[str]] = Field(default=None, alias="brand")
    body_type: Optional[list[str]] = Field(default=None, alias="bodyType")

    # 文本搜索
    text: Optional[str] = Field(default=None, alias="text")

    # 结果数量限制
    result_count_limit: Optional[int] = Field(default=6000, alias="resultCountLimit")


    @classmethod
    def from_parse_param(
        cls,
        param: AbroadGoodsParseParam,
        style_list: Optional[list[str]] = None,
        sort_type: Optional[int] = None,
        platform_type_list_override: Optional[list[str]] = None,
        include_put_on_sale: bool = False,
        brand_tags: Optional[list[str]] = None,
    ) -> "AbroadGoodsSearchRequest":
        """
        从 LLM 解析结果转换为请求参数

        Args:
            param: LLM 解析出的参数
            style_list: 解析后的风格列表
            sort_type: 排序类型（数字，1=最新上架, 8=近30天热销）
            brand_tags: Dify 品牌召回的标签列表

        Returns:
            API 请求参数
        """
        # categoryIdList 已经是 list[list[str]] 格式，直接使用
        category_id_list_2d = param.category_id_list if param.category_id_list else None

        # 价格转换：API 需要分为单位，LLM 解析的是美元
        min_sprice_cents = param.min_sprice * 100 if param.min_sprice else None
        max_sprice_cents = param.max_sprice * 100 if param.max_sprice else None

        # countryList 转换为数组
        country_list_arr = None
        if param.country_list:
            if isinstance(param.country_list, str):
                country_list_arr = [c.strip() for c in param.country_list.split(",") if c.strip()]
            elif isinstance(param.country_list, list):
                country_list_arr = param.country_list

        # brand 转换为数组 (优先使用 Dify 召回的 brand_tags)
        brand_arr = None
        if brand_tags:
            brand_arr = brand_tags
        elif param.brand:
            if isinstance(param.brand, str):
                brand_arr = [param.brand]
            elif isinstance(param.brand, list):
                brand_arr = param.brand

        # style 转换为数组（n8n 传入数组，但不做分词拆分）
        style_arr = None
        if style_list is not None:
            style_arr = style_list
        elif param.style:
            if isinstance(param.style, list):
                style_arr = param.style
            else:
                style_arr = [str(param.style).strip()]

        # label 转换为二维数组（设计细节路径）
        label_arr = None
        if param.label:
            label_paths: list[list[str]] = []

            def _add_label_path(parts: list[str]) -> None:
                cleaned = [p.strip() for p in parts if str(p).strip()]
                if not cleaned:
                    return
                label_paths.append(cleaned)

            def _split_label_paths(raw: str) -> None:
                for path_str in re.split(r"[;；|\n]+", raw):
                    path_str = path_str.strip()
                    if not path_str:
                        continue
                    parts = [p.strip() for p in re.split(r"[，,]+", path_str) if p.strip()]
                    _add_label_path(parts)

            if isinstance(param.label, list):
                for item in param.label:
                    if isinstance(item, list):
                        _add_label_path([str(v) for v in item])
                    else:
                        _split_label_paths(str(item))
            else:
                _split_label_paths(str(param.label))

            if label_paths:
                deduped: list[list[str]] = []
                seen = set()
                for path in label_paths:
                    key = tuple(path)
                    if key in seen:
                        continue
                    seen.add(key)
                    deduped.append(path)
                label_arr = deduped

        # feature/color/bodyType 转换为数组（保持原值，不拆分）
        feature_arr = [param.feature] if param.feature else None
        color_arr = [param.color] if param.color else None
        body_type_arr = None
        if param.body_type:
            if isinstance(param.body_type, list):
                body_type_arr = param.body_type
            else:
                body_type_arr = [str(param.body_type).strip()]

        # platformTypeList 兼容来自站点匹配的列表或 LLM 的字符串
        # 注意：后端 API 需要整数数组，需要将字符串转换为整数
        platform_type_list_arr = None
        if platform_type_list_override is not None:
            # 站点匹配返回的是字符串数组，需要转成整数
            platform_type_list_arr = []
            for item in platform_type_list_override:
                if isinstance(item, int):
                    platform_type_list_arr.append(item)
                elif isinstance(item, str) and item.strip().isdigit():
                    platform_type_list_arr.append(int(item.strip()))
                elif isinstance(item, str) and item.strip():
                    # 如果不是纯数字，保留原值（兼容性）
                    platform_type_list_arr.append(item.strip())
        elif param.platform_type_list:
            if isinstance(param.platform_type_list, list):
                platform_type_list_arr = param.platform_type_list
            else:
                raw = str(param.platform_type_list)
                if "," in raw:
                    platform_type_list_arr = [v.strip() for v in raw.split(",") if v.strip()]
                else:
                    platform_type_list_arr = [raw.strip()]

        # 销量范围：0值转为None（0表示不限制）
        min_sale_volume = param.min_sale_volume_total if param.min_sale_volume_total and param.min_sale_volume_total > 0 else None
        max_sale_volume = param.max_sale_volume_total if param.max_sale_volume_total and param.max_sale_volume_total < 999999 else None

        return cls(
            # 必需参数使用默认值，无需显式传递
            minSaleVolume30Day=min_sale_volume,
            maxSaleVolume30Day=max_sale_volume,
            minSprice=min_sprice_cents,
            maxSprice=max_sprice_cents,
            startDate=param.put_on_sale_start_date,
            endDate=param.put_on_sale_end_date,
            putOnSaleStartDate=param.put_on_sale_start_date if include_put_on_sale else None,
            putOnSaleEndDate=param.put_on_sale_end_date if include_put_on_sale else None,
            categoryIdList=category_id_list_2d,
            regionId=param.region_ids,
            countryList=country_list_arr,
            platformTypeList=platform_type_list_arr,
            feature=feature_arr,
            style=style_arr,
            label=label_arr,
            color=color_arr,
            brand=brand_arr,
            bodyType=body_type_arr,
            text=param.text,
            sortType=sort_type,
            resultCountLimit=param.limit,
        )

    def to_simplified(self, keep_brand: bool = False) -> "AbroadGoodsSearchRequest":
        """生成精简版请求（用于兜底查询）

        Args:
            keep_brand: 是否保留brand过滤（品牌查询场景应保留）

        Returns:
            精简后的请求对象
        """
        # 兜底逻辑：移除 style，根据场景决定是否保留 brand
        update_dict = {"style": []}
        if not keep_brand:
            update_dict["brand"] = []
        return self.model_copy(update=update_dict)


# ============================================================
# 海外探款商品 - 响应模型
# ============================================================


class AbroadGoodsMetricInfo(BaseModel):
    """商品指标信息（metricInfo 嵌套对象）"""
    model_config = ConfigDict(populate_by_name=True)

    sale_volume_total: int | None = Field(default=None, alias="saleVolumeTotal")
    sale_amount_total: int | None = Field(default=None, alias="saleAmountTotal")
    sale_volume_30day: int | None = Field(default=None, alias="saleVolume30day")
    sale_volume_15day: int | None = Field(default=None, alias="saleVolume15day")
    sale_volume_7day: int | None = Field(default=None, alias="saleVolume7day")
    sale_volume_yesterday: int | None = Field(default=None, alias="saleVolumeYesterday")
    sale_volume_90day_area: int | None = Field(default=None, alias="saleVolume90dayArea")
    sale_amount_30day: int | None = Field(default=None, alias="saleAmount30day")
    sale_amount_7day: int | None = Field(default=None, alias="saleAmount7day")
    sale_amount_yesterday: int | None = Field(default=None, alias="saleAmountYesterday")
    comment_num: int | None = Field(default=None, alias="commentNum")
    comment_num_7day: int | None = Field(default=None, alias="commentNum7day")
    comment_num_30day: int | None = Field(default=None, alias="commentNum30day")
    min_price: int | None = Field(default=None, alias="minPrice")
    skc_num: int | None = Field(default=None, alias="skcNum")


class AbroadGoodsEntity(BaseModel):
    """海外探款商品实体（字段来自 n8n 工作流和实际 API 响应）"""
    model_config = ConfigDict(populate_by_name=True)

    # API 返回 productId/productName 而非 goodsId/goodsName
    goods_id: Optional[str] = Field(default=None, alias="productId")
    goods_name: Optional[str] = Field(default=None, alias="productName")
    goods_img: Optional[str] = Field(default=None, alias="picUrl")
    product_url: Optional[str] = Field(default=None, alias="productUrl")
    sprice: Optional[float] = Field(default=None, alias="sprice")
    oprice: Optional[float] = Field(default=None, alias="oprice")
    category_id: Optional[str] = Field(default=None, alias="categoryId")
    category_name: Optional[str] = Field(default=None, alias="categoryName")
    platform: Optional[str] = Field(default=None, alias="platform")
    brand: Optional[str] = Field(default=None, alias="brand")
    # 店铺信息
    shop_id: Optional[str] = Field(default=None, alias="shopId")
    shop_name: Optional[str] = Field(default=None, alias="shopName")
    # 类目信息
    category_id_list: list[int] | None = Field(default=None, alias="categoryIdList")
    category_name_list: list[str] | None = Field(default=None, alias="categoryNameList")
    origin_category_name: str | None = Field(default=None, alias="originCategoryName")
    origin_category_id: str | None = Field(default=None, alias="originCategoryId")
    # 平台信息
    platform_type: int | None = Field(default=None, alias="platformType")
    platform_name: str | None = Field(default=None, alias="platformName")
    metric_info: AbroadGoodsMetricInfo | None = Field(default=None, alias="metricInfo")
    # 其他
    properties: list[str] | None = Field(default=None, alias="properties")
    currency: str | None = Field(default=None, alias="currency")
    usd_sprice: float | None = Field(default=None, alias="usdSprice")
    on_sale_date: str | None = Field(default=None, alias="onSaleDate")
    score: int | None = Field(default=None, alias="score")
    comment_num: int | None = Field(default=None, alias="commentNum")


# ============================================================
# 海外探款 - Instagram 博客模型
# ============================================================


class InsBlogListRequest(BaseModel):
    """Instagram 博客列表请求参数"""
    model_config = ConfigDict(populate_by_name=True)

    page_no: Optional[int] = Field(default=1, alias="pageNo")
    page_size: Optional[int] = Field(default=60, alias="pageSize")

    start_date: Optional[str] = Field(default=None, alias="startDate")
    end_date: Optional[str] = Field(default=None, alias="endDate")
    category_list: Optional[list[list[str]]] = Field(default=None, alias="categoryList")
    label: Optional[list[list[str]]] = Field(default=None, alias="label")
    style_list: Optional[list[str]] = Field(default=None, alias="styleList")
    region_list: Optional[list[list[str]]] = Field(default=None, alias="regionList")
    blogger_skin_color_list: Optional[list[str]] = Field(default=None, alias="bloggerSkinColorList")
    blogger_shapes: Optional[list[str]] = Field(default=None, alias="bloggerShapes")
    min_fans_num: Optional[int] = Field(default=None, alias="minFansNum")
    max_fans_num: Optional[int] = Field(default=None, alias="maxFansNum")
    search_content: Optional[str] = Field(default=None, alias="searchContent")
    sort_type: Optional[str] = Field(default=None, alias="sortType")
    result_count_limit: Optional[int] = Field(default=None, alias="resultCountLimit")
    is_monitor_streamer: Optional[int] = Field(default=None, alias="isMonitorStreamer")

    @classmethod
    def from_parse_param(cls, param: AbroadInsParseParam) -> "InsBlogListRequest":
        """
        从 LLM 解析结果转换为请求参数

        Args:
            param: LLM 解析出的参数

        Returns:
            API 请求参数
        """
        data = param.model_dump(by_alias=True, exclude={"limit", "title", "user_data"})
        data["resultCountLimit"] = param.limit
        return cls.model_validate(data)


class InsBloggerInfo(BaseModel):
    """Instagram 博主信息"""
    model_config = ConfigDict(populate_by_name=True)

    blogger_id: Optional[str] = Field(default=None, alias="bloggerId")
    nick_name: Optional[str] = Field(default=None, alias="nickName")
    full_name: Optional[str] = Field(default=None, alias="fullName")
    head_img: Optional[str] = Field(default=None, alias="headImg")
    origin_link: Optional[str] = Field(default=None, alias="originLink")
    email_number: Optional[str] = Field(default=None, alias="emailNumber")
    sum_tags: Optional[str] = Field(default=None, alias="sumTags")
    region: Optional[str] = None
    industry: Optional[str] = None
    style: Optional[str] = None
    identity: Optional[str] = None
    fans_num: Optional[int] = Field(default=None, alias="fansNum")
    like_num: Optional[int] = Field(default=None, alias="likeNum")
    month_fans_num: Optional[int] = Field(default=None, alias="monthFansNum")
    month_like_num: Optional[int] = Field(default=None, alias="monthLikeNum")
    partic_num: Optional[int] = Field(default=None, alias="particNum")
    month_partic_num: Optional[int] = Field(default=None, alias="monthParticNum")
    blog_num: Optional[int] = Field(default=None, alias="blogNum")
    ins_item_num: Optional[int] = Field(default=None, alias="insItemNum")
    introduction: Optional[str] = None
    biography_links: Optional[str] = Field(default=None, alias="biographyLinks")
    is_verified: Optional[int] = Field(default=None, alias="isVerified")
    follow_num: Optional[int] = Field(default=None, alias="followNum")
    included_time: Optional[str] = Field(default=None, alias="includedTime")
    team_monitored: Optional[bool] = Field(default=None, alias="teamMonitored")
    user_monitored: Optional[bool] = Field(default=None, alias="userMonitored")
    involvement_degree: Optional[float] = Field(default=None, alias="involvementDegree")
    avg_like_num: Optional[int] = Field(default=None, alias="avgLikeNum")
    avg_comment_num: Optional[int] = Field(default=None, alias="avgCommentNum")
    blog_num_7day: Optional[int] = Field(default=None, alias="blogNum7day")
    fans_profile: Optional[str] = Field(default=None, alias="fansProfile")
    linktr_url: Optional[str] = Field(default=None, alias="linktrUrl")
    ltk_blogger_id: Optional[str] = Field(default=None, alias="ltkBloggerId")
    ltk_shop_url: Optional[str] = Field(default=None, alias="ltkShopUrl")
    amazon_blogger_id: Optional[str] = Field(default=None, alias="amazonBloggerId")
    tiktok_blogger_id: Optional[str] = Field(default=None, alias="tiktokBloggerId")
    whatsapp_url: Optional[str] = Field(default=None, alias="whatsappUrl")


class EntityObjData(BaseModel):
    """博客内容数据"""
    model_config = ConfigDict(populate_by_name=True)

    comment_num: Optional[int] = Field(default=None, alias="commentNum")
    like_num: Optional[int] = Field(default=None, alias="likeNum")
    image_num: Optional[int] = Field(default=None, alias="imageNum")
    text_content: Optional[str] = Field(default=None, alias="textContent")
    blog_url: Optional[str] = Field(default=None, alias="blogUrl")
    blog_type: Optional[int] = Field(default=None, alias="blogType")
    like_num_increment: Optional[int] = Field(default=None, alias="likeNumIncrement")
    like_num_relative_ratio: Optional[float] = Field(default=None, alias="likeNumRelativeRatio")
    like_fans_ratio: Optional[float] = Field(default=None, alias="likeFansRatio")


class AbroadInsBlogEntity(BaseModel):
    """海外探款 - Instagram 博客实体"""
    model_config = ConfigDict(populate_by_name=True)

    entity_id: Optional[str] = Field(default=None, alias="entityId")
    union_id: Optional[str] = Field(default=None, alias="unionId")
    source_time: Optional[str] = Field(default=None, alias="sourceTime")
    pic_url: Optional[str] = Field(default=None, alias="picUrl")
    pic_list: Optional[list[str]] = Field(default=None, alias="picList")
    images: Optional[list[str]] = None
    height: Optional[int] = None
    width: Optional[int] = None
    sort_values: Optional[list[str]] = Field(default=None, alias="sortValues")
    entity_obj_data: Optional[EntityObjData] = Field(default=None, alias="entityObjData")
    mn_platform_type: Optional[str] = Field(default=None, alias="mnPlatformType")
    mn_entity_id: Optional[str] = Field(default=None, alias="mnEntityId")
    distance: Optional[float] = None


# ============================================================
# 数据洞察分析 API - 请求模型
# ============================================================


class AbroadTrendSummaryRequest(BaseModel):
    """trend-summary 接口请求参数"""
    model_config = ConfigDict(populate_by_name=True)

    platform_type_list: list[int] = Field(alias="platformTypeList")
    category_id_list: Optional[list[list[str]]] = Field(default=None, alias="categoryIdList")
    start_date: Optional[str] = Field(default=None, alias="startDate")
    end_date: Optional[str] = Field(default=None, alias="endDate")
    date_granularity: str = Field(default="MONTH", alias="dateGranularity")
    is_area_data: Optional[bool] = Field(default=None, alias="isAreaData")
    category_type: str = Field(default="normal", alias="categoryType")
    dimension: str = Field(default="PLATFORM", alias="dimension")
    sub_category_flag: bool = Field(default=True, alias="subCategoryFlag")
    exchange_currency: str = Field(default="USD", alias="exchangeCurrency")


class AbroadDimensionInfoRequest(BaseModel):
    """维度分析 info 接口请求参数"""
    model_config = ConfigDict(populate_by_name=True)

    platform_type_list: list[int] = Field(alias="platformTypeList")
    dimension: str = Field(description="KJ_CATEGORY / COLOR / PRICE / FABRIC")
    category_id_list: Optional[list[list[str]]] = Field(default=None, alias="categoryIdList")
    start_date: Optional[str] = Field(default=None, alias="startDate")
    end_date: Optional[str] = Field(default=None, alias="endDate")
    date_granularity: str = Field(default="MONTH", alias="dateGranularity")
    is_area_data: Optional[bool] = Field(default=None, alias="isAreaData")
    category_type: str = Field(default="normal", alias="categoryType")
    sub_category_flag: bool = Field(default=True, alias="subCategoryFlag")
    exchange_currency: str = Field(default="USD", alias="exchangeCurrency")

    # 价格带分析独有参数：
    price_trend_min: int | None = Field(default=None, alias="priceTrendMin", description="价格带下限")
    price_trend_max: int | None = Field(default=None, alias="priceTrendMax", description="价格带上限")
    price_trend_num: int | None = Field(default=None, alias="priceTrendNum", description="价格带数量")


class AbroadPropertyTrendRequest(BaseModel):
    """属性趋势 v2/trend 接口请求参数"""
    model_config = ConfigDict(populate_by_name=True)

    platform_type_list: list[int] = Field(alias="platformTypeList")
    dimension: str = Field(default=None, alias="dimension")
    property: str | None = Field(default=None, alias="property")
    category_id_list: Optional[list[list[str]]] = Field(default=None, alias="categoryIdList")
    start_date: Optional[str] = Field(default=None, alias="startDate")
    end_date: Optional[str] = Field(default=None, alias="endDate")
    date_granularity: str = Field(default="MONTH", alias="dateGranularity")
    is_area_data: Optional[bool] = Field(default=None, alias="isAreaData")
    menu_code: str = Field(default="MARKET_ANALYSIS", alias="menuCode")
    category_type: str = Field(default="normal", alias="categoryType")
    sub_category_flag: bool = Field(default=True, alias="subCategoryFlag")
    exchange_currency: str = Field(default="USD", alias="exchangeCurrency")


class AbroadPropertyListRequest(BaseModel):
    """属性列表 property-list 接口请求参数"""
    model_config = ConfigDict(populate_by_name=True)

    platform_type_list: list[int] = Field(alias="platformTypeList")
    category_id_list: Optional[list[list[str]]] = Field(default=None, alias="categoryIdList")
    menu_code: str = Field(default="MARKET_ANALYSIS", alias="menuCode")
    category_type: str = Field(default="normal", alias="categoryType")
    analysis_range: str = Field(default="NORMAL_SITES", alias="analysisRange")
    sub_category_flag: bool = Field(default=True, alias="subCategoryFlag")

class AbroadAggPriceRangeRequest(BaseModel):
    """价格带分析 /price-range/agg-price-range 接口请求参数"""
    model_config = ConfigDict(populate_by_name=True)

    platform_type_list: list[int] = Field(alias="platformTypeList")
    category_type: str = Field(default="normal", alias="categoryType")
    category_id_list: Optional[list[list[str]]] = Field(default=None, alias="categoryIdList", description="品类路径")
    max_sprice: int | None = Field(default=None, alias="maxSprice", description="价格上限")
    min_sprice: int | None = Field(default=None, alias="minSprice", description="价格下限")


class AbroadAggPriceRangeResponse(BaseModel):
    """价格带分析 /price-range/agg-price-range 接口结果定义"""
    model_config = ConfigDict(populate_by_name=True)

    success: bool = Field(default=None, alias="success")
    result: PriceRange = Field(default=None, alias="result")

    class PriceRange(BaseModel):
        model_config = ConfigDict(populate_by_name=True)
        band_num: int | None = Field(default=None, alias="bandNum", description="价格带数量")
        price_max: int | None = Field(default=None, alias="priceMax", description="价格带上限")
        price_min: int | None = Field(default=None, alias="priceMin", description="价格带下限")



class AbroadTopGoodsAnalysisRequest(BaseModel):
    """Top商品分析请求参数"""
    model_config = ConfigDict(populate_by_name=True)

    platform_type_list: list[int] = Field(alias="platformTypeList")
    category_id_list: Optional[list[list[str]]] = Field(default=None, alias="categoryIdList")
    start_date: Optional[str] = Field(default=None, alias="startDate")
    end_date: Optional[str] = Field(default=None, alias="endDate")
    page_no: int = Field(default=1, alias="pageNo")
    page_size: int = Field(default=10, alias="pageSize")
    sort_type: int = Field(default=8, alias="sortType")
    goods_zone_type: str = Field(default="ALL", alias="goodsZoneType")
    on_sale_flag: int = Field(default=1, alias="onSaleFlag")
