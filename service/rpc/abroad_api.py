# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2025/11/28 10:07
# @File     : abroad_api.py
"""
海外探款业务 API 封装
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Union

import requests
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.config import settings
from app.core.clients.db_client import mysql_session_readonly
from app.core.config.constants import DBAlias
from app.core.errors import AppException, ErrorCode
from app.schemas.entities.workflow.llm.abroad_output import AbroadGoodsParseParam
from app.schemas.entities.workflow.llm.abroad_ins_output import AbroadInsParseParam
from app.schemas.response.common import CommonResponse, PageResult


class AbroadGoodsSearchRequest(BaseModel):
    """海外探款商品搜索请求参数"""
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

    class Config:
        populate_by_name = True

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

class AbroadGoodsEntity(BaseModel):
    """海外探款商品实体（字段来自 n8n 工作流和实际 API 响应）"""
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
    # 店铺信息（用于验证店铺筛选是否生效）
    shop_id: Optional[str] = Field(default=None, alias="shopId")
    shop_name: Optional[str] = Field(default=None, alias="shopName")

    class Config:
        populate_by_name = True



# ============================================================
# 海外探款 - Instagram 博客实体模型
# ============================================================

class InsBlogListRequest(BaseModel):
    """Instagram 博客列表请求参数"""
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

    class Config:
        populate_by_name = True

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

    class Config:
        populate_by_name = True


class EntityObjData(BaseModel):
    """博客内容数据"""
    comment_num: Optional[int] = Field(default=None, alias="commentNum")
    like_num: Optional[int] = Field(default=None, alias="likeNum")
    image_num: Optional[int] = Field(default=None, alias="imageNum")
    text_content: Optional[str] = Field(default=None, alias="textContent")
    blog_url: Optional[str] = Field(default=None, alias="blogUrl")
    blog_type: Optional[int] = Field(default=None, alias="blogType")
    # ins_blogger_info: Optional[InsBloggerInfo] = Field(default=None, alias="insBloggerInfo")
    like_num_increment: Optional[int] = Field(default=None, alias="likeNumIncrement")
    like_num_relative_ratio: Optional[float] = Field(default=None, alias="likeNumRelativeRatio")
    like_fans_ratio: Optional[float] = Field(default=None, alias="likeFansRatio")

    class Config:
        populate_by_name = True


class AbroadInsBlogEntity(BaseModel):
    """海外探款 - Instagram 博客实体"""
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

    class Config:
        populate_by_name = True


# ============================================================
# 海外探款 API 封装
# ============================================================

class AbroadAPI:
    """
    海外探款 API
    封装对海外探款后端服务的调用
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 20.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        user_id: str,
        team_id: str,
        **kwargs,
    ) -> dict:
        """
        发送请求

        Args:
            method: HTTP 方法
            path: 请求路径
            user_id: 用户 ID（header）
            team_id: 团队 ID（header）
            **kwargs: 其他 requests 参数

        Returns:
            响应 JSON
        """
        url = f"{self._base_url}{path}"
        headers = kwargs.pop("headers", {})
        headers.update({
            "USER-ID": str(user_id),
            "TEAM-ID": str(team_id),
            "Content-Type": "application/json",
        })

        logger.debug(f"[AbroadAPI] {method} {url}")
        if "json" in kwargs:
            logger.debug(f"[AbroadAPI] Request Body: {json.dumps(kwargs['json'], ensure_ascii=False)}")

        try:
            resp = requests.request(
                method=method,
                url=url,
                headers=headers,
                timeout=self._timeout,
                **kwargs,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            logger.error(f"[AbroadAPI] 请求超时: {url}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, "海外探款服务请求超时")
        except Exception as e:
            logger.error(f"[AbroadAPI] 请求异常: {e}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, str(e))

    def search_ins_blogs(
        self,
        user_id: str,
        team_id: str,
        params: InsBlogListRequest,
    ) -> PageResult[AbroadInsBlogEntity]:
        """
        搜索 Instagram 博客列表

        Args:
            user_id: 用户 ID
            team_id: 团队 ID
            params: 搜索参数（LLM 解析结果）

        Returns:
            分页结果，包含博客实体列表
        """
        data = self._request(
            method="POST",
            path="/external/for-zxy/fashion/ins/blog/list",
            user_id=user_id,
            team_id=team_id,
            json=params.model_dump(by_alias=True, exclude_none=True),
        )

        # 解析响应
        response = CommonResponse[PageResult[AbroadInsBlogEntity]].model_validate(data)

        if not response.success:
            logger.warning(f"[AbroadInsAPI] 业务错误: {response.error_code} - {response.error_desc}")
            raise AppException(
                ErrorCode.EXTERNAL_API_ERROR,
                response.error_desc or "海外探款服务返回错误",
            )

        return response.result or PageResult[AbroadInsBlogEntity]()

    def search_goods(
        self,
        user_id: str,
        team_id: str,
        params: AbroadGoodsSearchRequest,
    ) -> PageResult[AbroadGoodsEntity]:
        """
        搜索商品列表（默认专区）

        Args:
            user_id: 用户ID
            team_id: 团队ID
            params: 搜索参数

        Returns:
            分页结果
        """
        data = self._request(
            method="POST",
            path="/goods-center/goods-zone-list",
            user_id=user_id,
            team_id=team_id,
            json=params.model_dump(by_alias=True, exclude_none=True),
        )

        # 调试：记录API返回的原始数据结构
        logger.debug(f"[AbroadGoodsAPI] 原始响应 success={data.get('success')}, result keys={list(data.get('result', {}).keys()) if data.get('result') else None}")
        if data.get('result'):
            result = data['result']
            logger.debug(f"[AbroadGoodsAPI] resultCount={result.get('resultCount')}, resultList长度={len(result.get('resultList', []))}")

        response = CommonResponse[PageResult[AbroadGoodsEntity]].model_validate(data)
        if not response.success:
            logger.warning(f"[AbroadGoodsAPI] 业务错误: {response.error_code} - {response.error_desc}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, response.error_desc or "海外探款服务返回错误")

        return response.result or PageResult(result_list=[], result_count=0)

    def monitor_site_new(
        self,
        user_id: str,
        team_id: str,
        params: AbroadGoodsSearchRequest | dict,
    ) -> PageResult[AbroadGoodsEntity]:
        """监控站点 - 上新"""
        payload = (
            params.model_dump(by_alias=True, exclude_none=True)
            if isinstance(params, AbroadGoodsSearchRequest)
            else params
        )
        data = self._request(
            method="POST",
            path="/goods-list/monitor-site-new-list",
            user_id=user_id,
            team_id=team_id,
            json=payload,
        )
        response = CommonResponse[PageResult[AbroadGoodsEntity]].model_validate(data)
        if not response.success:
            logger.warning(f"[AbroadGoodsAPI] monitor-site-new 错误: {response.error_desc}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, response.error_desc or "海外探款服务返回错误")
        return response.result or PageResult(result_list=[], result_count=0)

    def monitor_site_hot(
        self,
        user_id: str,
        team_id: str,
        params: AbroadGoodsSearchRequest | dict,
    ) -> PageResult[AbroadGoodsEntity]:
        """监控站点 - 热销"""
        payload = (
            params.model_dump(by_alias=True, exclude_none=True)
            if isinstance(params, AbroadGoodsSearchRequest)
            else params
        )
        data = self._request(
            method="POST",
            path="/goods-list/monitor-site-hot-list",
            user_id=user_id,
            team_id=team_id,
            json=payload,
        )
        response = CommonResponse[PageResult[AbroadGoodsEntity]].model_validate(data)
        if not response.success:
            logger.warning(f"[AbroadGoodsAPI] monitor-site-hot 错误: {response.error_desc}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, response.error_desc or "海外探款服务返回错误")
        return response.result or PageResult(result_list=[], result_count=0)

    # ============================================================
    # Amazon 专区 API
    # ============================================================

    def amazon_monitor_new(
        self,
        user_id: str,
        team_id: str,
        params: dict,
    ) -> PageResult[AbroadGoodsEntity]:
        """Amazon 监控店铺 - 上新"""
        data = self._request(
            method="POST",
            path="/amazon/goods/monitor-shop-new-list",
            user_id=user_id,
            team_id=team_id,
            json=params,
        )
        response = CommonResponse[PageResult[AbroadGoodsEntity]].model_validate(data)
        if not response.success:
            logger.warning(f"[AmazonAPI] monitor-new 错误: {response.error_desc}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, response.error_desc or "Amazon API 错误")
        return response.result or PageResult(result_list=[], result_count=0)

    def amazon_monitor_hot(
        self,
        user_id: str,
        team_id: str,
        params: dict,
    ) -> PageResult[AbroadGoodsEntity]:
        """Amazon 监控店铺 - 热销"""
        data = self._request(
            method="POST",
            path="/amazon/goods/monitor-shop-hot-list",
            user_id=user_id,
            team_id=team_id,
            json=params,
        )
        response = CommonResponse[PageResult[AbroadGoodsEntity]].model_validate(data)
        if not response.success:
            logger.warning(f"[AmazonAPI] monitor-hot 错误: {response.error_desc}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, response.error_desc or "Amazon API 错误")
        return response.result or PageResult(result_list=[], result_count=0)

    def amazon_goods_list(
        self,
        user_id: str,
        team_id: str,
        params: dict,
    ) -> PageResult[AbroadGoodsEntity]:
        """Amazon 商品库"""
        data = self._request(
            method="POST",
            path="/amazon/goods/list",
            user_id=user_id,
            team_id=team_id,
            json=params,
        )
        response = CommonResponse[PageResult[AbroadGoodsEntity]].model_validate(data)
        if not response.success:
            logger.warning(f"[AmazonAPI] goods-list 错误: {response.error_desc}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, response.error_desc or "Amazon API 错误")
        return response.result or PageResult(result_list=[], result_count=0)

    def get_amazon_platforms(self) -> List[Dict[str, Any]]:
        """获取 Amazon 可选站点列表"""
        try:
            with mysql_session_readonly(DBAlias.OLAP_ZXY_AGENT) as session:
                rows = session.execute(
                    text("SELECT biz_type, platform_type, platform_name FROM zxy_abroad_zone_platform WHERE biz_type = 'AMAZON'")
                ).fetchall()
            return [{"biz_type": r[0], "platform_type": r[1], "platform_name": r[2]} for r in rows]
        except Exception as e:
            logger.warning(f"[AmazonAPI] 获取站点列表失败: {e}")
            return []

    def get_amazon_category(self, platform_type: int) -> List[Dict[str, Any]]:
        """获取 Amazon 原站类目"""
        try:
            data = self._request(
                method="POST",
                path="/goods-category/amazon-list",
                user_id="0",
                team_id="0",
                json={"platformType": platform_type, "categoryType": "origin", "menuCode": "AMAZON_ALL_GOODS"},
            )
            response = CommonResponse.model_validate(data)
            return response.result or []
        except Exception as e:
            logger.warning(f"[AmazonAPI] 获取类目失败: {e}")
            return []

    # ============================================================
    # Temu 专区 API
    # ============================================================

    def temu_monitor_new(
        self,
        user_id: str,
        team_id: str,
        params: dict,
    ) -> PageResult[AbroadGoodsEntity]:
        """Temu 监控店铺 - 新品"""
        data = self._request(
            method="POST",
            path="/temu/goods/monitor-shop-new-list",
            user_id=user_id,
            team_id=team_id,
            json=params,
        )
        response = CommonResponse[PageResult[AbroadGoodsEntity]].model_validate(data)
        if not response.success:
            logger.warning(f"[TemuAPI] monitor-new 错误: {response.error_desc}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, response.error_desc or "Temu API 错误")
        return response.result or PageResult(result_list=[], result_count=0)

    def temu_monitor_hot(
        self,
        user_id: str,
        team_id: str,
        params: dict,
    ) -> PageResult[AbroadGoodsEntity]:
        """Temu 监控店铺 - 热销"""
        data = self._request(
            method="POST",
            path="/temu/goods/monitor-shop-hot-list",
            user_id=user_id,
            team_id=team_id,
            json=params,
        )
        response = CommonResponse[PageResult[AbroadGoodsEntity]].model_validate(data)
        if not response.success:
            logger.warning(f"[TemuAPI] monitor-hot 错误: {response.error_desc}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, response.error_desc or "Temu API 错误")
        return response.result or PageResult(result_list=[], result_count=0)

    def temu_goods_list(
        self,
        user_id: str,
        team_id: str,
        params: dict,
    ) -> PageResult[AbroadGoodsEntity]:
        """Temu 商品库"""
        data = self._request(
            method="POST",
            path="/temu/goods/list",
            user_id=user_id,
            team_id=team_id,
            json=params,
        )
        response = CommonResponse[PageResult[AbroadGoodsEntity]].model_validate(data)
        if not response.success:
            logger.warning(f"[TemuAPI] goods-list 错误: {response.error_desc}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, response.error_desc or "Temu API 错误")
        return response.result or PageResult(result_list=[], result_count=0)

    def get_temu_platforms(self) -> List[Dict[str, Any]]:
        """获取 Temu 可选站点列表"""
        try:
            with mysql_session_readonly(DBAlias.OLAP_ZXY_AGENT) as session:
                rows = session.execute(
                    text("SELECT biz_type, platform_type, platform_name FROM zxy_abroad_zone_platform WHERE biz_type = 'TEMU'")
                ).fetchall()
            return [{"biz_type": r[0], "platform_type": r[1], "platform_name": r[2]} for r in rows]
        except Exception as e:
            logger.warning(f"[TemuAPI] 获取站点列表失败: {e}")
            return []

    def get_temu_category(self, platform_type: int) -> List[Dict[str, Any]]:
        """获取 Temu 原站类目"""
        try:
            data = self._request(
                method="POST",
                path="/goods-category/list",
                user_id="0",
                team_id="0",
                json={"platformType": platform_type, "categoryType": "origin", "menuCode": "GOODS_LIST", "hideInvisible": True},
            )
            response = CommonResponse.model_validate(data)
            return response.result or []
        except Exception as e:
            logger.warning(f"[TemuAPI] 获取类目失败: {e}")
            return []

    # ============================================================
    # 独立站专区 API
    # ============================================================

    def site_goods_list(
        self,
        user_id: str,
        team_id: str,
        params: dict,
    ) -> PageResult[AbroadGoodsEntity]:
        """独立站商品列表"""
        data = self._request(
            method="POST",
            path="/goods-list/site-goods-list",
            user_id=user_id,
            team_id=team_id,
            json=params,
        )
        response = CommonResponse[PageResult[AbroadGoodsEntity]].model_validate(data)
        if not response.success:
            logger.warning(f"[AbroadAPI] site-goods-list 错误: {response.error_desc}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, response.error_desc or "独立站 API 错误")
        return response.result or PageResult(result_list=[], result_count=0)

    def external_site_goods_list(
        self,
        user_id: str,
        team_id: str,
        params: dict,
    ) -> PageResult[AbroadGoodsEntity]:
        """独立站商品列表（外部接口）"""
        data = self._request(
            method="POST",
            path="/external/for-zxy/site-goods-list",
            user_id=user_id,
            team_id=team_id,
            json=params,
        )
        response = CommonResponse[PageResult[AbroadGoodsEntity]].model_validate(data)
        if not response.success:
            logger.warning(f"[AbroadAPI] external-site-goods-list 错误: {response.error_desc}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, response.error_desc or "独立站外部 API 错误")
        return response.result or PageResult(result_list=[], result_count=0)

    def get_available_sites(
        self,
        tag_text: str,
    ) -> List[Dict[str, Any]]:
        """获取可用的独立站站点"""
        try:
            data = self._request(
                method="POST",
                path="/external/for-zxy/site-match",
                user_id="0",
                team_id="0",
                json={"tag_text": tag_text},
            )
            response = CommonResponse.model_validate(data)
            if response.success and response.result:
                return response.result.get("outputs", {}).get("match_tags", [])
            return []
        except Exception as e:
            logger.warning(f"[AbroadAPI] 获取可用站点失败: {e}")
            return []

    def get_category_list(self, zone_type: Optional[str] = None) -> dict:
        """
        获取品类列表

        Args:
            zone_type: 专区类型 (amazon/temu/None)

        Returns:
            品类列表数据
        """
        # 根据 zone_type 选择 API 端点
        if zone_type == "amazon":
            path = "/goods-category/amazon-list"
        elif zone_type == "temu":
            path = "/goods-category/list"
        else:
            path = "/goods-category/list"

        try:
            url = f"{self._base_url}{path}"
            resp = requests.get(url, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()

            response = CommonResponse.model_validate(data)
            if not response.success:
                logger.warning(f"[AbroadAPI] Category 接口返回异常: {response.error_code} - {response.error_desc}")

            return response.result or {}
        except Exception as e:
            logger.warning(f"[AbroadAPI] Category 接口请求失败: {e}")
            return {}

    def query_goods_labels(
        self,
        user_id: str,
        team_id: str,
        goods_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        查询商品属性标签 - 对应 n8n 的商品属性接口

        Args:
            user_id: 用户ID
            team_id: 团队ID
            goods_list: 商品列表，格式 [{"productId": "xxx", ...}]

        Returns:
            商品属性标签列表
        """
        if not goods_list:
            return []

        try:
            data = self._request(
                method="POST",
                path="/external/for-zxy/query-goods-labels",
                user_id=user_id,
                team_id=team_id,
                json=goods_list,
            )
            response = CommonResponse.model_validate(data)
            if response.success and response.result:
                return response.result if isinstance(response.result, list) else []
            return []
        except Exception as e:
            logger.warning(f"[AbroadAPI] 查询商品属性失败: {e}")
            return []

    def recall_brand_tags(self, brand: str) -> List[str]:
        """
        调用 Dify 工作流进行品牌召回 - 对应 n8n 的 "HTTP Request" (Dify) 节点

        Args:
            brand: 品牌名称

        Returns:
            匹配的品牌标签列表
        """
        if not brand:
            return []

        try:
            resp = requests.post(
                "https://dify-internal.zhiyitech.cn/v1/workflows/run",
                headers={
                    "Authorization": "Bearer app-pl75Uq9zYLr0KFAwYurxjosf",
                    "Content-Type": "application/json",
                },
                json={
                    "inputs": {"tag_text": brand},
                    "response_mode": "blocking",
                    "user": "zxy-workflow",
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            # Dify 返回格式: {"data": {"outputs": {"match_tags": [...]}}}
            match_tags = data.get("data", {}).get("outputs", {}).get("match_tags", [])
            return match_tags if isinstance(match_tags, list) else []
        except Exception as e:
            logger.warning(f"[AbroadAPI] Dify 品牌召回失败: {e}")
            return []

    def get_common_selections(self) -> dict:
        """
        获取通用筛选项数据 - 从数据库查询

        Returns:
            包含 category_paths 和 common_selections 的字典
        """
        result: Dict[str, Any] = {
            "category_paths": [],
            "common_selections": {},
        }

        try:
            # 1. 查询品类数据
            with mysql_session_readonly(DBAlias.OLAP_ZXY_AGENT) as session:
                category_rows = session.execute(
                    text("SELECT category_id_path, category_name_path FROM zxy_abroad_selection_category")
                ).fetchall()

            category_paths = []
            for row in category_rows:
                category_paths.append({
                    "id_path": row[0].strip() if row[0] else "",
                    "name_path": row[1].strip() if row[1] else "",
                })
            result["category_paths"] = category_paths
            logger.debug(f"[AbroadAPI] 查询到 {len(category_paths)} 条品类数据")

            # 2. 查询通用筛选项
            with mysql_session_readonly(DBAlias.OLAP_ZXY_AGENT) as session:
                selection_rows = session.execute(
                    text("SELECT type, label, value FROM zxy_abroad_selection_common")
                ).fetchall()

            common_selections: Dict[str, List[Dict[str, str]]] = {}
            for row in selection_rows:
                sel_type = row[0].strip() if row[0] else ""
                if sel_type not in common_selections:
                    common_selections[sel_type] = []
                common_selections[sel_type].append({
                    "label": row[1].strip() if row[1] else "",
                    "value": row[2].strip() if row[2] else "",
                })
            result["common_selections"] = common_selections
            logger.debug(f"[AbroadAPI] 查询到 {len(selection_rows)} 条通用筛选项")

            return result

        except Exception as e:
            logger.warning(f"[AbroadAPI] 查询筛选项数据失败: {e}")
            return result



# ============================================================
# 延迟初始化单例
# ============================================================

_abroad_api_client: Optional[AbroadAPI] = None


def get_abroad_api() -> AbroadAPI:
    """
    获取海外探款 API 实例（延迟初始化）

    首次调用时才读取配置并创建实例，避免模块加载时配置未就绪的问题。
    """
    global _abroad_api_client
    if _abroad_api_client is None:
        base_url = settings.abroad_api_url
        if not base_url:
            raise RuntimeError("abroad_ins_api_url 未配置，请检查 Apollo 或环境变量")
        _abroad_api_client = AbroadAPI(
            base_url=base_url,
            timeout=settings.abroad_api_timeout,
        )
    return _abroad_api_client


# 向后兼容的别名（推荐使用 get_abroad_api()）
class _LazyAbroadAPIProxy:
    """延迟代理，保持 abroad_api_client 的使用方式不变"""

    def __getattr__(self, name):
        return getattr(get_abroad_api(), name)


abroad_api_client = _LazyAbroadAPIProxy()
