# -*- coding: utf-8 -*-
"""
知衣选品 API 封装
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

import requests
from loguru import logger
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field, AliasChoices, field_validator

from app.config import settings
from app.core.errors import AppException, ErrorCode
from app.schemas.entities.workflow.llm_output import ZhiyiParseParam, DouyiParseParam
from app.schemas.response.common import CommonResponse, PageResult


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



class ZhiyiAPI:
    """知衣选品 API"""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self._base_url = base_url.rstrip("/") if base_url else ""
        self._timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        user_id: int,
        team_id: int,
        **kwargs,
    ) -> dict:
        url = f"{self._base_url}{path}"
        headers = kwargs.pop("headers", {})
        headers.update({
            "USER-ID": str(user_id),
            "TEAM-ID": str(team_id),
            "Content-Type": "application/json",
        })

        logger.debug(f"[ZhiyiAPI] {method} {url}")
        if "json" in kwargs:
            logger.debug(f"[ZhiyiAPI] Request Body: {json.dumps(kwargs['json'], ensure_ascii=False)}")

        def _is_retryable(exc: Exception) -> bool:
            if isinstance(exc, requests.exceptions.Timeout):
                return True
            if isinstance(exc, requests.exceptions.ConnectionError):
                return True
            if isinstance(exc, requests.exceptions.HTTPError):
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                return isinstance(status_code, int) and 500 <= status_code < 600
            return False

        @retry(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, max=2),
            reraise=True,
            before_sleep=lambda retry_state: logger.warning(
                f"[ZhiyiAPI] 请求失败重试 {retry_state.attempt_number}/3: {retry_state.outcome.exception()}"
            ),
        )
        def _do_request() -> dict:
            resp = requests.request(
                method=method,
                url=url,
                headers=headers,
                timeout=self._timeout,
                **kwargs,
            )
            resp.raise_for_status()
            return resp.json()

        try:
            return _do_request()
        except requests.exceptions.Timeout:
            logger.error(f"[ZhiyiAPI] 请求超时: {url}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, "知衣选品服务请求超时")
        except Exception as e:
            logger.error(f"[ZhiyiAPI] 请求异常: {e}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, str(e))

    def search_hot_sale(
        self,
        user_id: int,
        team_id: int,
        params: Union[ZhiyiSearchRequest, Dict[str, Any]],
    ) -> PageResult[ZhiyiGoodsEntity]:
        """搜索热销商品"""
        data = self._request(
            method="POST",
            path="/v2-0-x/item/list",
            user_id=user_id,
            team_id=team_id,
            json=self._normalize_request_params(params),
        )
        return self._parse_response(data)

    def search_all_goods(
        self,
        user_id: int,
        team_id: int,
        params: Union[ZhiyiSearchRequest, Dict[str, Any]],
    ) -> PageResult[ZhiyiGoodsEntity]:
        """搜索全网商品（新品）"""
        data = self._request(
            method="POST",
            path="/v2-0-x/item/list",
            user_id=user_id,
            team_id=team_id,
            json=self._normalize_request_params(params),
        )
        return self._parse_response(data)

    def search_shop_goods(
        self,
        user_id: int,
        team_id: int,
        params: Union[ZhiyiSearchRequest, Dict[str, Any]],
    ) -> PageResult[ZhiyiGoodsEntity]:
        """搜索监控店铺商品"""
        data = self._request(
            method="POST",
            path="/v2-0-x/item/shop/all-item-list",
            user_id=user_id,
            team_id=team_id,
            json=self._normalize_request_params(params),
        )
        return self._parse_response(data)

    def search_simple_goods(
        self,
        user_id: int,
        team_id: int,
        params: Union[ZhiyiSearchRequest, Dict[str, Any]],
    ) -> PageResult[ZhiyiGoodsEntity]:
        """搜索新品商品（对齐 n8n: /v2-0-x/item/simple-item-list，用于 all_new/monitor_new）"""
        data = self._request(
            method="POST",
            path="/v2-0-x/item/simple-item-list",
            user_id=user_id,
            team_id=team_id,
            json=self._normalize_request_params(params),
        )
        return self._parse_response(data)

    def search_brand_goods(
        self,
        user_id: int,
        team_id: int,
        params: Union[ZhiyiSearchRequest, Dict[str, Any]],
    ) -> PageResult[ZhiyiGoodsEntity]:
        """搜索品牌商品"""
        data = self._request(
            method="POST",
            path="/v2-0-x/item/shop/all-item-list",
            user_id=user_id,
            team_id=team_id,
            json=self._normalize_request_params(params),
        )
        return self._parse_response(data)

    def _parse_response(self, data: dict) -> PageResult[ZhiyiGoodsEntity]:
        """解析响应"""
        response = CommonResponse[PageResult[ZhiyiGoodsEntity]].model_validate(data)
        if not response.success:
            logger.warning(f"[ZhiyiAPI] 业务错误: {response.error_code} - {response.error_desc}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, response.error_desc or "知衣选品服务返回错误")
        return response.result or PageResult(result_list=[], result_count=0)

    def common_search_douyi(
        self,
        user_id: int,
        team_id: int,
        params: DouyiSearchRequest,
    ) -> PageResult[DouyiGoodsEntity]:
        """搜索抖音商品"""
        data = self._request(
            method="POST",
            path="/live-bus/douyin/window-goods/common-search",
            user_id=user_id,
            team_id=team_id,
            json=params.model_dump(by_alias=True, exclude_none=True),
        )

        response = CommonResponse[PageResult[DouyiGoodsEntity]].model_validate(data)
        if not response.success:
            logger.warning(f"[ZhiyiAPI] 业务错误: {response.error_code} - {response.error_desc}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, response.error_desc or "抖衣选品服务返回错误")

        return response.result or PageResult(result_list=[], result_count=0)

    def get_user_profile(self, user_id: int, team_id: int) -> Dict[str, Any]:
        """获取用户画像"""
        try:
            data = self._request(
                method="GET",
                path=f"/zhiyi-bus/user/profile",
                user_id=user_id,
                team_id=team_id,
            )
            response = CommonResponse[Dict[str, Any]].model_validate(data)
            return response.result or {}
        except Exception as e:
            logger.warning(f"[ZhiyiAPI] 获取用户画像失败: {e}")
            return {}

    @staticmethod
    def _normalize_request_params(params: Union[ZhiyiSearchRequest, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(params, ZhiyiSearchRequest):
            return params.model_dump(by_alias=True, exclude_none=False)
        if isinstance(params, dict):
            return params
        raise TypeError(f"Unsupported params type: {type(params)}")


# 延迟初始化的全局实例
_zhiyi_api_client: Optional[ZhiyiAPI] = None


def get_zhiyi_api_client() -> ZhiyiAPI:
    """获取知衣 API 客户端（延迟初始化）"""
    global _zhiyi_api_client
    if _zhiyi_api_client is None:
        _zhiyi_api_client = ZhiyiAPI(
            base_url=settings.zhiyi_api_url or "",
            timeout=settings.zhiyi_api_timeout,
        )
    return _zhiyi_api_client


__all__ = [
    "ZhiyiAPI",
    "ZhiyiSearchRequest",
    "ZhiyiGoodsEntity",
    "DouyiSearchRequest",
    "DouyiGoodsEntity",
    "get_zhiyi_api_client",
]
