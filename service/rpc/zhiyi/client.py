# -*- coding: utf-8 -*-
"""
知衣选品 API 客户端
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Union

import requests
from loguru import logger
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import settings
from app.core.errors import AppException, ErrorCode
from app.schemas.response.common import CommonResponse, PageResult

from .schemas import (
    DouyiGoodsEntity,
    DouyiSearchRequest,
    ZhiyiGoodsEntity,
    ZhiyiSearchRequest,
    ZhiyiSaleTrendRequest,
    ZhiyiPriceRangeTrendRequest,
    ZhiyiPropertyTrendRequest,
    ZhiyiCategoryTrendRequest,
    ZhiyiShopHotItemRequest,
    ZhiyiPropertyTopRequest,
    # 原始响应模型
    SaleTrendRawResponse,
    PriceRangeRawResponse,
    PropertyTrendRawResponse,
    CategoryTrendRawResponse,
    ShopHotItemRawResponse,
    PropertyTopRawResponse,
    # 大盘分析请求模型
    ZhiyiHydcTrendListRequest,
    ZhiyiHydcRankItemRequest,
    # 大盘品牌响应模型
    BrandRawResponse,
    # 大盘趋势响应模型
    HydcTrendListRawResponse,
    # 抖衣深度思考工作流模型
    DouyiTrendAnalysisRequest,
    DouyiTrendAnalysisRawResponse,
    DouyiTopItemsRequest,
    DouyiTopItemsRawResponse,
    DouyiPropertySelectorRequest,
    DouyiPropertySelectorRawResponse,
)


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

    # ========== 深度思考工作流API ==========

    def get_sale_trend(
        self,
        user_id: int,
        team_id: int,
        params: ZhiyiSaleTrendRequest,
    ) -> SaleTrendRawResponse:
        """获取销售趋势数据 - /v1-6-2/trend/sale-trend"""
        data = self._request(
            method="POST",
            path="/v1-6-2/trend/sale-trend",
            user_id=user_id,
            team_id=team_id,
            json=params.model_dump(by_alias=True, exclude_none=True),
        )
        return SaleTrendRawResponse.model_validate(data)

    def get_price_range_trend(
        self,
        user_id: int,
        team_id: int,
        params: ZhiyiPriceRangeTrendRequest,
    ) -> PriceRangeRawResponse:
        """获取价格带趋势数据 - /monitor-shop/v2-3-1/trend/price-range-trend"""
        data = self._request(
            method="POST",
            path="/monitor-shop/v2-3-1/trend/price-range-trend",
            user_id=user_id,
            team_id=team_id,
            json=params.model_dump(by_alias=True, exclude_none=True),
        )
        return PriceRangeRawResponse.model_validate(data)

    def get_property_trend(
        self,
        user_id: int,
        team_id: int,
        params: ZhiyiPropertyTrendRequest,
    ) -> PropertyTrendRawResponse:
        """获取属性趋势数据（颜色/面料等） - /v1-6-0/monitor/trend/property-trend"""
        data = self._request(
            method="POST",
            path="/v1-6-0/monitor/trend/property-trend",
            user_id=user_id,
            team_id=team_id,
            json=params.model_dump(by_alias=True, exclude_none=True),
        )
        return PropertyTrendRawResponse.model_validate(data)

    def get_category_trend(
        self,
        user_id: int,
        team_id: int,
        params: ZhiyiCategoryTrendRequest,
    ) -> CategoryTrendRawResponse:
        """获取品类趋势数据 - /v1-6-0/monitor/trend/category-trend"""
        data = self._request(
            method="POST",
            path="/v1-6-0/monitor/trend/category-trend",
            user_id=user_id,
            team_id=team_id,
            json=params.model_dump(by_alias=True, exclude_none=True),
        )
        return CategoryTrendRawResponse.model_validate(data)

    def get_shop_hot_items(
        self,
        user_id: int,
        team_id: int,
        params: ZhiyiShopHotItemRequest,
    ) -> ShopHotItemRawResponse:
        """获取店铺热销商品Top10 - /v2-0-x/item/shop/hot-item-list"""
        data = self._request(
            method="POST",
            path="/v2-0-x/item/shop/hot-item-list",
            user_id=user_id,
            team_id=team_id,
            json=params.model_dump(by_alias=True, exclude_none=True),
        )
        return ShopHotItemRawResponse.model_validate(data)

    def get_property_top_list(
        self,
        user_id: int,
        team_id: int,
        params: ZhiyiPropertyTopRequest,
    ) -> PropertyTopRawResponse:
        """获取属性列表（用于属性分析时的属性提取） - /v1-6-0/item/taobao-item-property-top"""
        data = self._request(
            method="GET",
            path="/v1-6-0/item/taobao-item-property-top",
            user_id=user_id,
            team_id=team_id,
            params=params.model_dump(by_alias=True, exclude_none=True),
        )
        return PropertyTopRawResponse.model_validate(data)

    def get_category_property_options(
        self,
        user_id: int,
        team_id: int,
        *,
        category_id_list: list[int] | None,
        entrance: int = 5,
    ) -> list[dict[str, Any]]:
        """
        获取“类目下可用的属性筛选项”候选集。

        使用 item/taobao-item-property 接口（支持 categoryId/categoryIdList）。
        """
        if not category_id_list:
            return []

        params: dict[str, Any] = {"entrance": entrance}
        if len(category_id_list) == 1:
            params["categoryId"] = category_id_list[0]
        else:
            params["categoryIdList"] = ",".join(str(cid) for cid in category_id_list)

        data = self._request(
            method="GET",
            path="/item/taobao-item-property",
            user_id=user_id,
            team_id=team_id,
            params=params,
        )
        response = CommonResponse[Any].model_validate(data)
        if not response.success:
            raise AppException(
                ErrorCode.EXTERNAL_API_ERROR,
                response.error_desc or "知衣属性筛选项接口返回错误",
            )
        result = response.result
        if result is None:
            return []
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)]
        if isinstance(result, dict):
            items = result.get("list") or result.get("resultList") or result.get("data")
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
        raise AppException(ErrorCode.EXTERNAL_API_ERROR, "知衣属性筛选项接口返回结构不符合预期")

    # ========== 知衣大盘分析(HYDC) API - 对齐n8n实际使用的端点 ==========

    def get_hydc_trend_list(
        self,
        user_id: int,
        team_id: int,
        params: ZhiyiHydcTrendListRequest,
    ) -> HydcTrendListRawResponse:
        """获取大盘趋势数据（统一接口） - /hydc/trend-list

        通过 groupType 区分数据类型:
        - overview: 概览趋势
        - cprice: 价格带
        - color: 颜色
        - property: 属性
        - brand: 品牌
        """
        data = self._request(
            method="POST",
            path="/hydc/trend-list",
            user_id=user_id,
            team_id=team_id,
            json=params.model_dump(by_alias=True, exclude_none=True),
        )
        return HydcTrendListRawResponse.model_validate(data)

    def get_hydc_rank_items(
        self,
        user_id: int,
        team_id: int,
        params: ZhiyiHydcRankItemRequest,
    ) -> ShopHotItemRawResponse:
        """获取大盘商品排名 - /v2-5-6/rank/item-list-v3"""
        data = self._request(
            method="POST",
            path="/v2-5-6/rank/item-list-v3",
            user_id=user_id,
            team_id=team_id,
            json=params.model_dump(by_alias=True, exclude_none=True),
        )
        return ShopHotItemRawResponse.model_validate(data)

    # ========== 抖衣深度思考工作流API ==========

    def get_douyi_trend_analysis(
        self,
        user_id: int,
        team_id: int,
        params: DouyiTrendAnalysisRequest,
    ) -> DouyiTrendAnalysisRawResponse:
        """抖衣趋势分析 - /v1-2-4/douyin/item-analysis/trend-analysis

        用于查询品类分析、价格分析、属性分析的趋势数据。
        根据 queryType 参数不同返回不同类型的分析数据：
        - categoryAnalysis: 品类分析
        - priceAnalysis: 价格分析
        - propertyAnalysis: 属性分析
        """
        data = self._request(
            method="POST",
            path="/v1-2-4/douyin/item-analysis/trend-analysis",
            user_id=user_id,
            team_id=team_id,
            json=params.model_dump(by_alias=True, exclude_none=True),
        )
        return DouyiTrendAnalysisRawResponse.model_validate(data)

    def get_douyi_top_items(
        self,
        user_id: int,
        team_id: int,
        params: DouyiTopItemsRequest,
    ) -> DouyiTopItemsRawResponse:
        """抖衣Top商品 - /v1-2-4/douyin/item-analysis/item-analysis-top-list

        获取指定条件下的热销商品Top10列表。
        """
        data = self._request(
            method="POST",
            path="/v1-2-4/douyin/item-analysis/item-analysis-top-list",
            user_id=user_id,
            team_id=team_id,
            json=params.model_dump(by_alias=True, exclude_none=True),
        )
        return DouyiTopItemsRawResponse.model_validate(data)

    def get_douyi_property_selector(
        self,
        user_id: int,
        team_id: int,
        params: DouyiPropertySelectorRequest,
    ) -> DouyiPropertySelectorRawResponse:
        """抖衣属性选择 - /douyin-common/item/property-selector-list

        获取指定品类下可用的属性列表，用于属性分析时的属性选择。
        """
        data = self._request(
            method="GET",
            path="/douyin-common/item/property-selector-list",
            user_id=user_id,
            team_id=team_id,
            params=params.model_dump(by_alias=True, exclude_none=True),
        )
        return DouyiPropertySelectorRawResponse.model_validate(data)


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
    "get_zhiyi_api_client",
]
