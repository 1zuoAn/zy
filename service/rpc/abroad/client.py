# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2025/01/15
# @File     : client.py
"""
海外探款 API - 客户端实现

从 abroad_api.py 分离出的 Client 类和单例管理
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import requests
from loguru import logger
from sqlalchemy import text

from app.config import settings
from app.core.clients.db_client import mysql_session_readonly
from app.core.config.constants import DBAlias
from app.core.errors import AppException, ErrorCode
from app.schemas.response.common import CommonResponse, PageResult

from .schemas import (
    AbroadGoodsEntity,
    AbroadGoodsSearchRequest,
    AbroadInsBlogEntity,
    InsBlogListRequest,
    AbroadTrendSummaryRequest,
    AbroadDimensionInfoRequest,
    AbroadPropertyTrendRequest,
    AbroadPropertyListRequest,
    AbroadTopGoodsAnalysisRequest, AbroadAggPriceRangeRequest, AbroadAggPriceRangeResponse,
)

# 导入数据洞察相关的原始响应模型
from app.service.chains.workflow.deepresearch.abroad.schema import (
    AbroadTrendSummaryRawResponse,
    AbroadDimensionInfoRawResponse,
    AbroadPropertyRawResponse,
    AbroadTopGoodsRawResponse,
    AbroadPropertyListRawResponse,
)


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
    # 数据洞察 API（deepresearch 工作流使用）
    # ============================================================

    def get_trend_summary(
        self,
        user_id: str,
        team_id: str,
        request: AbroadTrendSummaryRequest,
    ) -> AbroadTrendSummaryRawResponse:
        """
        获取概览趋势数据
        接口: /overview/dimension-analyze/trend-summary

        Args:
            user_id: 用户ID
            team_id: 团队ID
            request: 趋势数据请求参数

        Returns:
            趋势数据原始响应
        """
        try:
            data = self._request(
                method="POST",
                path="/overview/dimension-analyze/trend-summary",
                user_id=user_id,
                team_id=team_id,
                json=request.model_dump(by_alias=True, exclude_none=True),
            )
            return AbroadTrendSummaryRawResponse.model_validate(data)
        except Exception as e:
            logger.warning(f"[AbroadAPI] get_trend_summary 失败: {e}")
            return AbroadTrendSummaryRawResponse(success=False, error_desc=str(e))

    def get_dimension_info(
        self,
        user_id: str,
        team_id: str,
        request: AbroadDimensionInfoRequest,
    ) -> AbroadDimensionInfoRawResponse:
        """
        获取维度分析数据（品类/颜色/价格带/面料通用）
        接口: /overview/dimension-analyze/info

        Args:
            user_id: 用户ID
            team_id: 团队ID
            request: 维度分析请求参数（包含 dimension 字段决定查询类型）

        Returns:
            维度分析数据原始响应
        """
        try:
            data = self._request(
                method="POST",
                path="/overview/dimension-analyze/info",
                user_id=user_id,
                team_id=team_id,
                json=request.model_dump(by_alias=True, exclude_none=True),
            )
            return AbroadDimensionInfoRawResponse.model_validate(data)
        except Exception as e:
            logger.warning(f"[AbroadAPI] get_dimension_info 失败: {e}")
            return AbroadDimensionInfoRawResponse(success=False, error_desc=str(e))

    def get_property_trend(
        self,
        user_id: str,
        team_id: str,
        request: AbroadPropertyTrendRequest,
    ) -> AbroadPropertyRawResponse:
        """
        获取属性趋势数据
        接口: /overview/dimension-analyze/v2/trend

        Args:
            user_id: 用户ID
            team_id: 团队ID
            request: 属性趋势请求参数（包含 property_name 字段）

        Returns:
            属性趋势数据原始响应
        """
        try:
            data = self._request(
                method="POST",
                path="/overview/dimension-analyze/v2/trend",
                user_id=user_id,
                team_id=team_id,
                json=request.model_dump(by_alias=True, exclude_none=True),
            )
            return AbroadPropertyRawResponse.model_validate(data)
        except Exception as e:
            logger.warning(f"[AbroadAPI] get_property_trend 失败: {e}")
            return AbroadPropertyRawResponse(success=False, error_desc=str(e))

    def get_property_list(
        self,
        user_id: str,
        team_id: str,
        request: AbroadPropertyListRequest,
    ) -> AbroadPropertyListRawResponse:
        """
        获取可用属性列表
        接口: /overview/selection/property-list

        Args:
            user_id: 用户ID
            team_id: 团队ID
            request: 属性列表请求参数

        Returns:
            属性列表原始响应
        """
        try:
            data = self._request(
                method="POST",
                path="/overview/selection/property-list",
                user_id=user_id,
                team_id=team_id,
                json=request.model_dump(by_alias=True, exclude_none=True),
            )
            return AbroadPropertyListRawResponse.model_validate(data)
        except Exception as e:
            logger.warning(f"[AbroadAPI] get_property_list 失败: {e}")
            return AbroadPropertyListRawResponse(success=False, error_desc=str(e))

    def get_agg_price_range(
        self,
        user_id: str,
        team_id: str,
        request: AbroadAggPriceRangeRequest,
    ) -> AbroadAggPriceRangeResponse:
        try:
            data = self._request(
                method="POST",
                path="/price-range/agg-price-range",
                user_id=user_id,
                team_id=team_id,
                json=request.model_dump(by_alias=True, exclude_none=True),
            )
            return AbroadAggPriceRangeResponse.model_validate(data)
        except Exception as e:
            logger.warning(f"[AbroadAPI] get_agg_price_range失败: {e}")
            return AbroadAggPriceRangeResponse(success=False)

    def get_top_goods_for_analysis(
        self,
        user_id: str,
        team_id: str,
        request: AbroadTopGoodsAnalysisRequest,
    ) -> AbroadTopGoodsRawResponse:
        """
        获取Top商品数据（用于数据洞察分析）
        接口: /goods-center/goods-zone-list

        Args:
            user_id: 用户ID
            team_id: 团队ID
            request: Top商品请求参数

        Returns:
            Top商品数据原始响应
        """
        try:
            data = self._request(
                method="POST",
                path="/goods-center/goods-zone-list",
                user_id=user_id,
                team_id=team_id,
                json=request.model_dump(by_alias=True, exclude_none=True),
            )
            return AbroadTopGoodsRawResponse.model_validate(data)
        except Exception as e:
            logger.warning(f"[AbroadAPI] get_top_goods_for_analysis 失败: {e}")
            return AbroadTopGoodsRawResponse(success=False, error_desc=str(e))


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



__all__ = [
    "AbroadAPI",
    "get_abroad_api",
]
