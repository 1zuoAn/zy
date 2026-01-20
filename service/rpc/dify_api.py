# -*- coding: utf-8 -*-
"""
Dify 工作流 API 封装
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import requests
from loguru import logger
from pydantic import BaseModel, Field

from app.config import settings
from app.core.errors import AppException, ErrorCode


class DifyWorkflowResponse(BaseModel):
    """Dify 工作流响应"""

    class Config:
        populate_by_name = True

    workflow_run_id: Optional[str] = Field(default=None, alias="workflow_run_id")
    task_id: Optional[str] = Field(default=None, alias="task_id")
    data: Optional[Dict[str, Any]] = Field(default=None)


class DifyAPI:
    """Dify 工作流 API"""

    def __init__(self, base_url: str, api_key: str, timeout: float = 60.0):
        self._base_url = base_url.rstrip("/") if base_url else ""
        self._api_key = api_key or ""
        self._timeout = timeout

    def run_workflow(
        self,
        workflow_id: str,
        inputs: Dict[str, Any],
        user: str = "test",
    ) -> Dict[str, Any]:
        """
        运行 Dify 工作流

        Args:
            workflow_id: 工作流 App Token（n8n 中 Authorization: Bearer app-xxx）
            inputs: 输入参数
            user: 用户标识

        Returns:
            工作流输出结果
        """
        url = f"{self._base_url}/v1/workflows/run"
        token = (workflow_id or "").strip() or (self._api_key or "").strip()
        if not token:
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, "Dify 未配置可用的 App Token")
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": inputs,
            "response_mode": "blocking",
            "user": user,
        }

        logger.debug(f"[DifyAPI] POST {url}")
        logger.debug(f"[DifyAPI] Inputs: {json.dumps(inputs, ensure_ascii=False)}")

        try:
            resp = requests.post(
                url=url,
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            result = resp.json()

            # 提取输出
            if "data" in result and "outputs" in result["data"]:
                return result["data"]["outputs"]
            return result

        except requests.exceptions.Timeout:
            logger.error(f"[DifyAPI] 请求超时: {url}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, "Dify 工作流请求超时")
        except Exception as e:
            logger.error(f"[DifyAPI] 请求异常: {e}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, str(e))

    def parse_properties(self, properties_text: str) -> List[List[str]]:
        """
        调用 Dify 解析属性词

        Args:
            properties_text: 属性词文本

        Returns:
            属性列表（二维数组）
        """
        if not properties_text or not properties_text.strip():
            return []

        workflow_id = settings.dify_property_workflow_id
        if not workflow_id:
            logger.warning("[DifyAPI] 未配置 dify_property_workflow_id，跳过属性解析")
            return []

        try:
            result = self.run_workflow(
                workflow_id=workflow_id,
                inputs={"properties": properties_text},
            )

            # 解析结果
            if isinstance(result, dict) and "propertyList" in result:
                return result["propertyList"]
            elif isinstance(result, dict) and "property_list" in result:
                return result["property_list"]

            return []

        except Exception as e:
            logger.warning(f"[DifyAPI] 属性解析失败: {e}，返回空列表")
            return []

    def parse_property_tags_for_douyi(self, origin_text: str, property_tags: str) -> List[str]:
        """
        抖衣工作流属性标签解析（对齐 n8n: match_tags）。

        返回格式:
        - ["属性:值", "属性:值", ...]
        """
        if not property_tags or not property_tags.strip():
            return []

        workflow_id = settings.dify_property_workflow_id
        if not workflow_id:
            # 复用知衣的属性解析工作流
            workflow_id = settings.dify_zhiyi_property_workflow_id
        if not workflow_id:
            logger.warning("[DifyAPI] 未配置 dify_property_workflow_id，跳过抖衣属性标签解析")
            return []

        def _flatten_property_list(value: Any) -> List[str]:
            tags: List[str] = []
            if not isinstance(value, list):
                return tags
            if all(isinstance(item, list) for item in value):
                for group in value:
                    for item in group:
                        if isinstance(item, str) and item.strip():
                            tags.append(item.strip())
                return tags
            if all(isinstance(item, str) for item in value):
                return [item.strip() for item in value if item.strip()]
            if all(isinstance(item, dict) for item in value):
                for item in value:
                    name = (item.get("name") or item.get("property") or "").strip()
                    values = item.get("values") or item.get("value") or ""
                    if isinstance(values, str):
                        value_list = [v.strip() for v in values.split(",") if v.strip()]
                    elif isinstance(values, list):
                        value_list = [str(v).strip() for v in values if str(v).strip()]
                    else:
                        value_list = []
                    if not value_list:
                        continue
                    if name:
                        tags.extend([f"{name}:{value}" for value in value_list])
                    else:
                        tags.extend(value_list)
                return tags
            return tags

        try:
            result = self.run_workflow(
                workflow_id=workflow_id,
                inputs={"origin_text": origin_text, "property_tags": property_tags},
            )

            if isinstance(result, dict):
                match_tags = result.get("match_tags") or result.get("matchTags")
                if isinstance(match_tags, list):
                    return [tag.strip() for tag in match_tags if isinstance(tag, str) and tag.strip()]

                for key in ("propertyList", "property_list"):
                    value = result.get(key)
                    if isinstance(value, list):
                        return _flatten_property_list(value)

            return []
        except Exception as e:
            logger.warning(f"[DifyAPI] 抖衣属性标签解析失败: {e}，返回空列表")
            return []

    def parse_properties_for_zhiyi(self, origin_text: str, property_tags: str) -> List[str]:
        """
        知衣工作流属性解析 - 对齐 n8n outputs.match_tags

        n8n 调用参数:
        - inputs.origin_text: 用户原始查询
        - inputs.property_tags: 属性词文本

        返回格式 (对齐 n8n match_tags):
        ["属性:值", "属性:值", ...]
        """
        workflow_id = settings.dify_zhiyi_property_workflow_id
        if not workflow_id:
            logger.warning("[DifyAPI] 未配置 dify_zhiyi_property_workflow_id，跳过知衣属性解析")
            return []

        result = self.run_workflow(
            workflow_id=workflow_id,
            inputs={"origin_text": origin_text, "property_tags": property_tags},
        )

        if not isinstance(result, dict):
            return []
        match_tags = result.get("match_tags") or result.get("matchTags") or []
        if isinstance(match_tags, list):
            return [str(tag).strip() for tag in match_tags if str(tag).strip()]
        return []

    def parse_shop_id_for_zhiyi(self, origin_text: str, brand: str) -> Optional[int]:
        """
        知衣品牌 -> 店铺ID 解析（对齐 n8n 的「知衣店铺检索」工作流）。

        n8n 调用参数:
        - inputs.origin_text: 用户原始查询
        - inputs.brand: 品牌文本

        预期输出:
        - match_tags: ["品牌名,12345", ...] 或类似结构
        """
        brand = (brand or "").strip()
        if not brand:
            return None

        workflow_id = settings.dify_zhiyi_shop_workflow_id
        if not workflow_id:
            logger.warning("[DifyAPI] 未配置 dify_zhiyi_shop_workflow_id，跳过知衣店铺检索")
            return None

        try:
            result = self.run_workflow(
                workflow_id=workflow_id,
                inputs={"origin_text": origin_text, "brand": brand},
            )

            if not isinstance(result, dict):
                return None

            match_tags = result.get("match_tags") or result.get("matchTags")
            if not isinstance(match_tags, list) or not match_tags:
                return None

            first = match_tags[0]
            if isinstance(first, dict):
                first = first.get("品牌") or first.get("brand")
            if not isinstance(first, str):
                return None

            parts = [p.strip() for p in first.split(",") if p.strip()]
            if len(parts) < 2:
                return None

            shop_id_str = parts[1]
            try:
                return int(float(shop_id_str))
            except Exception:
                return None
        except Exception as e:
            logger.warning(f"[DifyAPI] 知衣店铺检索失败: {e}，返回 None")
            return None


# 延迟初始化的全局实例
_dify_api_client: Optional[DifyAPI] = None


def get_dify_api_client() -> DifyAPI:
    """获取 Dify API 客户端（延迟初始化）"""
    global _dify_api_client
    if _dify_api_client is None:
        _dify_api_client = DifyAPI(
            base_url=settings.dify_api_url or "",
            api_key=settings.dify_api_key or "",
            timeout=60.0,
        )
    return _dify_api_client


__all__ = ["DifyAPI", "get_dify_api_client"]
