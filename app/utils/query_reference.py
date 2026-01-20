# -*- coding: utf-8 -*-
"""
占位符引用处理工具

提供 user_query 中占位符的替换和引用数据的提取功能。
占位符格式: @{reference_name}，如 @{shop_1}
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from app.schemas.request.workflow_request import (
        WorkflowQueryReferenceItem,
        WorkflowRequest,
    )


# 占位符正则表达式: 支持 @{reference_name} 和 @reference_name 两种格式
PLACEHOLDER_PATTERN = re.compile(r"@\{?(\w+)\}?")


class QueryReferenceHelper:
    """占位符引用处理器"""

    def __init__(self, query_references: Optional[List[WorkflowQueryReferenceItem]] = None):
        """
        初始化引用处理器

        Args:
            query_references: 引用实体列表
        """
        self._by_name: Dict[str, Dict[str, Any]] = {}
        self._by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        if query_references:
            for ref in query_references:
                ref_dict = {
                    "reference_name": ref.reference_name,
                    "entity_type": ref.entity_type,
                    "entity_id": ref.entity_id,
                    "display_name": ref.display_name,
                    "platform_type": getattr(ref, "platform_type", None),
                    "platform_name": getattr(ref, "platform_name", None),
                }
                self._by_name[ref.reference_name] = ref_dict
                self._by_type[ref.entity_type].append(ref_dict)

    @classmethod
    def from_request(cls, req: WorkflowRequest) -> QueryReferenceHelper:
        """从 WorkflowRequest 创建实例"""
        return cls(req.query_references)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，用于存储到 state"""
        return {
            "by_name": self._by_name,
            "by_type": dict(self._by_type),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> QueryReferenceHelper:
        """从字典恢复实例"""
        helper = cls()
        if data:
            helper._by_name = data.get("by_name", {})
            helper._by_type = defaultdict(list, data.get("by_type", {}))
        return helper

    # ==================== 占位符替换 ====================

    def replace_placeholders(self, text: str) -> str:
        """
        将文本中的占位符替换为展示名称

        Args:
            text: 包含占位符的文本，如 "帮我找@{shop_1}的热销商品"

        Returns:
            替换后的文本，如 "帮我找某某旗舰店的热销商品"
        """
        def replacer(match: re.Match) -> str:
            ref_name = match.group(1)
            ref_data = self._by_name.get(ref_name)
            if ref_data:
                return ref_data["display_name"]
            # 未找到引用时保留原占位符
            return match.group(0)

        return PLACEHOLDER_PATTERN.sub(replacer, text)

    def get_display_query(self, user_query: str) -> str:
        """
        获取用于 LLM 的展示查询（占位符已替换）

        Args:
            user_query: 原始用户查询

        Returns:
            替换占位符后的查询文本
        """
        return self.replace_placeholders(user_query)

    # ==================== 引用数据提取 ====================

    def get_by_name(self, reference_name: str) -> Optional[Dict[str, Any]]:
        """
        按引用名称获取实体数据

        Args:
            reference_name: 引用名称，如 "shop_1"

        Returns:
            实体数据字典，包含 entity_id, entity_type, display_name
        """
        return self._by_name.get(reference_name)

    def get_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """
        按实体类型获取所有实体数据

        Args:
            entity_type: 实体类型，如 "TAOBAO_SHOP"

        Returns:
            该类型的所有实体数据列表
        """
        return self._by_type.get(entity_type, [])

    def get_entity_id(self, reference_name: str) -> Optional[str]:
        """
        获取指定引用的实体ID

        Args:
            reference_name: 引用名称

        Returns:
            实体ID，如 "123456"
        """
        ref_data = self._by_name.get(reference_name)
        return ref_data["entity_id"] if ref_data else None

    def get_entity_ids_by_type(self, entity_type: str) -> List[str]:
        """
        获取指定类型的所有实体ID

        Args:
            entity_type: 实体类型

        Returns:
            实体ID列表
        """
        return [ref["entity_id"] for ref in self._by_type.get(entity_type, [])]

    def get_first_entity_by_type(self, entity_type: str) -> Optional[WorkflowQueryReferenceItem]:
        """
        获取指定类型的第一个实体（常用于单实体场景）
        """
        refs = self._by_type.get(entity_type, [])
        return refs[0] if refs else None

    def get_first_entity_id_by_type(self, entity_type: str) -> Optional[str]:
        """
        获取指定类型的第一个实体ID（常用于单实体场景）

        Args:
            entity_type: 实体类型

        Returns:
            第一个实体ID，如果不存在则返回 None
        """
        refs = self._by_type.get(entity_type, [])
        return refs[0]["entity_id"] if refs else None

    # ==================== 判断方法 ====================

    def has_reference(self, reference_name: str) -> bool:
        """检查是否存在指定引用"""
        return reference_name in self._by_name

    def has_type(self, entity_type: str) -> bool:
        """检查是否存在指定类型的引用"""
        return entity_type in self._by_type and len(self._by_type[entity_type]) > 0

    def is_empty(self) -> bool:
        """检查是否没有任何引用"""
        return len(self._by_name) == 0

    @property
    def all_entity_types(self) -> List[str]:
        """获取所有引用的实体类型"""
        return list(self._by_type.keys())


# ==================== 便捷函数 ====================


def build_query_ref_dict(req: WorkflowRequest) -> Dict[str, Any]:
    """
    从请求构建 query_ref_dict

    Args:
        req: 工作流请求

    Returns:
        可存储到 state 的字典格式
    """
    helper = QueryReferenceHelper.from_request(req)
    return helper.to_dict()


def replace_query_placeholders(user_query: str, query_ref_dict: Optional[Dict[str, Any]]) -> str:
    """
    替换查询中的占位符（便捷函数）

    Args:
        user_query: 原始用户查询
        query_ref_dict: 引用字典（from state）

    Returns:
        替换后的查询文本
    """
    helper = QueryReferenceHelper.from_dict(query_ref_dict)
    return helper.get_display_query(user_query)


def get_shop_id_from_refs(
    query_ref_dict: Optional[Dict[str, Any]], shop_type: Optional[str] = None
) -> Optional[str]:
    """
    从引用中获取店铺ID（便捷函数）

    Args:
        query_ref_dict: 引用字典
        shop_type: 店铺类型，如 "TAOBAO_SHOP"。如果为 None，则尝试所有店铺类型

    Returns:
        店铺ID
    """
    helper = QueryReferenceHelper.from_dict(query_ref_dict)

    if shop_type:
        return helper.get_first_entity_id_by_type(shop_type)

    # 尝试所有店铺类型
    shop_types = ["TAOBAO_SHOP", "DOUYIN_SHOP"]
    for st in shop_types:
        shop_id = helper.get_first_entity_id_by_type(st)
        if shop_id:
            return shop_id
    return None


__all__ = [
    "QueryReferenceHelper",
    "build_query_ref_dict",
    "replace_query_placeholders",
    "get_shop_id_from_refs",
    "PLACEHOLDER_PATTERN",
]
