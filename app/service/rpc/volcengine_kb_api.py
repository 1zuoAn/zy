# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2025/12/7
# @File     : volcengine_kb_api.py
"""
火山引擎知识库 API 封装
基于 /api/knowledge/service/chat 接口实现知识库问答/检索
"""
from __future__ import annotations

import json
from typing import Any, Optional, Union

import requests
from loguru import logger
from pydantic import BaseModel, Field

from app.config import settings
from app.core.errors import AppException, ErrorCode


# ============================================================
# 请求模型
# ============================================================

class KBMessage(BaseModel):
    """知识库对话消息"""
    role: str  # user / assistant / system
    content: Union[str, list[dict]]  # 支持文本和多模态


class KBDocFilter(BaseModel):
    """文档过滤条件"""
    op: str  # must / must_not / range / range_out / and / or
    field: Optional[str] = None
    conds: list[Any] = Field(default_factory=list)


class KBQueryParam(BaseModel):
    """检索过滤参数"""
    doc_filter: Optional[KBDocFilter] = Field(default=None, alias="doc_filter")

    class Config:
        populate_by_name = True


# ============================================================
# 响应模型
# ============================================================

class KBDocInfo(BaseModel):
    """文档信息"""
    doc_id: Optional[str] = Field(default=None, alias="doc_id")
    doc_name: Optional[str] = Field(default=None, alias="doc_name")
    doc_type: Optional[str] = Field(default=None, alias="doc_type")
    create_time: Optional[int] = Field(default=None, alias="create_time")
    doc_meta: Optional[str] = Field(default=None, alias="doc_meta")
    source: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None

    class Config:
        populate_by_name = True


class KBResultItem(BaseModel):
    """检索结果项"""
    id: Optional[str] = None
    content: Optional[str] = None
    md_content: Optional[str] = Field(default=None, alias="md_content")
    html_content: Optional[str] = Field(default=None, alias="html_content")
    score: Optional[float] = None
    rerank_score: Optional[float] = Field(default=None, alias="rerank_score")
    point_id: Optional[str] = Field(default=None, alias="point_id")
    chunk_id: Optional[int] = Field(default=None, alias="chunk_id")
    chunk_title: Optional[str] = Field(default=None, alias="chunk_title")
    chunk_type: Optional[str] = Field(default=None, alias="chunk_type")
    doc_info: Optional[KBDocInfo] = Field(default=None, alias="doc_info")
    recall_position: Optional[int] = Field(default=None, alias="recall_position")
    rerank_position: Optional[int] = Field(default=None, alias="rerank_position")
    original_question: Optional[str] = Field(default=None, alias="original_question")
    process_time: Optional[int] = Field(default=None, alias="process_time")
    table_chunk_fields: Optional[list[dict]] = Field(default=None, alias="table_chunk_fields")

    class Config:
        populate_by_name = True


class KBTokenUsage(BaseModel):
    """Token 使用信息"""
    embedding_token_usage: Optional[dict] = Field(default=None, alias="embedding_token_usage")
    rerank_token_usage: Optional[int] = Field(default=None, alias="rerank_token_usage")
    llm_token_usage: Optional[dict] = Field(default=None, alias="llm_token_usage")

    class Config:
        populate_by_name = True


class KBServiceChatData(BaseModel):
    """service_chat 响应数据"""
    count: Optional[int] = None
    rewrite_query: Optional[str] = Field(default=None, alias="rewrite_query")
    token_usage: Optional[KBTokenUsage] = Field(default=None, alias="token_usage")
    result_list: list[KBResultItem] = Field(default_factory=list, alias="result_list")
    generated_answer: Optional[str] = Field(default=None, alias="generated_answer")
    reasoning_content: Optional[str] = Field(default=None, alias="reasoning_content")

    class Config:
        populate_by_name = True


class KBServiceChatResponse(BaseModel):
    """service_chat 响应"""
    code: int
    message: str
    data: Optional[KBServiceChatData] = None
    request_id: Optional[str] = Field(default=None, alias="request_id")

    class Config:
        populate_by_name = True


# ============================================================
# API 客户端
# ============================================================

class VolcengineKBAPI:
    """
    火山引擎知识库 API
    封装对火山引擎知识库服务的调用
    """

    def __init__(self, base_url: str, timeout: float = 60.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _request(self, path: str, api_key: str, payload: dict) -> dict:
        """
        发送请求

        Args:
            path: 请求路径
            api_key: API Key
            payload: 请求体

        Returns:
            响应 JSON
        """
        url = f"{self._base_url}{path}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        logger.debug(f"[VolcengineKB] POST {url}")
        logger.debug(f"[VolcengineKB] Request Body: {json.dumps(payload, ensure_ascii=False)}")

        try:
            resp = requests.post(
                url=url,
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            logger.error(f"[VolcengineKB] 请求超时: {url}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, "火山引擎知识库服务请求超时")
        except requests.exceptions.HTTPError as e:
            logger.error(f"[VolcengineKB] HTTP 错误: {e}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, f"火山引擎知识库服务 HTTP 错误: {e}")
        except Exception as e:
            logger.error(f"[VolcengineKB] 请求异常: {e}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, str(e))

    def chat(
        self,
        messages: list[KBMessage],
        service_resource_id: Optional[str] = None,
        api_key: Optional[str] = None,
        query_param: Optional[KBQueryParam] = None,
    ) -> KBServiceChatResponse:
        """
        知识库问答/检索

        Args:
            messages: 对话消息列表
            service_resource_id: 知识服务 ID（直接指定，优先级高于 alias）
            api_key: API Key 无则使用配置的默认apikey
            query_param: 检索过滤条件

        Returns:
            KBServiceChatResponse
        """
        # 获取服务配置
        if service_resource_id and api_key:
            # 直接使用传入的配置
            _service_id = service_resource_id
            _api_key = api_key
        elif service_resource_id:
            _service_id = service_resource_id
            _api_key = settings.volcengine_kb_api_key
        else:
            raise AppException(ErrorCode.INTERNAL_ERROR, "知识库RAG查询缺少必须参数")

        # 构建请求体
        payload: dict[str, Any] = {
            "service_resource_id": _service_id,
            "messages": [msg.model_dump() for msg in messages],
            "stream": False,
        }
        if query_param:
            payload["query_param"] = query_param.model_dump(exclude_none=True)

        # 发送请求
        data = self._request(
            path="/api/knowledge/service/chat",
            api_key=_api_key,
            payload=payload,
        )

        # 解析响应
        response = KBServiceChatResponse.model_validate(data)

        if response.code != 0:
            logger.warning(f"[VolcengineKB] 业务错误: {response.code} - {response.message}")
            raise AppException(
                ErrorCode.EXTERNAL_API_ERROR,
                response.message or "火山引擎知识库服务返回错误",
            )

        return response

    # ============================================================
    # 便捷调用方法
    # ============================================================

    def chat_with_history(
        self,
        query: str,
        history: list[tuple[str, str]],
        service_resource_id: str,
        doc_filter: Optional[KBDocFilter] = None,
    ) -> KBServiceChatResponse:
        """
        带历史对话的知识库问答

        Args:
            query: 用户问题
            history: 历史对话 [(user_msg, assistant_msg), ...]
            service_resource_id: 知识服务id
            doc_filter: 文档过滤条件

        Returns:
            问答响应
        """
        messages = []
        for user_msg, assistant_msg in history:
            messages.append(KBMessage(role="user", content=user_msg))
            messages.append(KBMessage(role="assistant", content=assistant_msg))
        messages.append(KBMessage(role="user", content=query))

        query_param = KBQueryParam(doc_filter=doc_filter) if doc_filter else None

        return self.chat(
            messages=messages,
            service_resource_id=service_resource_id,
            query_param=query_param,
        )

    def simple_chat(
        self,
        query: str,
        service_resource_id: str = None,
        doc_filter: Optional[KBDocFilter] = None,
    ) -> KBServiceChatResponse:
        """
        简化的知识库问答（单轮对话）

        Args:
            query: 用户问题
            service_resource_id: 知识服务id
            doc_filter: 文档过滤条件

        Returns:
            问答响应
        """
        messages = [KBMessage(role="user", content=query)]
        query_param = KBQueryParam(doc_filter=doc_filter) if doc_filter else None

        return self.chat(
            messages=messages,
            service_resource_id=service_resource_id,
            query_param=query_param,
        )


# ============================================================
# 延迟初始化单例
# ============================================================

_volcengine_kb_client: Optional[VolcengineKBAPI] = None


def get_volcengine_kb_api() -> VolcengineKBAPI:
    """
    获取火山引擎知识库 API 实例（延迟初始化）

    首次调用时才读取配置并创建实例，避免模块加载时配置未就绪的问题。
    """
    global _volcengine_kb_client
    if _volcengine_kb_client is None:
        base_url = settings.volcengine_kb_api_url
        if not base_url:
            raise RuntimeError("volcengine_kb_api_url 未配置，请检查 Apollo 或环境变量")
        _volcengine_kb_client = VolcengineKBAPI(
            base_url=base_url,
            timeout=settings.volcengine_kb_api_timeout,
        )
    return _volcengine_kb_client


class _LazyVolcengineKBAPIProxy:
    """延迟代理，保持 volcengine_kb_client 的使用方式不变"""

    def __getattr__(self, name):
        return getattr(get_volcengine_kb_api(), name)


volcengine_kb_client = _LazyVolcengineKBAPIProxy()
