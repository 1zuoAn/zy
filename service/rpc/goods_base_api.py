# -*- coding: utf-8 -*-
"""
选品业务通用 API 基类
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests
from loguru import logger
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.core.errors import AppException, ErrorCode


class BaseGoodsAPI:
    """选品业务通用 API 基类"""

    def __init__(self, base_url: str, timeout: float = 30.0, logger_prefix: str = "BaseGoodsAPI"):
        self._base_url = base_url.rstrip("/") if base_url else ""
        self._timeout = timeout
        self._logger_prefix = logger_prefix

    def _is_retryable(self, exc: Exception) -> bool:
        """判断异常是否可重试"""
        if isinstance(exc, requests.exceptions.Timeout):
            return True
        if isinstance(exc, requests.exceptions.ConnectionError):
            return True
        if isinstance(exc, requests.exceptions.HTTPError):
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            return isinstance(status_code, int) and 500 <= status_code < 600
        return False

    def _request(
        self,
        method: str,
        path: str,
        user_id: int,
        team_id: int,
        **kwargs,
    ) -> dict:
        """
        发送 HTTP 请求 (带重试机制)
        """
        url = f"{self._base_url}{path}"
        headers = kwargs.pop("headers", {})
        headers.update({
            "USER-ID": str(user_id),
            "TEAM-ID": str(team_id),
            "Content-Type": "application/json",
        })

        logger.debug(f"[{self._logger_prefix}] {method} {url}")
        if "json" in kwargs:
            logger.debug(f"[{self._logger_prefix}] Request Body: {json.dumps(kwargs['json'], ensure_ascii=False)}")

        @retry(
            retry=retry_if_exception(self._is_retryable),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, max=2),
            reraise=True,
            before_sleep=lambda retry_state: logger.warning(
                f"[{self._logger_prefix}] 请求失败重试 {retry_state.attempt_number}/3: {retry_state.outcome.exception()}"
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
            logger.error(f"[{self._logger_prefix}] 请求超时: {url}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, f"{self._logger_prefix}服务请求超时")
        except Exception as e:
            logger.error(f"[{self._logger_prefix}] 请求异常: {e}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, str(e))
