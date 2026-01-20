"""
重试装饰器模块
提供可配置的重试机制
"""
from typing import Callable, Optional, Type, Union

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings


def retry_on_exception(
    exceptions: Union[Type[Exception], tuple[Type[Exception], ...]] = Exception,
    max_attempts: Optional[int] = None,
    wait_multiplier: Optional[int] = None,
    wait_max: Optional[int] = None,
) -> Callable:
    """
    异常重试装饰器

    Args:
        exceptions: 需要重试的异常类型
        max_attempts: 最大重试次数，默认使用配置值
        wait_multiplier: 等待时间乘数，默认使用配置值
        wait_max: 最大等待时间（秒），默认使用配置值

    Returns:
        装饰器函数

    Example:
        @retry_on_exception(exceptions=ValueError, max_attempts=3)
        def risky_function():
            ...
    """
    max_attempts = max_attempts or settings.retry_max_attempts
    wait_multiplier = wait_multiplier or settings.retry_wait_exponential_multiplier
    wait_max = wait_max or settings.retry_wait_exponential_max

    return retry(
        retry=retry_if_exception_type(exceptions),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=wait_multiplier, max=wait_max),
        before_sleep=lambda retry_state: logger.warning(
            f"重试 {retry_state.attempt_number}/{max_attempts}: {retry_state.outcome.exception()}"
        ),
    )


# 移除异步重试装饰器


def retry_llm_call(max_attempts: Optional[int] = None) -> Callable:
    from openai import APIError, RateLimitError, Timeout
    exceptions = (APIError, RateLimitError, Timeout, ConnectionError)
    return retry_on_exception(exceptions=exceptions, max_attempts=max_attempts, wait_multiplier=2, wait_max=60)
