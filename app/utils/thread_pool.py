from __future__ import annotations

import contextvars
import os
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Iterable, Optional, Any, List, Tuple

from loguru import logger

from app.config import settings


_executor_lock = threading.Lock()
_executor: Optional[ThreadPoolExecutor] = None


def get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                max_workers = settings.thread_pool_max_workers or (os.cpu_count() or 4) * 2
                _executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="zxy-workflow-thread")
    return _executor


def submit(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
    exec_ = get_executor()
    return exec_.submit(func, *args, **kwargs)


def run_in_pool(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return submit(func, *args, **kwargs).result()


def map_in_pool(func: Callable[..., Any], iterable: Iterable[Any]) -> List[Any]:
    exec_ = get_executor()
    return list(exec_.map(func, iterable))


def run_many(tasks: Iterable[Tuple[Callable[..., Any], Tuple[Any, ...], dict]]) -> List[Any]:
    futures: List[Future] = []
    for fn, args, kwargs in tasks:
        futures.append(submit(fn, *(args or ()), **(kwargs or {})))
    return [f.result() for f in futures]


def shutdown_thread_pool(wait: bool = True) -> None:
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=wait, cancel_futures=True)
        _executor = None


def submit_with_context(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
    """
    携带当前上下文提交任务到线程池。

    使用 contextvars.copy_context() 复制当前线程的上下文（包括 LangChain callback、
    CozeLoop trace 等），确保线程池中的任务能继承主线程的 trace 上下文。

    注意：
    - ctx.run() 会在复制的上下文中执行任务，执行完毕后线程会恢复原始上下文
    - 不会污染线程池中的线程

    Args:
        func: 要执行的函数
        *args: 位置参数
        **kwargs: 关键字参数

    Returns:
        Future 对象，可用于获取结果或等待完成

    Example:
        >>> def my_task(x):
        ...     # 这里可以使用主线程的 trace 上下文
        ...     chain.with_config(callbacks=[handler]).invoke(input)
        >>> future = submit_with_context(my_task, 42)
    """
    ctx = contextvars.copy_context()

    def wrapped() -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"线程池任务异常 [{func.__name__}]: {e}")
            raise

    return submit(ctx.run, wrapped)


__all__ = [
    "get_executor",
    "submit",
    "submit_with_context",
    "run_in_pool",
    "map_in_pool",
    "run_many",
    "shutdown_thread_pool",
]
