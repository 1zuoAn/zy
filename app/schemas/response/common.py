from __future__ import annotations

from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class CommonResponse(BaseModel, Generic[T]):
    """
    通用响应体（最外层）
    适用于所有业务 API 的标准响应格式
    """
    success: bool = True
    result: Optional[T] = None
    error_code: Optional[str] = Field(default=None, alias="errorCode")
    error_desc: Optional[str] = Field(default=None, alias="errorDesc")

    class Config:
        populate_by_name = True


class PageResult(BaseModel, Generic[T]):
    """
    分页列表结果
    包含分页元数据和结果列表
    """
    result_count_limit: Optional[int] = Field(default=0, alias="resultCountLimit")
    start: Optional[int] = 0
    fetch_row_num: Optional[int] = Field(default=0, alias="fetchRowNum")
    page_size: Optional[int] = Field(default=0, alias="pageSize")
    result_count: Optional[int] = Field(default=0, alias="resultCount")
    result_count_max: Optional[int] = Field(default=0, alias="resultCountMax")
    current_page_no: Optional[int] = Field(default=1, alias="currentPageNo")
    param: Optional[Any] = None
    result_list: list[T] = Field(default_factory=list, alias="resultList")
    empty_page: Optional[bool] = Field(default=True, alias="emptyPage")

    class Config:
        populate_by_name = True