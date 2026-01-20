from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from app.schemas.response.common import CommonResponse


class ErrorCode(str, Enum):
    OK = "OK"
    BAD_REQUEST = "BAD_REQUEST"
    WORKFLOW_ERROR = "WORKFLOW_ERROR"
    LLM_ERROR = "LLM_ERROR"
    HTTP_ERROR = "HTTP_ERROR"
    DB_ERROR = "DB_ERROR"
    REDIS_ERROR = "REDIS_ERROR"
    CONFIG_ERROR = "CONFIG_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    EXTERNAL_API_ERROR = "EXTERNAL_API_ERROR"


class AppException(Exception):
    def __init__(self, code: ErrorCode, message: str, http_status: int = 500, details: Optional[dict] = None, trace_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status
        self.details = details or {}
        self.trace_id = trace_id


def error_payload(code: ErrorCode, message: str, trace_id: Optional[str] = None, details: Optional[dict] = None) -> dict:
    response = CommonResponse[dict](
        success=False,
        error_code=code.value,
        error_desc=message
    )
    return response.model_dump()

def success_payload(data: Any = None, message: Optional[str] = None, code: ErrorCode = ErrorCode.OK, trace_id: Optional[str] = None, details: Optional[dict] = None) -> dict:
    return {
        "success": True,
        "code": str(code),
        "message": message,
        "trace_id": trace_id,
        "details": details or {},
        "data": data,
    }


__all__ = ["ErrorCode", "AppException", "error_payload", "success_payload"]
