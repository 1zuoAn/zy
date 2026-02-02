# -*- coding: utf-8 -*-
from __future__ import annotations

import json

from fastapi import APIRouter
from langchain_core.messages import AIMessage, ToolMessage

from app.schemas.request.workflow_request import WorkflowRequest
from app.schemas.response.common import CommonResponse
from app.service.chains.workflow.main_orchestrator_graph import MainOrchestratorGraph

router = APIRouter()


@router.post("/main", response_model=CommonResponse[dict])
def run_main_orchestrator(request: WorkflowRequest):
    """主编排工作流入口（返回工具轨迹，不包含思维链）。"""
    state = MainOrchestratorGraph().run(request)
    messages = state.get("messages") or []
    trace: list[dict] = []

    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                trace.append(
                    {
                        "type": "tool_call",
                        "name": call.get("name"),
                        "args": call.get("args"),
                        "id": call.get("id"),
                    }
                )
        elif isinstance(msg, ToolMessage):
            content = msg.content
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except Exception:
                    pass
            trace.append(
                {
                    "type": "tool_result",
                    "name": msg.name,
                    "tool_call_id": msg.tool_call_id,
                    "content": content,
                }
            )

    workflow_response = state.get("workflow_response") or {}
    reply = (
        workflow_response.get("select_result")
        if isinstance(workflow_response, dict)
        else getattr(workflow_response, "select_result", None)
    )

    result = {
        "reply": reply,
        "trace": trace,
        "note": "出于安全规范，不返回思维链，仅提供工具调用与结果轨迹。",
    }
    return CommonResponse[dict](result=result)


__all__ = ["router"]
