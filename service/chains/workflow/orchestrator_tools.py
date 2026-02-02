"""
子工作流工具定义（ReAct 模式）

架构说明：
- 使用 @tool 装饰器 + args_schema 定义工具（符合 LangChain 标准）
- 使用 InjectedState 从 Graph State 注入上下文（request 等）
- LLM 通过 bind_tools(ALL_TOOLS) 获得工具列表并自主决策
- 无硬编码决策规则，完全由 LLM 推理
- 返回 Artifact 模式：实际数据存储在 Redis，只返回 artifact_id
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional
from uuid import uuid4

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated

from app.core.config.constants import WorkflowType
from app.schemas.entities.workflow.graph_state import MainOrchestratorState
from app.schemas.request.schedule_request import ScheduleTaskRequest
from app.schemas.request.workflow_request import WorkflowQueryReferenceItem, WorkflowRequest
from app.service.chains.workflow.artifact_description import DescriptionContext, resolve_description
from app.service.chains.workflow.artifact_store import get_artifact_store
from app.service.chains.workflow.workflow_delegate import get_delegate


# ============================================================
# 工具参数 Schema 定义
# ============================================================


class InjectedRequestArgs(BaseModel):
    """工具通用上下文字段（系统自动注入，LLM 无需传递）"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")
    request: Annotated[WorkflowRequest, InjectedState("request")]  # 从 Graph State 自动注入


class SelectionArgs(InjectedRequestArgs):
    """选品类工具入参（select_zhiyi, select_douyi, select_abroad_goods）"""

    user_query: Optional[str] = Field(default=None, description="选品关键词")
    preferred_entity: Optional[str] = Field(default=None, description="平台偏好")
    industry: Optional[str] = Field(default=None, description="行业")
    user_preferences: Optional[str] = Field(default=None, description="用户偏好")
    is_monitored: Optional[bool] = Field(default=None, description="是否参考监控数据")
    is_user_preferences: Optional[bool] = Field(default=None, description="是否参考用户画像")
    query_references: Optional[list[WorkflowQueryReferenceItem]] = Field(
        default=None,
        description="可选：关联实体列表（系统透传）",
    )


class AbroadSelectionArgs(SelectionArgs):
    """海外选品工具入参。"""

    abroad_type: Optional[str] = Field(default=None, description="海外站点类型")


class MediaArgs(InjectedRequestArgs):
    """媒体类工具入参。"""

    user_query: Optional[str] = Field(default=None, description="检索关键词")
    preferred_entity: Optional[str] = Field(default=None, description="平台偏好")
    industry: Optional[str] = Field(default=None, description="行业")
    user_preferences: Optional[str] = Field(default=None, description="用户偏好")
    is_monitored: Optional[bool] = Field(default=None, description="是否参考监控数据")
    is_user_preferences: Optional[bool] = Field(default=None, description="是否参考用户画像")


class MediaAbroadArgs(MediaArgs):
    """海外媒体类工具入参。"""

    abroad_type: Optional[str] = Field(default=None, description="海外站点类型")


class ShopRankArgs(InjectedRequestArgs):
    """店铺排行工具入参。"""

    user_query: Optional[str] = Field(default=None, description="店铺排行关键词")
    preferred_entity: Optional[str] = Field(default=None, description="平台偏好")
    industry: Optional[str] = Field(default=None, description="行业")
    user_preferences: Optional[str] = Field(default=None, description="用户偏好")


class ImageSearchArgs(InjectedRequestArgs):
    """图搜工具入参。"""

    image_url: Optional[str] = Field(default=None, description="图搜图片链接")
    images: Optional[list[str]] = Field(default=None, description="用户上传图片URL数组")
    input_images: Optional[list[str] | str] = Field(
        default=None, description="备用输入图片(URL列表或#分隔字符串)"
    )
    user_query: Optional[str] = Field(default=None, description="补充说明关键词")
    preferred_entity: Optional[str] = Field(default=None, description="平台偏好")
    abroad_type: Optional[str] = Field(default=None, description="海外站点类型")


class ImageCreateArgs(InjectedRequestArgs):
    """文生图工具入参。"""

    image_prompt: Optional[str] = Field(default=None, description="画面描述")
    user_query: Optional[str] = Field(default=None, description="补充说明关键词")


class ImageEditArgs(InjectedRequestArgs):
    """图生图工具入参。"""

    input_images: Optional[list[str] | str] = Field(
        default=None,
        description="输入图片URL列表或字符串",
    )
    edit_prompt: Optional[str] = Field(default=None, description="编辑指令")
    user_query: Optional[str] = Field(default=None, description="补充说明关键词")


class ScheduleTaskArgs(InjectedRequestArgs):
    """定时任务工具入参。"""

    task_title: Optional[str] = Field(default=None, description="定时任务标题")
    task_content: Optional[str] = Field(default=None, description="定时任务内容")
    cron_expression: Optional[str] = Field(default=None, description="定时任务 cron 表达式")


class InspectArtifactArgs(BaseModel):
    """inspect_artifact 入参。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")
    request: Annotated[WorkflowRequest, InjectedState("request")]
    artifact_id: str = Field(..., description="需要查看的资产ID")
    limit: int = Field(default=20, ge=1, le=50, description="最大返回行数，默认20，最大50")
    filter_keyword: Optional[str] = Field(default=None, description="可选：内容过滤关键词")
    filter_field: Optional[str] = Field(default=None, description="指定过滤字段，如 'title', 'price'")
    filter_keywords: Optional[list[str]] = Field(
        default=None,
        description="多关键词过滤（OR 逻辑）",
    )


# ============================================================
# 工具工厂函数
# ============================================================


def _safe_parse_json(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return raw
    return raw


def _resolve_artifact_type(tool_name: str) -> str:
    return {
        "image_create": "image_asset",
        "image_edit": "image_asset",
        "schedule_task": "task_receipt",
    }.get(tool_name, "product_list")


def _merge_request_context(tool_args: dict[str, Any], request: WorkflowRequest) -> dict[str, Any]:
    merged = dict(tool_args)
    if not merged.get("user_query"):
        merged["user_query"] = getattr(request, "user_query", None)

    context_fields = [
        "team_id",
        "user_id",
        "session_id",
        "message_id",
        "preferred_entity",
        "industry",
        "user_preferences",
        "is_monitored",
        "is_user_preferences",
        "query_references",
        "abroad_type",
        "images",
        "image_url",
        "input_images",
        "image_prompt",
        "edit_prompt",
    ]
    for field in context_fields:
        if merged.get(field) is None:
            value = getattr(request, field, None)
            if value is not None:
                merged[field] = value
    return merged


def _resolve_user_query(workflow_type: WorkflowType, tool_args: dict[str, Any]) -> str:
    if workflow_type == WorkflowType.IMAGE_CREATE:
        image_prompt = tool_args.get("image_prompt")
        if image_prompt not in (None, ""):
            return str(image_prompt)
    if workflow_type == WorkflowType.IMAGE_EDIT:
        edit_prompt = tool_args.get("edit_prompt")
        if edit_prompt not in (None, ""):
            return str(edit_prompt)
    if workflow_type == WorkflowType.IMAGE_SEARCH:
        return "图搜同款"

    user_query = tool_args.get("user_query")
    if user_query not in (None, ""):
        return str(user_query)

    return ""


def _build_content_cache(
    tool_name: str,
    tool_args: dict[str, Any],
    request_obj: Any,
) -> str | None:
    if tool_name == "image_create":
        return tool_args.get("image_prompt") or getattr(request_obj, "user_query", None)
    if tool_name == "image_edit":
        return tool_args.get("edit_prompt") or getattr(request_obj, "user_query", None)
    return None


def _build_tool_request(
    workflow_type: WorkflowType,
    tool_args: dict[str, Any],
) -> Any:
    if workflow_type == WorkflowType.SCHEDULE:
        payload = {
            "team_id": tool_args.get("team_id"),
            "user_id": tool_args.get("user_id"),
            "session_id": tool_args.get("session_id"),
            "message_id": tool_args.get("message_id"),
            "task_title": tool_args.get("task_title") or tool_args.get("taskTitle"),
            "task_content": tool_args.get("task_content") or tool_args.get("taskContent"),
            "task_default_cron": tool_args.get("cron_expression")
            or tool_args.get("task_default_cron")
            or tool_args.get("taskDefaultCron"),
        }
        return ScheduleTaskRequest.model_validate(payload)

    payload = dict(tool_args)
    payload["user_query"] = _resolve_user_query(workflow_type, tool_args)
    if not payload["user_query"]:
        raise ValueError("缺少 user_query")
    return WorkflowRequest.model_validate(payload)


def _create_workflow_tool(
    name: str,
    description: str,
    args_schema: type[BaseModel],
    workflow_type: WorkflowType,
) -> BaseTool:
    """工厂函数：创建子工作流工具（符合 LangGraph 规范）"""

    def execute_workflow(  # type: ignore[valid-type]
        *,
        request: WorkflowRequest,  # 自动注入
        **kwargs: Any,  # LLM 传递的参数
    ) -> dict[str, Any]:
        """执行子工作流并返回 Artifact 模式结果"""
        delegate = get_delegate()
        artifact_store = get_artifact_store()

        tool_args = {k: v for k, v in kwargs.items() if v is not None}
        tool_args = _merge_request_context(tool_args, request)
        logger.info(f"[工具执行] {name}: {tool_args}")

        try:
            request_obj = _build_tool_request(workflow_type, tool_args)
            result = delegate.execute(workflow_type, request_obj)

            if not result.success:
                return {
                    "status": "error",
                    "message": result.error_message or "子工作流执行失败",
                }

            payload = _safe_parse_json(result.relate_data)
            if workflow_type == WorkflowType.SCHEDULE and payload is None:
                payload = {
                    "task_title": tool_args.get("task_title") or tool_args.get("taskTitle"),
                    "task_content": tool_args.get("task_content") or tool_args.get("taskContent"),
                    "cron_expression": tool_args.get("cron_expression"),
                }
            has_payload = payload is not None
            artifact_id = None

            if has_payload or workflow_type == WorkflowType.SCHEDULE:
                artifact_id = f"{name}_{int(time.time() * 1000)}_{uuid4().hex[:6]}"
                artifact_type = _resolve_artifact_type(name)
                content_cache = _build_content_cache(name, tool_args, request_obj)
                count = len(payload) if isinstance(payload, list) else None
                description = resolve_description(
                    DescriptionContext(
                        tool_name=name,
                        count=count,
                        req=tool_args,
                        tool_args=tool_args,
                        payload=payload,
                        workflow_type=workflow_type,
                    )
                )
                meta = {
                    "tool": name,
                    "workflow": result.workflow_name,
                    "count": count,
                }
                record = {
                    "id": artifact_id,
                    "type": artifact_type,
                    "description": description,
                    "content_cache": content_cache,
                    "meta": meta,
                    "payload": payload,
                }
                artifact_store.save_payload(
                    session_id=str(tool_args.get("session_id") or ""),
                    artifact_id=artifact_id,
                    payload=record,
                )
                return {
                    "status": "success",
                    "message": f"{name} 执行完成",
                    "data": {
                        "artifact_id": artifact_id,
                        "type": artifact_type,
                        "description": description,
                        "meta": meta,
                    },
                }

            return {
                "status": "success",
                "message": result.output or f"{name} 执行完成",
            }

        except Exception as e:
            logger.exception(f"[工具执行] {name} 异常: {e}")
            return {"status": "error", "message": str(e)}

    execute_workflow.__name__ = name
    # tool() 函数不支持 name 参数，工具名从函数名获取
    return tool(description=description, args_schema=args_schema)(execute_workflow)


# ============================================================
# 创建可执行工具
# ============================================================

select_zhiyi = _create_workflow_tool(
    name="select_zhiyi",
    description="知衣(淘宝)选品，搜索淘宝平台的服装商品",
    args_schema=SelectionArgs,
    workflow_type=WorkflowType.SELECT_ZHIYI,
)

select_douyi = _create_workflow_tool(
    name="select_douyi",
    description="抖衣(抖音)选品，搜索抖音平台的服装商品",
    args_schema=SelectionArgs,
    workflow_type=WorkflowType.SELECT_DOUYI,
)

select_abroad_goods = _create_workflow_tool(
    name="select_abroad_goods",
    description="海外探款选品，搜索 Amazon/Temu/独立站等海外平台的服装商品",
    args_schema=AbroadSelectionArgs,
    workflow_type=WorkflowType.SELECT_ABROAD_GOODS,
)

media_abroad_ins = _create_workflow_tool(
    name="media_abroad_ins",
    description="海外探款 INS 媒体搜索，搜索 Instagram 上的服装相关帖子",
    args_schema=MediaAbroadArgs,
    workflow_type=WorkflowType.MEDIA_ABROAD_INS,
)

media_zhikuan_ins = _create_workflow_tool(
    name="media_zhikuan_ins",
    description="知款 INS 媒体搜索，搜索 Instagram 上的服装相关帖子",
    args_schema=MediaArgs,
    workflow_type=WorkflowType.MEDIA_ZHIKUAN_INS,
)

media_zxh_xhs = _create_workflow_tool(
    name="media_zxh_xhs",
    description="知小红书媒体搜索，搜索小红书上的服装相关笔记",
    args_schema=MediaArgs,
    workflow_type=WorkflowType.MEDIA_ZXH_XHS,
)

shop_rank = _create_workflow_tool(
    name="shop_rank",
    description="店铺排行查询，查询淘宝平台的店铺排行榜",
    args_schema=ShopRankArgs,
    workflow_type=WorkflowType.SHOP,
)

image_search = _create_workflow_tool(
    name="image_search",
    description="图搜同款，根据用户上传的图片搜索相似服装商品",
    args_schema=ImageSearchArgs,
    workflow_type=WorkflowType.IMAGE_SEARCH,
)

image_create = _create_workflow_tool(
    name="image_create",
    description="文生图，根据用户的文字描述生成服装设计图片",
    args_schema=ImageCreateArgs,
    workflow_type=WorkflowType.IMAGE_CREATE,
)

image_edit = _create_workflow_tool(
    name="image_edit",
    description="图生图/改款，根据用户提供的参考图片和编辑指令生成新的设计图片",
    args_schema=ImageEditArgs,
    workflow_type=WorkflowType.IMAGE_EDIT,
)

schedule_task = _create_workflow_tool(
    name="schedule_task",
    description="创建定时任务，设置定时执行的监控或提醒任务",
    args_schema=ScheduleTaskArgs,
    workflow_type=WorkflowType.SCHEDULE,
)


def _inspect_artifact_core(
    *,
    session_id: str,
    artifact_id: str,
    limit: int = 20,
    filter_keyword: Optional[str] = None,
    filter_field: Optional[str] = None,
    filter_keywords: Optional[list[str]] = None,
) -> dict[str, Any]:
    """inspect_artifact 核心实现（从 Redis 查询，根据类型返回不同格式）"""
    store = get_artifact_store()
    record = store.get_payload(session_id=session_id, artifact_id=artifact_id)

    if record is None:
        # Artifact 不存在：返回可用资产列表帮助 LLM 纠错
        available = store.list_artifacts(session_id)
        available_ids = [a["id"] for a in available]

        return {
            "status": "error",
            "message": f"资产 {artifact_id} 不存在",
            "data": {"available_artifacts": available_ids},
        }

    artifact_type = record.get("type")
    payload = record.get("payload")
    content_cache = record.get("content_cache")
    description = record.get("description")
    meta = record.get("meta")

    # CASE 1: 列表类 - 返回分页数据
    if artifact_type == "product_list":
        safe_limit = min(max(limit, 1), 50)  # 硬性限制最大 50 条，防止 Context 爆炸
        items, total, has_more = store.get_list_data(
            session_id=session_id,
            artifact_id=artifact_id,
            page=1,
            page_size=safe_limit,
        )

        # 应用过滤逻辑
        if filter_keyword or filter_keywords:
            items = _apply_list_filters(items, filter_keyword, filter_field, filter_keywords)

        return {
            "status": "success",
            "message": f"已展示前 {len(items)} 条数据（共 {total} 条）",
            "data": {
                "artifact_id": artifact_id,
                "type": "product_list",
                "description": description,
                "pagination": {
                    "page": 1,
                    "page_size": safe_limit,
                    "total": total,
                    "has_more": has_more,
                },
                "items": items,
            },
        }

    # ═══════════════════════════════════════════════════════════
    # CASE 2: 图片类 - 返回缓存的 caption（⚠️ 核心优化）
    # ═══════════════════════════════════════════════════════════
    elif artifact_type == "image_asset":
        return {
            "status": "success",
            "message": "图片资产详情",
            "data": {
                "artifact_id": artifact_id,
                "type": "image_asset",
                "description": description,
                "caption": content_cache,  # ⚠️ 直接返回缓存，不调用 VLM
                "meta": meta,
            },
        }

    # ═══════════════════════════════════════════════════════════
    # CASE 3: 任务类 - 返回任务配置
    # ═══════════════════════════════════════════════════════════
    elif artifact_type == "task_receipt":
        return {
            "status": "success",
            "message": "任务资产详情",
            "data": {
                "artifact_id": artifact_id,
                "type": "task_receipt",
                "description": description,
                "config": {
                    "task_id": meta.get("task_id") if meta else None,
                    "title": meta.get("title") if meta else None,
                    "cron": meta.get("cron") if meta else None,
                    "cron_human": meta.get("cron_human") if meta else None,
                    "status": meta.get("status") if meta else None,
                    "next_run": meta.get("next_run") if meta else None,
                },
            },
        }

    # 未知类型
    return {
        "status": "error",
        "message": f"未知的资产类型: {artifact_type}",
        "data": None,
    }


@tool(
    "inspect_artifact",
    description="查看资产详情，获取之前生成的选品列表、图片或任务的具体内容",
    args_schema=InspectArtifactArgs,
)
def inspect_artifact_tool(  # type: ignore[valid-type]
    *,
    request: WorkflowRequest,  # 自动注入
    artifact_id: str,
    limit: int = 20,
    filter_keyword: Optional[str] = None,
    filter_field: Optional[str] = None,
    filter_keywords: Optional[list[str]] = None,
) -> dict[str, Any]:
    """查看资产详情（图片类返回缓存描述，列表类支持分页和过滤，硬性限制 limit 最大 50）"""
    return _inspect_artifact_core(
        session_id=request.session_id,
        artifact_id=artifact_id,
        limit=limit,
        filter_keyword=filter_keyword,
        filter_field=filter_field,
        filter_keywords=filter_keywords,
    )


def _apply_list_filters(
    data: list,
    filter_keyword: str | None,
    filter_field: str | None,
    filter_keywords: list[str] | None,
) -> list:
    """应用过滤逻辑到列表数据"""
    if not any([filter_keyword, filter_keywords]):
        return data

    def match_item(item: Any) -> bool:
        # 确定搜索目标
        if filter_field and isinstance(item, dict):
            target = str(item.get(filter_field, ""))
        else:
            target = str(item)

        # 单关键词匹配
        if filter_keyword and filter_keyword in target:
            return True
        # 多关键词匹配（OR 逻辑）
        if filter_keywords and any(kw in target for kw in filter_keywords):
            return True
        # 如果指定了关键词但没匹配上
        return not (filter_keyword or filter_keywords)

    return [item for item in data if match_item(item)]


# ============================================================
# 导出
# ============================================================

ALL_TOOLS: list[BaseTool] = [
    select_zhiyi,
    select_douyi,
    select_abroad_goods,
    media_abroad_ins,
    media_zhikuan_ins,
    media_zxh_xhs,
    shop_rank,
    image_search,
    image_create,
    image_edit,
    schedule_task,
    inspect_artifact_tool,
]

__all__ = ["ALL_TOOLS"]
