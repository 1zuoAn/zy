# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.core.config.constants import WorkflowType
from app.schemas.request.workflow_request import WorkflowRequest


def _get_req_value(req: WorkflowRequest | dict[str, Any], key: str) -> Any:
    if isinstance(req, dict):
        return req.get(key)
    return getattr(req, key, None)


def _shorten_text(text: str | None, limit: int = 40) -> str:
    if not text:
        return ""
    cleaned = str(text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: max(0, limit - 3)]}..."


@dataclass(slots=True)
class DescriptionContext:
    tool_name: str
    count: int | None
    req: WorkflowRequest | dict[str, Any]
    tool_args: dict[str, Any]
    payload: Any | None = None
    workflow_type: WorkflowType | None = None

    def get(self, key: str, default: Any = None) -> Any:
        value = self.tool_args.get(key)
        if value not in (None, ""):
            return value
        req_value = _get_req_value(self.req, key)
        if req_value not in (None, ""):
            return req_value
        return default

    def get_any(self, *keys: str, default: Any = None) -> Any:
        for key in keys:
            value = self.tool_args.get(key)
            if value not in (None, ""):
                return value
        for key in keys:
            req_value = _get_req_value(self.req, key)
            if req_value not in (None, ""):
                return req_value
        return default

    @property
    def user_query(self) -> str:
        return _shorten_text(self.get("user_query") or "", 40)

    @property
    def preferred_entity(self) -> str:
        return self.get("preferred_entity") or ""

    @property
    def abroad_type(self) -> str:
        return self.get("abroad_type") or ""

    @property
    def count_text(self) -> str:
        return f"，共{self.count}条" if self.count is not None else ""


DescriptionBuilder = Callable[[DescriptionContext], str]
_DESCRIPTION_REGISTRY: dict[str, DescriptionBuilder] = {}


def register_description(tool_name: str) -> Callable[[DescriptionBuilder], DescriptionBuilder]:
    def decorator(builder: DescriptionBuilder) -> DescriptionBuilder:
        _DESCRIPTION_REGISTRY[tool_name] = builder
        return builder

    return decorator


def resolve_description(ctx: DescriptionContext) -> str:
    builder = _DESCRIPTION_REGISTRY.get(ctx.tool_name) or _DESCRIPTION_REGISTRY.get("*")
    if builder:
        return builder(ctx)
    return "任务已完成"


def _build_list_desc(ctx: DescriptionContext, platform: str, suffix: str) -> str:
    base = f"{platform}{suffix}".strip()
    if ctx.user_query:
        return f"{base}：{ctx.user_query}{ctx.count_text}"
    return f"{base}{ctx.count_text}"


@register_description("select_zhiyi")
def _desc_select_zhiyi(ctx: DescriptionContext) -> str:
    return _build_list_desc(ctx, "知衣", "选品结果")


@register_description("select_douyi")
def _desc_select_douyi(ctx: DescriptionContext) -> str:
    return _build_list_desc(ctx, "抖衣", "选品结果")


@register_description("select_abroad_goods")
def _desc_select_abroad(ctx: DescriptionContext) -> str:
    platform = ctx.abroad_type or ctx.preferred_entity or "海外站点"
    return _build_list_desc(ctx, platform, "选品结果")


@register_description("media_abroad_ins")
def _desc_media_abroad(ctx: DescriptionContext) -> str:
    return _build_list_desc(ctx, "海外INS", "媒体内容结果")


@register_description("media_zhikuan_ins")
def _desc_media_zhikuan(ctx: DescriptionContext) -> str:
    return _build_list_desc(ctx, "知款INS", "媒体内容结果")


@register_description("media_zxh_xhs")
def _desc_media_xhs(ctx: DescriptionContext) -> str:
    return _build_list_desc(ctx, "小红书", "媒体内容结果")


@register_description("shop_rank")
def _desc_shop_rank(ctx: DescriptionContext) -> str:
    platform = ctx.preferred_entity or "平台"
    base = f"{platform}店铺排行"
    if ctx.user_query:
        return f"{base}，关键词“{ctx.user_query}”{ctx.count_text}"
    return f"{base}{ctx.count_text}"


@register_description("image_search")
def _desc_image_search(ctx: DescriptionContext) -> str:
    base = "图搜相似款结果"
    if ctx.user_query:
        return f"{base}：{ctx.user_query}{ctx.count_text}"
    return f"{base}{ctx.count_text}"


@register_description("image_create")
def _desc_image_create(ctx: DescriptionContext) -> str:
    prompt = ctx.get_any("image_prompt", "user_query", default="")
    prompt = _shorten_text(prompt, 60)
    return f"图片生成：{prompt}" if prompt else "图片已生成"


@register_description("image_edit")
def _desc_image_edit(ctx: DescriptionContext) -> str:
    prompt = ctx.get_any("edit_prompt", "user_query", default="")
    prompt = _shorten_text(prompt, 60)
    return f"图片编辑：{prompt}" if prompt else "图片已编辑"


@register_description("schedule_task")
def _desc_schedule_task(ctx: DescriptionContext) -> str:
    title = ctx.get_any("task_title", "taskTitle", default="")
    content = ctx.get_any("task_content", "taskContent", default="")
    cron = ctx.get_any("cron_expression", "task_default_cron", "taskDefaultCron", default="")
    brief = _shorten_text(title or content, 40)
    desc = f"定时任务：{brief}" if brief else "定时任务已生成"
    if cron:
        desc = f"{desc}，cron={cron}"
    return desc


@register_description("*")
def _desc_default(ctx: DescriptionContext) -> str:
    fallback = "任务已完成"
    if ctx.user_query:
        return f"{fallback}：{ctx.user_query}"
    return fallback


__all__ = ["DescriptionContext", "resolve_description"]
