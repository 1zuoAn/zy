"""
建议生成服务（仅 LLM）

流程：
1. 行业选择（LLM）
2. 维度选择（LLM，3 个不同维度）
3. 值选择 + 成功文案生成（LLM，每维度 1 次并行）

失败场景：LLM 单次生成 3 条放宽建议。
"""

from __future__ import annotations

import json
import time
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from loguru import logger
from pydantic import BaseModel, Field

from app.core.config.constants import LlmModelName
from app.core.tools.llm_factory import LLMFactory
from app.resources.suggestion_type_library import TYPE_LIBRARY_ROWS
from app.schemas.request.suggestion_request import SuggestionGenerateRequest
from app.schemas.response.suggestion_response import (
    SuggestionGenerateResponse,
    SuggestionItem,
)

# =============================================================================
# 常量和配置
# =============================================================================

SUCCESS_COUNT = 3
FAILED_COUNT = 3

TEXT_MIN_LEN = 5
TEXT_MAX_LEN = 15

HUANXIN_MODEL = LlmModelName.HUANXIN_GEMINI_2_5_FLASH.value
LLM_MAX_TOKENS = 8192


# =============================================================================
# 数据模型
# =============================================================================


class RewriteItem(BaseModel):
    """LLM 文案结果项"""

    id: int
    text: str
    filled_query: str


class RewriteDecision(BaseModel):
    """LLM 文案结果集合"""

    items: list[RewriteItem] = Field(default_factory=list)


class IndustryDecision(BaseModel):
    """LLM 选择的行业"""

    industry: str


class DimensionDecision(BaseModel):
    """LLM 选择的维度列表"""

    dimensions: list[str] = Field(default_factory=list)


class SuccessValueDecision(BaseModel):
    """LLM 选择的值 + 成功文案"""

    dimension: str
    value: str
    text: str
    filled_query: str


@dataclass(frozen=True)
class TypeLibrary:
    """类型库数据结构"""

    index: dict[str, dict[str, list[str]]]
    industries: list[str]


# =============================================================================
# 类型库
# =============================================================================


@lru_cache(maxsize=1)
def _load_type_library() -> TypeLibrary:
    """加载类型库，构建行业/维度索引"""
    if not TYPE_LIBRARY_ROWS:
        logger.warning("类型库为空")
        return TypeLibrary(index={}, industries=[])

    index: dict[str, dict[str, list[str]]] = {}
    industries: list[str] = []

    for industry, dimension, value in TYPE_LIBRARY_ROWS:
        if industry not in index:
            index[industry] = {}
            industries.append(industry)
        if dimension not in index[industry]:
            index[industry][dimension] = []
        if value not in index[industry][dimension]:
            index[industry][dimension].append(value)

    return TypeLibrary(index=index, industries=industries)


# =============================================================================
# Prompt 模板
# =============================================================================


_INDUSTRY_SELECTION_PROMPT = """你是行业选择器，请只从【行业列表】中选择一个最匹配用户输入的行业。

【用户输入】{user_query}
【行业提示】{industry_hint}
【行业列表】{industry_list}

【要求 - 严格遵守】
- 只能从列表中选择一个行业，不要改写或新增。
- 如果用户输入不明确或为空，优先使用【行业提示】；若提示也为空，选择列表中最通用的行业。
- 仅输出 JSON，不要输出任何解释。

【JSON 格式】
{{"industry":"..."}}
"""


_DIMENSION_SELECTION_PROMPT = """你是维度选择器，请从【维度列表】中选择 3 个不同维度。

【用户输入】{user_query}
【行业】{industry}
【维度列表】{dimensions_text}

【要求 - 严格遵守】
- 必须从列表中选择，不能新增或改写。
- 必须输出 3 个互不相同的维度。
- 优先选择用户未表达的维度；若不足，再选择已表达维度补齐。
- 仅输出 JSON，不要输出任何解释。

【JSON 格式】
{{"dimensions":["...","...","..."]}}
"""


_SUCCESS_VALUE_SUGGESTION_PROMPT = """你是追问建议生成器（成功场景），请为给定维度从候选值中选择 1 个值，并生成 1 条建议。

【用户输入】{user_query}
【行业】{industry}
【维度】{dimension}
【候选值】{values_text}

【要求 - 严格遵守】
- value 必须从候选值中选择 1 个，不能新增或改写。
- dimension 必须等于【维度】原文，不要改写。
- 若用户输入已表达该维度的某个值，优先选不同值；若无法避免，再任选其一。
- text 长度 {min_len}~{max_len} 字，简洁清晰，必须包含该 value 原文。
- text 使用短语或陈述句，不要问号/感叹号/括号，不要客服式话术。
- text 在保持原义前提下更生动，避免空泛词汇。
- filled_query 必须是完整自然口语句（第一人称更好），必须包含用户输入的核心语义与该 value，不添加新条件。
- filled_query 不要出现“请/是否/能否/推荐/建议/为你/为您/为我/根据/生成”等指令或系统提示语。
- filled_query 在保持原义前提下更生动自然，不要夸张或引入新条件。
- 不得改变用户核心品类/行业词，不得新增品牌/店铺/人群/场景/功能等条件。
- 语义需自然、无矛盾；不要把用户已有值与新 value 混在一起。
- 仅输出 JSON，不要输出任何解释。

【JSON 格式】
{{"dimension":"...","value":"...","text":"...","filled_query":"..."}}
"""


_FAILED_SUGGESTION_PROMPT = """你是追问建议生成器（失败场景），目标是放宽用户的搜索条件。

【用户输入】{user_query}

【任务 - 严格遵守】
1) 识别用户已表达的限制条件（如颜色/价格/材质/风格/季节/图案/人群/场景/款式/功能/品牌/店铺等）。
2) 输出 3 条建议，每条只移除一个限制条件。
3) 若可移除条件不足 3 个：剩余建议使用“行业放宽”（保留核心品类/行业词，放宽到更泛的搜索）。
4) 若行业也无法判断：使用通用放宽建议补齐。
5) 每条建议的放宽点必须不同，避免重复。
6) 识别店铺名或品牌名时，不要进行内部拆分。
【输出要求】
- text 长度 {min_len}~{max_len} 字，简洁清晰。
- text 在保持原义前提下更生动，避免空泛词汇。
- filled_query 必须是完整自然口语句，保持原意但放宽条件，并且更生动自然。
- 仅输出 JSON，不要输出任何解释。

【JSON 格式】
{{"items":[{{"id":0,"text":"...","filled_query":"..."}},{{"id":1,"text":"...","filled_query":"..."}},{{"id":2,"text":"...","filled_query":"..."}}]}}
"""


# =============================================================================
# 文本工具
# =============================================================================


def _dedupe_keep_order(values: Sequence[str]) -> list[str]:
    """去重并保持顺序"""
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _format_dimension_list(dimensions: Sequence[str]) -> str:
    """格式化维度列表用于提示词"""
    return "、".join(dimensions) if dimensions else "无"


def _format_value_list(values: Sequence[str]) -> str:
    """格式化候选值列表用于提示词"""
    return "、".join(values) if values else "无"


# =============================================================================
# 服务类
# =============================================================================


class SuggestionService:
    """建议生成服务，负责生成成功和失败场景的追问建议"""

    def __init__(self) -> None:
        self._industry_parser = PydanticOutputParser(pydantic_object=IndustryDecision)
        self._industry_prompt = PromptTemplate.from_template(
            _INDUSTRY_SELECTION_PROMPT + "\n{format_instructions}",
            partial_variables={
                "format_instructions": self._industry_parser.get_format_instructions()
            },
        )
        self._dimension_parser = PydanticOutputParser(pydantic_object=DimensionDecision)
        self._dimension_prompt = PromptTemplate.from_template(
            _DIMENSION_SELECTION_PROMPT + "\n{format_instructions}",
            partial_variables={
                "format_instructions": self._dimension_parser.get_format_instructions()
            },
        )
        self._success_value_parser = PydanticOutputParser(
            pydantic_object=SuccessValueDecision
        )
        self._success_value_prompt = PromptTemplate.from_template(
            _SUCCESS_VALUE_SUGGESTION_PROMPT + "\n{format_instructions}",
            partial_variables={
                "format_instructions": self._success_value_parser.get_format_instructions(),
                "min_len": TEXT_MIN_LEN,
                "max_len": TEXT_MAX_LEN,
            },
        )
        self._failed_parser = PydanticOutputParser(pydantic_object=RewriteDecision)
        self._failed_prompt = PromptTemplate.from_template(
            _FAILED_SUGGESTION_PROMPT + "\n{format_instructions}",
            partial_variables={
                "format_instructions": self._failed_parser.get_format_instructions(),
                "min_len": TEXT_MIN_LEN,
                "max_len": TEXT_MAX_LEN,
            },
        )

    def _create_llm(self, max_tokens: int | None = None):
        """统一创建 LLM 实例"""
        return LLMFactory.create_huanxin_llm(
            model=HUANXIN_MODEL,
            max_tokens=max_tokens or LLM_MAX_TOKENS,
        )

    def _invoke_chain(self, prompt, llm, parser, payload, error_message: str):
        """统一封装 LLM 调用与解析"""
        start_time = time.perf_counter()
        thread_name = threading.current_thread().name
        try:
            format_start = time.perf_counter()
            prompt_value = prompt.format_prompt(**payload)
            prompt_text = (
                prompt_value.to_string()
                if hasattr(prompt_value, "to_string")
                else str(prompt_value)
            )
            prompt_chars = len(prompt_text)
            format_elapsed = time.perf_counter() - format_start

            llm_start = time.perf_counter()
            llm_result = llm.invoke(prompt_value)
            llm_elapsed = time.perf_counter() - llm_start

            parse_start = time.perf_counter()
            result = parser.invoke(llm_result)
            parse_elapsed = time.perf_counter() - parse_start
            total_elapsed = time.perf_counter() - start_time
            logger.info(
                "LLM 调用耗时: total={:.2f}s llm={:.2f}s parse={:.2f}s "
                "format={:.2f}s chars={} thread={} - {}",
                total_elapsed,
                llm_elapsed,
                parse_elapsed,
                format_elapsed,
                prompt_chars,
                thread_name,
                error_message,
            )
            return result
        except Exception as e:
            total_elapsed = time.perf_counter() - start_time
            logger.warning(
                "{}: {} (耗时: {:.2f}秒 thread={})",
                error_message,
                e,
                total_elapsed,
                thread_name,
            )
            return None

    def _get_raw_output(self, prompt, llm, payload, error_message: str) -> str | None:
        """获取 LLM 原始输出，便于兜底解析"""
        start_time = time.perf_counter()
        thread_name = threading.current_thread().name
        try:
            format_start = time.perf_counter()
            prompt_value = prompt.format_prompt(**payload)
            prompt_text = (
                prompt_value.to_string()
                if hasattr(prompt_value, "to_string")
                else str(prompt_value)
            )
            prompt_chars = len(prompt_text)
            format_elapsed = time.perf_counter() - format_start

            llm_start = time.perf_counter()
            raw_output = llm.invoke(prompt_value)
            llm_elapsed = time.perf_counter() - llm_start
            result = raw_output.content if hasattr(raw_output, "content") else str(raw_output)
            total_elapsed = time.perf_counter() - start_time
            logger.info(
                "LLM 原始输出耗时: total={:.2f}s llm={:.2f}s format={:.2f}s "
                "chars={} thread={} - {}",
                total_elapsed,
                llm_elapsed,
                format_elapsed,
                prompt_chars,
                thread_name,
                error_message,
            )
            return result
        except Exception as e:
            total_elapsed = time.perf_counter() - start_time
            logger.warning(
                "{}: {} (耗时: {:.2f}秒 thread={})",
                error_message,
                e,
                total_elapsed,
                thread_name,
            )
            return None

    def generate(self, request: SuggestionGenerateRequest) -> SuggestionGenerateResponse:
        """主入口：成功 3 步 LLM 链路（含并行）+ 失败单步 LLM 生成"""
        user_query = (request.user_query or "").strip()
        industry_hint = (request.industry or "").strip()

        library = _load_type_library()
        if not library.index:
            return SuggestionGenerateResponse(
                success_suggestions=[],
                failed_suggestions=[],
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            failed_future = executor.submit(self._generate_failed_suggestions, user_query)
            success_items = self._generate_success_suggestions(
                user_query, industry_hint, library
            )
            failed_items = failed_future.result()

        return SuggestionGenerateResponse(
            success_suggestions=success_items,
            failed_suggestions=failed_items,
        )

    def _generate_success_suggestions(
        self,
        user_query: str,
        industry_hint: str,
        library: TypeLibrary,
    ) -> list[SuggestionItem]:
        """成功链路：行业 -> 维度 -> (按维度并行 选值+文案)"""
        industry = self._call_llm_for_industry(user_query, industry_hint, library.industries)
        if not industry or industry not in library.index:
            logger.warning("行业选择失败")
            return []

        industry_dimensions = library.index.get(industry, {})
        dimensions = self._call_llm_for_dimensions(
            user_query, industry, list(industry_dimensions.keys())
        )
        if len(dimensions) != SUCCESS_COUNT:
            logger.warning("维度选择数量不足")
            return []
        # 保持与维度选择顺序一致
        results: list[SuggestionItem | None] = [None] * len(dimensions)
        with ThreadPoolExecutor(max_workers=SUCCESS_COUNT) as executor:
            future_map = {
                executor.submit(
                    self._generate_success_item_for_dimension,
                    user_query,
                    industry,
                    dimension,
                    industry_dimensions,
                ): idx
                for idx, dimension in enumerate(dimensions)
            }
            for future, idx in future_map.items():
                try:
                    item = future.result()
                except Exception as exc:
                    logger.warning(f"维度并行链路异常: {exc}")
                    return []
                if not item:
                    logger.warning("维度并行链路失败")
                    return []
                results[idx] = item

        return [item for item in results if item]

    def _generate_failed_suggestions(
        self,
        user_query: str,
    ) -> list[SuggestionItem]:
        """失败链路：单步 LLM 直接生成"""
        decision = self._call_llm_for_failed_suggestions(user_query)
        return self._build_items_from_decision(decision, FAILED_COUNT)

    def _generate_success_item_for_dimension(
        self,
        user_query: str,
        industry: str,
        dimension: str,
        industry_dimensions: dict[str, list[str]],
    ) -> SuggestionItem | None:
        """单维度并行链路：选值 + 文案"""
        candidates = industry_dimensions.get(dimension, [])
        decision = self._call_llm_for_success_item(
            user_query, industry, dimension, candidates
        )
        if not decision:
            logger.warning("单维度生成失败")
            return None
        return SuggestionItem(text=decision.text, filled_query=decision.filled_query)

    def _build_items_from_decision(
        self,
        decision: RewriteDecision | None,
        limit: int,
    ) -> list[SuggestionItem]:
        """将 LLM 输出的 items 转换为 SuggestionItem"""
        if not decision or not decision.items:
            return []
        items = sorted(decision.items, key=lambda item: item.id)
        if len(items) < limit:
            logger.warning("LLM 输出数量不足")
            return []
        return [
            SuggestionItem(text=item.text, filled_query=item.filled_query)
            for item in items[:limit]
        ]

    # LLM 决策：行业、维度、成功单维度、失败
    def _call_llm_for_industry(
        self,
        user_query: str,
        industry_hint: str,
        industries: Sequence[str],
    ) -> str:
        """使用 LLM 选择行业"""
        llm = self._create_llm()
        payload = {
            "user_query": user_query,
            "industry_hint": industry_hint or "无",
            "industry_list": "、".join(industries) if industries else "无",
        }
        label = f"行业选择(industries={len(industries)})"
        result = self._invoke_chain(
            self._industry_prompt,
            llm,
            self._industry_parser,
            payload,
            label,
        )
        if result and result.industry in industries:
            return result.industry

        raw_output = self._get_raw_output(
            self._industry_prompt,
            llm,
            payload,
            f"{label}(原始输出)",
        )
        if raw_output:
            try:
                parsed = json.loads(raw_output)
                decision = IndustryDecision(**parsed)
                if decision.industry in industries:
                    return decision.industry
            except (json.JSONDecodeError, ValueError, TypeError):
                logger.warning("行业选择解析失败")
        return ""

    def _call_llm_for_dimensions(
        self,
        user_query: str,
        industry: str,
        dimensions: Sequence[str],
    ) -> list[str]:
        """使用 LLM 选择维度"""
        if not dimensions:
            return []
        llm = self._create_llm()
        payload = {
            "user_query": user_query,
            "industry": industry,
            "dimensions_text": _format_dimension_list(dimensions),
        }
        label = f"维度选择(dimensions={len(dimensions)})"
        result = self._invoke_chain(
            self._dimension_prompt,
            llm,
            self._dimension_parser,
            payload,
            label,
        )
        if result and result.dimensions:
            selected = _dedupe_keep_order(
                [dim for dim in result.dimensions if dim in dimensions]
            )
            if len(selected) == SUCCESS_COUNT:
                return selected

        raw_output = self._get_raw_output(
            self._dimension_prompt,
            llm,
            payload,
            f"{label}(原始输出)",
        )
        if raw_output:
            try:
                parsed = json.loads(raw_output)
                decision = DimensionDecision(**parsed)
                selected = _dedupe_keep_order(
                    [dim for dim in decision.dimensions if dim in dimensions]
                )
                if len(selected) == SUCCESS_COUNT:
                    return selected
            except (json.JSONDecodeError, ValueError, TypeError):
                logger.warning("维度选择解析失败")
        return []

    def _call_llm_for_success_item(
        self,
        user_query: str,
        industry: str,
        dimension: str,
        candidates: Sequence[str],
    ) -> SuccessValueDecision | None:
        """使用 LLM 为单维度选值并生成成功文案"""
        if not candidates:
            return None
        llm = self._create_llm()
        payload = {
            "user_query": user_query,
            "industry": industry,
            "dimension": dimension,
            "values_text": _format_value_list(candidates),
        }
        label = f"成功场景-选值+文案生成(dim={dimension}, values={len(candidates)})"
        result = self._invoke_chain(
            self._success_value_prompt,
            llm,
            self._success_value_parser,
            payload,
            label,
        )
        decision = self._validate_success_value_decision(result, dimension, candidates)
        if decision:
            return decision

        raw_output = self._get_raw_output(
            self._success_value_prompt,
            llm,
            payload,
            f"{label}(原始输出)",
        )
        if raw_output:
            try:
                parsed = json.loads(raw_output)
                decision = SuccessValueDecision(**parsed)
                decision = self._validate_success_value_decision(
                    decision, dimension, candidates
                )
                if decision:
                    return decision
            except (json.JSONDecodeError, ValueError, TypeError):
                logger.warning("成功场景选值+文案解析失败")
        return None

    def _validate_success_value_decision(
        self,
        decision: SuccessValueDecision | None,
        dimension: str,
        candidates: Sequence[str],
    ) -> SuccessValueDecision | None:
        """校验值是否来自候选列表，维度是否匹配"""
        if not decision:
            return None
        if decision.dimension != dimension:
            return None
        if decision.value not in candidates:
            return None
        return decision

    def _call_llm_for_failed_suggestions(
        self,
        user_query: str,
    ) -> RewriteDecision | None:
        """使用 LLM 生成失败场景建议"""
        llm = self._create_llm()
        payload = {"user_query": user_query}
        label = f"失败场景-文案生成(query_len={len(user_query)})"
        result = self._invoke_chain(
            self._failed_prompt,
            llm,
            self._failed_parser,
            payload,
            label,
        )
        if result and len(result.items) >= FAILED_COUNT:
            return result

        raw_output = self._get_raw_output(
            self._failed_prompt,
            llm,
            payload,
            f"{label}(原始输出)",
        )
        if raw_output:
            try:
                parsed = json.loads(raw_output)
                decision = RewriteDecision(**parsed)
                if len(decision.items) >= FAILED_COUNT:
                    return decision
            except (json.JSONDecodeError, ValueError, TypeError):
                logger.warning("失败场景文案解析失败")
        return None


suggestion_service = SuggestionService()
