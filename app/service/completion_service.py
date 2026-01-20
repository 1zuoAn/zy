# -*- coding: utf-8 -*-
"""
输入补全服务
用户还没输完句子就自动给补全，补全内容与平台功能相关。
"""

from __future__ import annotations

import asyncio
import re

from cachetools import TTLCache
from loguru import logger

from app.config import settings

from app.core.config.constants import LlmModelName
from app.core.tools.llm_factory import LLMFactory
from app.schemas.request.completion_request import CompletionRequest

# ============================================================
# 配置
# ============================================================

COMPLETION_MODEL = LlmModelName.OPENROUTER_GPT_4O_MINI
COMPLETION_TEMPERATURE = 0.7
COMPLETION_MAX_TOKENS = 60
COMPLETION_TIMEOUT_SEC = 8.0
COMPLETION_MAX_TOTAL_CHARS = 60

CACHE_TTL = 300
CACHE_MAX_SIZE = 512

# 正则
PLACEHOLDER_PATTERN = re.compile(r"@\{[^}]*\}?")
MULTISPACE_PATTERN = re.compile(r"\s+")

# ============================================================
# Prompt
# ============================================================

COMPLETION_PROMPT = """你是服装选品平台的输入补全助手。用户正在输入搜索词，你需要预测并补全剩余部分，用户输入的内容中包含“@”或符号的话，忽略掉这个符号进行理解。

## 平台能力
- 商品选品：筛选服装商品
- 趋势报告：品类趋势、爆款榜单
- 媒体选品：小红书笔记选品、INS帖子选品
— 生图改图：AI生成服装图片、图片风格转换

## 当前上下文
{context}

## 输出规则
1. 只输出【补全部分】，不重复用户已输入内容，补全内容要和用户输入相关，语言要自然
2. 补全后整句不超过{max_chars}字，你最多输出{remain}字
3. 输出纯文本，无标点开头，无引号，无解释
4. 禁止：@ 符号、"店铺"、"品牌"、动作词（生成/推荐/分析）
5. 优先补充：时间范围、排序方式、品类、筛选条件
6.补全的内容必须符合人的思维习惯，并且不能超出平台的能力
7.补充的内容只能是单一款式，不能是多个款式，例如：“连衣裙”。不能是”连衣裙、短裙、长裙等“。

## 限制：遇到以下情况，什么都不用输出：                    
    - 乱码/键盘乱按（如 asdf、qwer、jkl）                                 
    - 无意义重复（如 啊啊啊、111）                                        
    - 纯符号或数字                                                        
    - 完全无法理解的内容
    - 用户在闲聊而不是输入搜索词

## 示例
输入：近30天 → 输出：的新款牛仔裤，按销量排序
输入：小红书热门 → 输出：的外套推荐，价格适中的
输入：帮我生成 → 输出：一张夏季短裙图片
输入：羽绒服 → 输出：长款连帽款，适合北方

用户输入：{input}
补全："""


# ============================================================
# 服务
# ============================================================


class CompletionService:
    """补全服务"""

    def __init__(self) -> None:
        self._cache: TTLCache = TTLCache(maxsize=CACHE_MAX_SIZE, ttl=CACHE_TTL)
        self._cache_lock = asyncio.Lock()
        self._llm = None
        self._llm_lock = asyncio.Lock()

    def _normalize(self, text: str) -> str:
        """标准化输入"""
        if not text:
            return ""
        text = PLACEHOLDER_PATTERN.sub(" ", text).replace("@", " ")
        return MULTISPACE_PATTERN.sub(" ", text).strip()

    def _should_skip(self, raw: str) -> bool:
        """是否跳过补全"""
        s = (raw or "").strip()
        return not s or s.endswith("@") or (s.count("{") > s.count("}"))

    def _build_context(self, req: CompletionRequest) -> str:
        """构建上下文描述"""
        parts = []
        if pe := getattr(req, "preferred_entity", None):
            parts.append(f"模式: {pe}")
        if ind := getattr(req, "industry", None):
            parts.append(f"行业: {ind}")
        if abroad := getattr(req, "abroad_type", None):
            parts.append(f"站点: {abroad}")
        return "、".join(parts) if parts else "通用选品"

    def _clean_output(self, text: str, normalized_input: str) -> str:
        """清理输出"""
        if not text:
            return ""
        # 取首行，去引号
        line = text.splitlines()[0].strip().strip('"""\'\"')
        # LLM 可能返回无效内容
        if line in ("空", "无", "null", "none", "empty", "None"):
            return ""
        # 过滤 LLM 的拒绝/道歉/元回复
        reject_keywords = (
            "抱歉", "无法", "对不起", "sorry", "cannot", "无意义",
            "什么都不", "不输出", "无需补全", "用户输入", "补全：",
            "没有相关", "无法补全", "无效输入", "没有可以", "请重新输入",
            "无输出", "（无", "(无"
        )
        if any(kw in line.lower() for kw in reject_keywords):
            return ""
        # 去掉重复的输入前缀
        if line.startswith(normalized_input):
            line = line[len(normalized_input):]
        # 去掉输入尾部重复
        for k in range(min(10, len(normalized_input)), 2, -1):
            if line.startswith(normalized_input[-k:]):
                line = line[k:]
                break
        # 清理
        line = line.replace("@", "").replace("店铺", "").replace("品牌", "")
        line = line.lstrip("、，,．.。 ")
        # 长度限制
        remain = COMPLETION_MAX_TOTAL_CHARS - len(normalized_input)
        return line[:remain].rstrip() if remain > 0 else ""

    async def _get_llm(self):
        async with self._llm_lock:
            if self._llm is None:
                self._llm = LLMFactory.create_openrouter_llm(
                    model=COMPLETION_MODEL.value,
                    temperature=COMPLETION_TEMPERATURE,
                    max_tokens=COMPLETION_MAX_TOKENS,
                )
        return self._llm

    async def get_completion(self, request: CompletionRequest) -> str:
        raw_input = request.input or ""
        if self._should_skip(raw_input):
            return ""

        normalized = self._normalize(raw_input)
        if not normalized:
            return ""

        remain = COMPLETION_MAX_TOTAL_CHARS - len(normalized)
        if remain <= 0:
            return ""

        # 缓存
        cache_key = (normalized, self._build_context(request))
        async with self._cache_lock:
            if cached := self._cache.get(cache_key):
                return cached

        # 构建 prompt
        prompt = COMPLETION_PROMPT.format(
            context=self._build_context(request),
            max_chars=COMPLETION_MAX_TOTAL_CHARS,
            remain=remain,
            input=normalized,
        )

        # 调用 LLM
        try:
            llm = await self._get_llm()
            resp = await asyncio.wait_for(llm.ainvoke(prompt), timeout=COMPLETION_TIMEOUT_SEC)
            raw_content = getattr(resp, "content", "")
            result = self._clean_output(raw_content, normalized)
            logger.debug(f"[补全服务] 输入={normalized}, LLM原始输出={raw_content}, 清理后={result}")
        except Exception as e:
            logger.warning(f"[补全服务] 调用失败: {e}")
            result = ""

        # 缓存结果
        if result:
            async with self._cache_lock:
                self._cache[cache_key] = result

        return result


completion_service = CompletionService()

__all__ = ["CompletionService", "completion_service"]
