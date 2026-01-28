# -*- coding: utf-8 -*-
"""
输入补全服务
用户还没输完句子就自动给补全，补全内容与平台功能相关。
"""

from __future__ import annotations

import asyncio
import re
from functools import partial

from cachetools import TTLCache
from loguru import logger

from app.core.clients.es_memory_client import es_memory_client
from app.core.config.constants import LlmModelName
from app.core.tools.llm_factory import LLMFactory
from app.schemas.request.completion_request import CompletionRequest

# ============================================================
# 配置
# ============================================================

COMPLETION_MODEL = LlmModelName.OPENROUTER_GEMINI_2_5_FLASH  # 与其他工作流保持一致
COMPLETION_TEMPERATURE = 0.7
COMPLETION_MAX_TOKENS = 60
COMPLETION_TIMEOUT_SEC = 8.0
COMPLETION_MAX_TOTAL_CHARS = 60
COMPLETION_RETRY_ATTEMPTS = 2
COMPLETION_RETRY_BACKOFF_SEC = 0.25

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
1.（最重要） 只输出【补全部分】，不重复用户已输入内容（比如用户输入“帮我找黑色的”你得补全“羽绒服、适合秋冬穿”，而不能是“的羽绒服，适合秋冬穿”），补全内容要和用户输入相关，语言要自然
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

    def _is_retryable_exception(self, exc: Exception) -> bool:
        """判断异常是否可重试"""
        try:
            from openai import APIConnectionError, APITimeoutError, RateLimitError, PermissionDeniedError
            retryable_types = (
                asyncio.TimeoutError,
                ConnectionError,
                OSError,
                APIConnectionError,
                APITimeoutError,
                RateLimitError,
                PermissionDeniedError,  # 403 区域限制
            )
        except ImportError:
            retryable_types = (asyncio.TimeoutError, ConnectionError, OSError)
        return isinstance(exc, retryable_types)

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

        # 记忆召回（异步执行，避免阻塞）
        memory_hints = ""
        if request.user_id:
            try:
                loop = asyncio.get_event_loop()
                recalled = await loop.run_in_executor(
                    None,
                    partial(es_memory_client.recall_queries, request.user_id, normalized, 5)
                )
                if recalled:
                    memory_hints = f"\n用户历史搜索偏好: {', '.join(recalled)}"
            except Exception as e:
                logger.warning(f"[补全服务] 记忆召回失败（已降级）: {e}")

        # 缓存
        context = self._build_context(request)
        cache_key = (normalized, context, request.user_id or "")
        async with self._cache_lock:
            if cached := self._cache.get(cache_key):
                return cached

        # 构建 prompt
        prompt = COMPLETION_PROMPT.format(
            context=context + memory_hints,
            max_chars=COMPLETION_MAX_TOTAL_CHARS,
            remain=remain,
            input=normalized,
        )

        # 调用 LLM
        logger.info(f"[补全服务] 开始处理: input=[{normalized}], user_id={request.user_id}")
        try:
            import time
            start_time = time.time()
            llm = await self._get_llm()
            raw_content = ""
            for attempt in range(COMPLETION_RETRY_ATTEMPTS):
                try:
                    resp = await asyncio.wait_for(
                        llm.ainvoke(prompt),
                        timeout=COMPLETION_TIMEOUT_SEC,
                    )
                    raw_content = getattr(resp, "content", "")
                    break
                except Exception as e:
                    if attempt < COMPLETION_RETRY_ATTEMPTS - 1 and self._is_retryable_exception(e):
                        logger.warning(
                            f"[补全服务] 调用失败(可重试) attempt={attempt + 1}: {type(e).__name__}: {e}"
                        )
                        await asyncio.sleep(COMPLETION_RETRY_BACKOFF_SEC)
                        continue
                    raise
            
            elapsed = time.time() - start_time
            result = self._clean_output(raw_content, normalized)
            logger.info(f"[补全服务] 完成: input=[{normalized}], result=[{result}], elapsed={elapsed:.2f}s")
        except asyncio.TimeoutError:
            logger.error(f"[补全服务] LLM 调用超时 (>{COMPLETION_TIMEOUT_SEC}s): input=[{normalized}]")
            result = ""
        except Exception as e:
            logger.error(f"[补全服务] LLM 调用异常: input=[{normalized}], error={e}", exc_info=True)
            result = ""

        # 缓存结果
        if result:
            async with self._cache_lock:
                self._cache[cache_key] = result

        # 异步存储查询
        if result and request.user_id:
            asyncio.create_task(self._store_query_async(request.user_id, normalized, context))

        return result

    async def _store_query_async(self, user_id: str, query: str, context: str):
        """异步存储查询到 ES"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                partial(es_memory_client.store_query, user_id, query, context)
            )
        except Exception as e:
            logger.warning(f"[补全服务] 记忆存储失败（不影响响应）: {e}")


completion_service = CompletionService()

__all__ = ["CompletionService", "completion_service"]
