from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import cozeloop
from cozeloop import Prompt
from cozeloop.integration.langchain.trace_callback import LoopTracer
from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from app.config import settings


# CozeLoop Role 到 LangChain Message 类的映射
ROLE_TO_MESSAGE_CLASS = {
    "system": SystemMessage,
    "user": HumanMessage,
    "human": HumanMessage,
    "assistant": AIMessage,
    "ai": AIMessage,
}


class CozeLoopClientProvider:
    """CozeLoop 客户端封装，提供提示词管理和 LangChain 集成"""

    def __init__(self) -> None:
        self._client = None

    def get_client(self):
        if self._client is None:
            logger.info(f"开始创建Coze Loop Client, 工作空间id: {settings.cozeloop_workspace_id}")
            self._client = cozeloop.new_client(
                workspace_id=settings.cozeloop_workspace_id,
                api_token=settings.cozeloop_api_token
            )
            logger.info("Coze Loop Client创建完成")
        return self._client

    def start_span(self, name: str, kind: str):
        client = self.get_client()
        return client.start_span(name, kind)

    def create_trace_callback_handler(self, modify_name_fn=None, add_tags_fn=None):
        """
        创建独立的call back handler

        Args:
            modify_name_fn: 修改 span 名称的函数
            add_tags_fn: 添加 span tags 的函数
        """
        return LoopTracer.get_callback_handler(
            self.get_client(), modify_name_fn=modify_name_fn, add_tags_fn=add_tags_fn
        )

    def get_prompt(self, prompt_key: str, label: Optional[str] = None) -> Prompt:
        """
        获取 CozeLoop 提示词

        Args:
            prompt_key: 提示词 key
            label: 版本标签 (production/gray)，为 None 时从 Apollo 配置读取

        Returns:
            Prompt 对象

        Raises:
            Exception: 获取提示词失败时抛出异常（不做降级处理）
        """
        if label is None:
            label = settings.cozeloop_prompt_label

        logger.debug(f"[CozeLoop] 获取提示词: {prompt_key}, label: {label}")
        return self.get_client().get_prompt(prompt_key=prompt_key, label=label)

    def format_prompt(self, prompt: Prompt, variables: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        使用 CozeLoop 内置方法格式化提示词。

        Args:
            prompt: CozeLoop Prompt 对象
            variables: 变量字典

        Returns:
            格式化后的消息列表，每个元素包含 role 和 content
        """
        return self.get_client().prompt_format(prompt, variables)

    def close(self) -> None:
        logger.info("关闭Coze Loop客户端...")
        try:
            self.get_client().close()
        except Exception:
            pass
        logger.info("Coze Loop客户端已关闭")

    # ==================== LangChain 集成方法（简化版） ====================

    def get_langchain_messages(
        self,
        prompt_key: str,
        variables: Dict[str, Any],
        label: Optional[str] = None,
    ) -> List[BaseMessage]:
        """
        获取格式化后的 LangChain 消息列表。

        流程：
        1. 从 CozeLoop 获取提示词（带 label）
        2. 使用 CozeLoop 的 prompt_format 填充变量
        3. 转换为 LangChain 消息列表

        Args:
            prompt_key: CozeLoop 中的提示词 key
            variables: 变量字典
            label: 版本标签 (production/gray)，为 None 时从 Apollo 配置读取

        Returns:
            LangChain 消息列表，可直接传给 LLM

        Raises:
            Exception: 获取提示词失败时抛出异常（不做降级处理）

        Example:
            >>> messages = provider.get_langchain_messages("my_prompt", {"query": "hello"})
            >>> result = llm.invoke(messages)
        """
        # 1. 获取提示词（自动从 Apollo 读取 label）
        prompt = self.get_prompt(prompt_key, label=label)
        logger.debug(f"[CozeLoop] 获取提示词: {prompt_key}")

        # 2. 使用 CozeLoop 格式化
        formatted = self.format_prompt(prompt, variables)
        logger.debug(f"[CozeLoop] 格式化后消息数量: {len(formatted)}")

        # 3. 转换为 LangChain 消息
        return self._convert_to_langchain_messages(formatted)

    def _convert_to_langchain_messages(
        self,
        formatted_messages: List[Dict[str, Any]],
    ) -> List[BaseMessage]:
        """
        将 CozeLoop 格式化后的消息转换为 LangChain 消息列表。

        Args:
            formatted_messages: CozeLoop prompt_format 返回的消息列表

        Returns:
            LangChain 消息列表
        """
        messages = []
        for msg in formatted_messages:
            role = self._extract_role(msg)
            content = self._extract_content(msg)

            message_class = ROLE_TO_MESSAGE_CLASS.get(role.lower(), HumanMessage)
            messages.append(message_class(content=content))

        return messages

    def _extract_role(self, msg: Union[Dict, Any]) -> str:
        """从消息中提取 role"""
        if isinstance(msg, dict):
            return msg.get("role", "user")
        if hasattr(msg, "role"):
            role = msg.role
            return role.value if hasattr(role, "value") else str(role)
        return "user"

    def _extract_content(self, msg: Union[Dict, Any]) -> str:
        """从消息中提取 content"""
        if isinstance(msg, dict):
            return msg.get("content", "")
        if hasattr(msg, "content") and msg.content:
            return msg.content
        if hasattr(msg, "parts") and msg.parts:
            parts = []
            for part in msg.parts:
                if hasattr(part, "text") and part.text:
                    parts.append(part.text)
                elif hasattr(part, "content") and part.content:
                    parts.append(part.content)
            return "\n".join(parts)
        return ""


coze_loop_client_provider = CozeLoopClientProvider()


__all__ = [
    "CozeLoopClientProvider",
    "coze_loop_client_provider",
]