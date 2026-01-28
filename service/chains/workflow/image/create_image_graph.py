# -*- coding: utf-8 -*-
"""
文生图工作流 - LangGraph 版本

对齐 n8n Greate Image 工作流：
1. 接收 imagePrompt
2. LLM 优化提示词
3. 调用 Gemini 生成图片
4. 上传 OSS
5. 推送 Redis 消息
6. 埋点
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Dict

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from sqlalchemy import text

from app.core.clients.coze_loop_client import coze_loop_client_provider
from app.core.clients.db_client import pg_session
from app.core.clients.redis_client import redis_client
from app.core.config.constants import (
    CozePromptHubKey,
    DBAlias,
    LlmModelName,
    LlmProvider,
    RedisMessageKeyName,
)
from app.core.tools import llm_factory
from app.schemas.entities.message.redis_message import (
    BaseRedisMessage,
    CustomDataContent,
    TextMessageContent,
    WithActionContent,
)
from app.schemas.entities.workflow.graph_state import ImageWorkflowState
from app.schemas.response.workflow_response import WorkflowResponse
from app.service.chains.workflow.base_graph import BaseWorkflowGraph
from app.service.rpc.image_api import get_image_api_client


class CreateImageGraph(BaseWorkflowGraph):
    """文生图工作流 - LangGraph 版本"""

    span_name = "文生图工作流"
    run_name = "create-image-graph"

    def _build_graph(self) -> CompiledStateGraph:
        """构建工作流图"""
        graph = StateGraph(ImageWorkflowState)

        # 添加节点
        graph.add_node("init_state", self._init_state_node)
        graph.add_node("notify_start", self._notify_start_node)
        graph.add_node("optimize_prompt", self._optimize_prompt_node)
        graph.add_node("generate_image", self._generate_image_node)
        graph.add_node("upload_oss", self._upload_oss_node)
        graph.add_node("notify_result", self._notify_result_node)
        graph.add_node("notify_failure", self._notify_failure_node)
        graph.add_node("track_usage", self._track_usage_node)
        graph.add_node("package", self._package_result_node)

        # 设置入口和边
        graph.set_entry_point("init_state")
        graph.add_edge("init_state", "notify_start")
        graph.add_edge("notify_start", "optimize_prompt")
        graph.add_edge("optimize_prompt", "generate_image")

        # 条件分支：生成成功 -> 上传OSS，失败 -> 通知失败
        graph.add_conditional_edges(
            "generate_image",
            self._check_generation_result,
            {
                "success": "upload_oss",
                "failed": "notify_failure",
            },
        )

        graph.add_edge("upload_oss", "notify_result")
        graph.add_edge("notify_result", "track_usage")
        graph.add_edge("notify_failure", "package")
        graph.add_edge("track_usage", "package")
        graph.add_edge("package", END)

        return graph.compile()

    def _check_generation_result(self, state: ImageWorkflowState) -> str:
        """检查图片生成结果"""
        base64_image = state.get("generated_image_base64")
        if base64_image:
            return "success"
        return "failed"

    def _push_message(self, message: BaseRedisMessage, cost_id: str | None = None) -> None:
        payload = message.model_dump(exclude_none=True)
        if cost_id:
            payload["costId"] = cost_id
        redis_client.list_left_push(
            RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
            json.dumps(payload, ensure_ascii=False),
        )

    def _build_message(
        self,
        req: Any,
        operate_id: str,
        status: str,
        content_type: int,
        content: Any,
    ) -> BaseRedisMessage:
        return BaseRedisMessage(
            session_id=req.session_id,
            reply_message_id=req.message_id,
            reply_id=f"reply_{req.message_id}",
            reply_seq=0,
            operate_id=operate_id,
            status=status,
            content_type=content_type,
            content=content,
            create_ts=int(round(time.time() * 1000)),
        )

    # ==================== 节点实现 ====================

    def _init_state_node(self, state: ImageWorkflowState) -> Dict[str, Any]:
        """初始化状态"""
        req = state["request"]

        # 从 request 中提取图片提示词
        image_prompt = req.image_prompt or req.user_query
        cost_id = f"{req.session_id}_{int(time.time() * 1000)}"

        return {
            "image_prompt": image_prompt,
            "cost_id": cost_id,
        }

    def _notify_start_node(self, state: ImageWorkflowState) -> Dict[str, Any]:
        """通知任务开始"""
        req = state["request"]
        cost_id = state.get("cost_id")

        # 推送任务开始消息
        start_message = self._build_message(
            req=req,
            operate_id="收到文生图任务",
            status="RUNNING",
            content_type=2,
            content=WithActionContent(
                text="收到，开始图片设计任务...",
                actions=["images"],
                agent="create_image",
            ),
        )
        self._push_message(start_message, cost_id=cost_id)

        # 推送启用助手消息
        assist_message = self._build_message(
            req=req,
            operate_id="正在启用文生图",
            status="RUNNING",
            content_type=1,
            content=TextMessageContent(text="正在启用图片设计助手"),
        )
        self._push_message(assist_message)

        return {}

    def _optimize_prompt_node(self, state: ImageWorkflowState) -> Dict[str, Any]:
        """LLM 优化提示词 - 使用 CozeLoop 提示词管理"""
        req = state["request"]
        image_prompt = state.get("image_prompt", "")

        # 推送正在生成消息
        generating_message = self._build_message(
            req=req,
            operate_id="正在文生图",
            status="RUNNING",
            content_type=1,
            content=TextMessageContent(text="正在生成图片"),
        )
        self._push_message(generating_message)

        try:
            prompt_params = {"image_prompt": image_prompt}
            messages = coze_loop_client_provider.get_langchain_messages(
                prompt_key=CozePromptHubKey.IMAGE_PROMPT_OPTIMIZE.value,
                variables=prompt_params,
            )

            llm: BaseChatModel = llm_factory.get_llm(
                LlmProvider.HUANXIN.name,
                LlmModelName.HUANXIN_GEMINI_3_FLASH_PREVIEW.value,
            )
            retry_llm = llm.with_retry(stop_after_attempt=2)

            chain = retry_llm | StrOutputParser()
            optimized = chain.with_config(run_name="提示词优化").invoke(messages)
            optimized = optimized.strip().replace("\n\n", "\n")

            if not optimized:
                optimized = image_prompt

            logger.debug(f"[文生图] 优化后提示词: {optimized[:100]}...")
            return {"optimized_prompt": optimized}

        except Exception as e:
            logger.warning(f"[文生图] 提示词优化失败，使用原始提示词: {e}")
            return {"optimized_prompt": image_prompt}

    def _generate_image_node(self, state: ImageWorkflowState) -> Dict[str, Any]:
        """生成图片"""
        optimized_prompt = state.get("optimized_prompt", "")
        image_client = get_image_api_client()
        response = image_client.generate_image_from_text(
            prompt=optimized_prompt,
            model=LlmModelName.OPENROUTER_GEMINI_3_PRO_IMAGE.value,
        )

        if response.success and response.base64_image:
            logger.info(f"[文生图] 图片生成成功")
            return {"generated_image_base64": response.base64_image}
        logger.warning(f"[文生图] 图片生成失败: {response.error_message}")
        return {"generated_image_base64": None}

    def _upload_oss_node(self, state: ImageWorkflowState) -> Dict[str, Any]:
        """上传图片到 OSS"""
        base64_image = state.get("generated_image_base64", "")

        image_client = get_image_api_client()
        response = image_client.upload_to_oss(base64_image)

        if response.success and response.oss_url:
            logger.info(f"[文生图] OSS上传成功: {response.oss_url}")
            return {"output_image_url": response.oss_url}
        else:
            logger.error(f"[文生图] OSS上传失败: {response.error_message}")
            return {"output_image_url": None}

    def _notify_result_node(self, state: ImageWorkflowState) -> Dict[str, Any]:
        """推送结果消息"""
        req = state["request"]
        output_url = state.get("output_image_url")

        if not output_url:
            return {}

        # 推送图片已生成消息
        result_message = self._build_message(
            req=req,
            operate_id="报告文生图结果",
            status="RUNNING",
            content_type=1,
            content=TextMessageContent(text="图片已生成"),
        )
        self._push_message(result_message)

        # 推送输出参数消息 (content_type=5)
        output_message = self._build_message(
            req=req,
            operate_id="文生图",
            status="END",
            content_type=5,
            content=CustomDataContent(
                data={
                    "actions": ["images"],
                    "images": [output_url],
                }
            ),
        )
        self._push_message(output_message)

        return {}

    def _notify_failure_node(self, state: ImageWorkflowState) -> Dict[str, Any]:
        """推送失败消息"""
        req = state["request"]
        cost_id = state.get("cost_id")

        failure_message = self._build_message(
            req=req,
            operate_id="生图任务失败",
            status="RUNNING",
            content_type=10,
            content=WithActionContent(
                text="图片设计任务失败",
                actions=["images"],
                agent="create_image",
            ),
        )
        self._push_message(failure_message, cost_id=cost_id)

        return {}

    def _track_usage_node(self, state: ImageWorkflowState) -> Dict[str, Any]:
        """埋点记录"""
        req = state["request"]
        image_prompt = state.get("image_prompt", "")
        output_url = state.get("output_image_url", "")
        if not output_url:
            return {}

        try:
            resp = requests.get(output_url, timeout=20.0)
            resp.raise_for_status()
            resp.close()
        except Exception as e:
            logger.warning(f"[文生图] 输出图片获取失败: {e}")

        try:
            with pg_session(DBAlias.DB_ABROAD_AI.value) as session:
                session.execute(
                    text("""
                        INSERT INTO zxy_edit_image_user_data_info(
                            session_id, prompt, input_image_link, output_image_link, time
                        )
                        VALUES (:session_id, :prompt, :input_link, :output_link, :time)
                    """),
                    {
                        "session_id": req.session_id,
                        "prompt": image_prompt,
                        "input_link": "文生图",
                        "output_link": output_url,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
        except Exception as e:
            logger.warning(f"[文生图] 埋点失败: {e}")
        return {}

    def _package_result_node(self, state: ImageWorkflowState) -> Dict[str, Any]:
        """封装返回结果"""
        output_url = state.get("output_image_url")

        if output_url:
            response = WorkflowResponse(
                select_result="图片设计任务已完成",
                relate_data=output_url,
            )
        else:
            response = WorkflowResponse(
                select_result="图片生成失败，请稍后重试",
                relate_data=None,
            )
        return {"workflow_response": response}


__all__ = ["CreateImageGraph"]
