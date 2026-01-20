# @Author   : kiro
# @Time     : 2025/12/14
# @File     : base_graph.py

"""
LangGraph 工作流基类

提供统一的图构建、执行和导出接口。
"""

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from app.config import settings
from app.core.clients.coze_loop_client import coze_loop_client_provider
from app.core.config.constants import LlmModelName, LlmProvider, VolcKnowledgeServiceId
from app.core.errors import AppException, ErrorCode
from app.core.tools import llm_factory
from app.schemas.request.workflow_request import WorkflowRequest
from app.service.chains.workflow.progress_pusher import ControlMessagePusher

StateType = TypeVar("StateType", bound=dict[str, Any])


class BaseWorkflowGraph(ABC):
    """LangGraph 工作流基类"""

    span_name: str = "workflow_graph"

    def __init__(self) -> None:
        self._compiled_graph: CompiledStateGraph = self._build_graph()

    @abstractmethod
    def _build_graph(self) -> CompiledStateGraph:
        """构建并编译图，子类必须实现"""
        pass

    def get_graph(self) -> CompiledStateGraph:
        """获取编译后的图"""
        return self._compiled_graph

    # 工作流全局超时时间（秒）
    workflow_timeout: int = 300

    def run(self, req: WorkflowRequest) -> dict[str, Any]:
        """执行工作流，带 CozeLoop 追踪和超时控制

        采用同步执行模式，符合项目整体架构。
        工作流内部节点可以根据需要使用线程池处理 I/O 操作。

        Returns:
            Dict[str, Any]: 完整的工作流 state，包含 workflow_response 等
        """
        from app.schemas.response.workflow_response import WorkflowResponse

        client = coze_loop_client_provider.get_client()
        try:
            with client.start_span(
                self.span_name, "workflow"
            ) as root_span:
                try:
                    root_span.set_deployment_env(settings.environment)
                    root_span.set_user_id_baggage(str(req.user_id))
                    root_span.set_message_id_baggage(req.message_id)
                    root_span.set_thread_id_baggage(req.session_id)
                    root_span.set_input(req)
                    # 设置服务名称，确保 UI 显示正确
                    root_span.set_service_name(self.span_name)

                    # 记录开始时间，用于超时检查
                    start_time = time.time()

                    run_name = getattr(self, "run_name", None) or self.span_name

                    # 使用子类提供的名称修改函数，或默认函数
                    if hasattr(self, "_get_trace_name_modifier"):
                        name_modifier = self._get_trace_name_modifier()
                    else:
                        def name_modifier(node_name: str) -> str:
                            if node_name == "init_state":
                                return self.span_name
                            return node_name

                    cozeloop_callback_handler = coze_loop_client_provider.create_trace_callback_handler(
                        modify_name_fn=name_modifier
                    )
                    state = self._compiled_graph.invoke(
                        {"request": req},
                        RunnableConfig(callbacks=[cozeloop_callback_handler], run_name=run_name)
                    )  # type: ignore[assignment]

                    # 检查是否超时（虽然已经执行完，但用于告警）
                    elapsed_time = time.time() - start_time
                    if elapsed_time > self.workflow_timeout:
                        logger.warning(
                            f"工作流执行耗时过长: {elapsed_time:.2f}秒 "
                            f"(超时阈值: {self.workflow_timeout}秒) "
                            f"[trace_id={req.session_id}_{req.message_id}]"
                        )

                    # 返回完整 state，保留所有上下文信息
                    # 确保 workflow_response 存在
                    if "workflow_response" not in state:
                        state["workflow_response"] = WorkflowResponse(
                            select_result="工作流执行完成", relate_data=None
                        )

                    # 将 WorkflowResponse 转换为字典，避免被 CozeLoop callback 序列化成字符串
                    if isinstance(state.get("workflow_response"), WorkflowResponse):
                        state["workflow_response"] = state["workflow_response"].model_dump()

                    logger.info(
                        f"工作流执行成功: 耗时 {elapsed_time:.2f}秒 "
                        f"[trace_id={req.session_id}_{req.message_id}]"
                    )

                    return state

                except AppException as e:
                    root_span.set_error(e)
                    # 发生执行异常则进行费用的回退，通过消息让后端回退费用
                    message_pusher = ControlMessagePusher(req)
                    message_pusher.push_cost_refund_message()
                    raise
                except Exception as e:
                    root_span.set_error(e)
                    # 发生执行异常则进行费用的回退，通过消息让后端回退费用
                    message_pusher = ControlMessagePusher(req)
                    message_pusher.push_cost_refund_message()

                    trace_id = f"{req.session_id}_{req.message_id}"
                    elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
                    logger.exception(
                        f"工作流执行失败 [trace_id={trace_id}]: {str(e)} "
                        f"(已耗时: {elapsed_time:.2f}秒)"
                    )
                    raise AppException(
                        ErrorCode.WORKFLOW_ERROR,
                        str(e),
                        http_status=500,
                        details={"stage": "workflow"},
                    ) from e
        finally:
            # 确保 CozeLoop 追踪数据被发送
            try:
                client.flush()
            except Exception:
                pass

    def export_mermaid(self) -> str:
        """导出为 Mermaid 图格式"""
        return self._compiled_graph.get_graph().draw_mermaid()

    def export_json(self) -> dict[str, Any]:
        """导出为 JSON 结构（用于前端渲染）"""
        graph = self._compiled_graph.get_graph()
        nodes: list[dict[str, str]] = []
        edges: list[dict[str, str]] = []

        for node_name in graph.nodes:
            nodes.append({"id": node_name, "label": node_name})

        for edge in graph.edges:
            edges.append({"source": edge[0], "target": edge[1]})

        return {"nodes": nodes, "edges": edges}

    # ==================== 类目向量检索通用方法 ====================

    def _fetch_category_vector(
        self,
        query: str,
        service_id: VolcKnowledgeServiceId,
        prompt_key: str | None = None,
    ) -> list[str]:
        """类目向量检索 - 通用实现

        Args:
            query: 用户查询文本
            service_id: 火山知识库服务 ID 枚举值
            prompt_key: Coze 提示词 key，用于清洗召回文本

        Returns:
            清洗后的类目标签列表，格式为 ["key#value", ...]
        """
        if not service_id or not service_id.value:
            return []

        try:
            from app.service.rpc.volcengine_kb_api import KBMessage, get_volcengine_kb_api

            kb_client = get_volcengine_kb_api()
            messages = [KBMessage(role="user", content=query)]
            response = kb_client.chat(messages=messages, service_resource_id=service_id.value)

            if not response.data or not response.data.result_list:
                return []

            # 提取检索结果 - 对齐 n8n 提取检索结果2 节点
            content_list = []
            for item in response.data.result_list:
                if hasattr(item, "table_chunk_fields") and item.table_chunk_fields:
                    key, value = "", ""
                    for chunk in item.table_chunk_fields:
                        field_name = (
                            chunk.get("field_name", "")
                            if isinstance(chunk, dict)
                            else getattr(chunk, "field_name", "")
                        )
                        field_value = (
                            chunk.get("field_value", "")
                            if isinstance(chunk, dict)
                            else getattr(chunk, "field_value", "")
                        )
                        if field_name == "key":
                            key = field_value
                        elif field_name == "value":
                            value = field_value
                    if key or value:
                        content_list.append(f"{key}#{value}")
                elif hasattr(item, "content") and item.content:
                    content_list.append(item.content)

            if not content_list:
                return []

            return content_list
        except Exception as e:
            logger.warning(f"[类目向量检索] 失败: {e}")
            return []


__all__ = ["BaseWorkflowGraph"]
