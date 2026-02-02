from __future__ import annotations

import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Literal, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger
from sqlalchemy import text

from app.config import settings
from app.core.clients.db_client import pg_session
from app.core.clients.redis_client import redis_client
from app.core.config.constants import (
    DBAlias,
    LlmModelName,
    LlmProvider,
    RedisMessageKeyName,
)
from app.core.tools import llm_factory
from app.schemas.entities.message.redis_message import (
    BaseRedisMessage,
    TextMessageContent,
)
from app.schemas.entities.workflow.graph_state import MainOrchestratorState
from app.schemas.request.workflow_request import WorkflowRequest
from app.schemas.response.workflow_response import WorkflowResponse
from app.service.chains.workflow.base_graph import BaseWorkflowGraph
from app.service.chains.workflow.orchestrator_tools import ALL_TOOLS
from app.service.chains.workflow.artifact_store import get_artifact_store
from app.service.rpc.vlm_service import get_vlm_service
from app.utils import thread_pool


class MainOrchestratorGraph(BaseWorkflowGraph):
    """
    主agent - 标准 LangGraph ReAct

    图结构 (3 节点):

        START ──→ agent ←──→ tools ──→ postprocess ──→ END
                    ↑         │
                    └─────────┘
                   (ReAct loop)
    """

    span_name = "zxy_agent_system"
    run_name = "zxy_agent_system"
    _max_agent_iterations = 6

    def __init__(self) -> None:
        super().__init__()

    def _build_graph(self) -> CompiledStateGraph:
        """构建 LangGraph 执行图（标准 ReAct 循环）"""
        graph = StateGraph(MainOrchestratorState)

        # Agent 节点：LLM 推理与决策
        graph.add_node("agent", self._agent_node)

        # Tools 节点：直接使用 ToolNode（符合 LangGraph 规范，Graph 会自动提供 runtime）
        graph.add_node("tools", ToolNode(ALL_TOOLS))

        # Postprocess 节点：提取 artifacts、推送结果、保存回复
        graph.add_node("postprocess", self._postprocess_node)

        # 定义边
        graph.add_edge(START, "agent")  # 起点

        # 条件边：由 _should_continue 判断是否继续调用工具
        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {"continue": "tools", "end": "postprocess"},
        )

        graph.add_edge("tools", "agent")  # ReAct 循环：工具执行后返回 agent 继续推理
        graph.add_edge("postprocess", END)  # 终点

        return graph.compile()

    # ==================== Agent 节点 ====================

    def _agent_node(self, state: MainOrchestratorState) -> Dict[str, Any]:
        """主 Agent 节点：LLM 推理 + 工具调用决策（ReAct 范式：无硬编码规则，完全由 LLM 自主决策）"""
        messages = state.get("messages") or []

        if not messages:
            # 首次调用：构建初始消息并写入 state
            messages, upload_image_ids = self._prepare_initial_messages(state)
            ai_message = self._invoke_llm_with_fallback(messages, state)
            self._log_response(ai_message)
            return {"messages": messages + [ai_message], "upload_image_ids": upload_image_ids}

        # 后续调用：直接使用 state 中的消息
        ai_message = self._invoke_llm_with_fallback(messages, state)
        self._log_response(ai_message)
        return {"messages": [ai_message]}

    def _log_response(self, ai_message: AIMessage) -> None:
        """记录 LLM 响应摘要"""
        content = getattr(ai_message, "content", "")
        tool_calls = getattr(ai_message, "tool_calls", None)
        if tool_calls:
            tool_names = [tc.get("name") for tc in tool_calls]
            logger.info(f"[Agent] 工具调用: {tool_names}")
        elif content:
            logger.info(f"[Agent] 回复: {content[:100]}...")

    def _invoke_llm_with_fallback(
        self,
        messages: List,
        state: MainOrchestratorState,
    ) -> AIMessage:
        """调用 LLM，带重试和降级策略"""
        try:
            # 使用 Kimi K2 Thinking（支持 extended thinking）
            from app.core.tools.llm_factory import LLMFactory
            llm: BaseChatModel = LLMFactory.create_openrouter_llm(
                model="moonshotai/kimi-k2-thinking",
                max_tokens=16384,
            )
        except Exception as e:
            logger.warning(f"[Agent] LLM initialization failed: {e}")
            return self._fallback_response(state, e)

        # 调用 LLM，带指数退避重试（最多 2 次）
        ai_message = self._invoke_with_retry(llm, messages, max_retries=2)
        if ai_message:
            return ai_message

        # 所有重试都失败 → 降级（友好错误提示，不用硬编码规则替代）
        return self._fallback_response(state, Exception("LLM retry exhausted"))

    def _fallback_response(
        self,
        state: MainOrchestratorState,
        error: Exception | None = None,
    ) -> AIMessage:
        """降级响应：仅在 LLM 完全失败时使用（只做友好错误处理，不用规则替代 LLM 决策）"""
        if error:
            logger.error(f"[Fallback] LLM failed: {error}")

        # 符合 ReAct 范式的降级策略：诚实告知用户，而不是用规则系统伪装成智能决策
        return AIMessage(
            content="抱歉，我现在遇到了一些技术问题，无法处理您的请求。请稍后再试，或者换一种方式描述您的需求。"
        )

    def _invoke_with_retry(
        self, llm: BaseChatModel, messages: List, max_retries: int = 2
    ) -> Optional[AIMessage]:
        """带指数退避和错误分类的 LLM 调用"""
        for attempt in range(max_retries):
            try:
                # bind_tools 让 LLM 知道可用的工具及其参数（LLM 自主决定调用哪些，无硬编码规则）
                result = llm.bind_tools(ALL_TOOLS).invoke(messages)
                if not isinstance(result, AIMessage):
                    result = AIMessage(content=str(getattr(result, "content", result)))
                return result
            except Exception as e:
                error_msg = str(e).lower()

                if "rate" in error_msg or "limit" in error_msg or "429" in error_msg:
                    # 限流错误：指数退避（1s, 2s）
                    logger.warning(f"[Agent] 限流错误(尝试 {attempt + 1}): {e}")
                    wait_time = (2**attempt) * 1
                    time.sleep(wait_time)
                elif "content" in error_msg and "filter" in error_msg:
                    # 内容审核错误：立即返回，不重试
                    logger.warning(f"[Agent] 内容审核拒绝: {e}")
                    return AIMessage(content="抱歉，您的请求包含不支持的内容，请调整后重试。")
                else:
                    # 其他错误：线性等待 1s 后重试
                    logger.warning(f"[Agent] 其他错误(尝试 {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
        return None

    def _prepare_initial_messages(self, state: MainOrchestratorState) -> tuple[List, List[str]]:
        """准备初始消息（首次进入 agent 时调用）"""
        req: WorkflowRequest = state["request"]

        # 异步记录用户查询到数据库（不阻塞主流程）
        self._record_user_query(req)

        # 处理上传图片：并行调用 VLM → 创建 Artifact → 生成描述文本
        # ⚠️ 唯一调用 VLM 的地方，后续通过 inspect_artifact 直接返回 content_cache
        upload_image_ids, image_content = self._extract_images_as_artifacts(req)

        # 查询会话历史（限制最近 20 条，避免 context 过长）
        history_messages = self._query_history(req)

        # 构建完整消息列表（System Prompt + History + Current）
        messages = self._build_messages(req, image_content, history_messages, upload_image_ids)
        return messages, upload_image_ids

    def _record_user_query(self, req: WorkflowRequest) -> None:
        """记录用户查询到数据库"""

        def insert_track() -> None:
            try:
                with pg_session(DBAlias.DB_ABROAD_AI.value) as session:
                    session.execute(
                        text(
                            """
                            INSERT INTO n8n_user_query_message(session_id, user_query)
                            VALUES (:session_id, :user_query)
                        """
                        ),
                        {"session_id": req.session_id, "user_query": f"human:{req.user_query}"},
                    )
            except Exception as e:
                logger.error(f"[主工作流] 记录用户查询失败: {e}")

        thread_pool.submit_with_context(insert_track)

    def _extract_images_as_artifacts(self, req: WorkflowRequest) -> tuple[List[str], Optional[str]]:
        """提取用户上传图片，并行调用 VLM 生成描述，创建 Artifact（⚠️ VLM 唯一调用点）"""
        images = getattr(req, "images", None) or []
        if not images:
            return [], None

        artifact_store = get_artifact_store()
        artifacts: dict[int, str] = {}  # idx -> artifact_id
        captions: dict[int, str] = {}   # idx -> caption

        max_workers = max(1, min(len(images), settings.main_workflow_image_max_workers))

        # 并行处理图片（VLM 调用耗时，必须并行）
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self._extract_single_image, url): idx
                for idx, url in enumerate(images)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                caption = future.result()
                if caption:
                    artifact_id = f"img_upload_{int(time.time() * 1000)}_{idx}"
                    record = {
                        "id": artifact_id,
                        "type": "image_asset",
                        "description": f"用户上传的图片 {idx + 1}",
                        "content_cache": caption,
                        "meta": {"source": "user_upload", "index": idx + 1, "url": images[idx]},
                    }
                    artifact_store.save_payload(req.session_id, artifact_id, record)
                    artifacts[idx] = artifact_id
                    captions[idx] = caption

        # 生成格式化描述文本（按原始索引排序，确保 artifact_id 与 caption 正确对应）
        if captions:
            sorted_indices = sorted(captions.keys())
            lines = [
                f"[Image {i + 1}] (artifact: {artifacts[i]}): {captions[i]}"
                for i in sorted_indices
            ]
            artifact_ids = [artifacts[i] for i in sorted_indices]
            return artifact_ids, "\n".join(lines)

        return [], None

    def _extract_single_image(self, image_url: str) -> Optional[str]:
        """提取单张图片的描述（使用 VLMService）"""
        max_retries = settings.main_workflow_image_retry_attempts
        vlm_service = get_vlm_service()

        for attempt in range(max_retries):
            try:
                caption = vlm_service.describe(image_url)
                return caption
            except Exception as e:
                logger.warning(f"[图片提取] 第{attempt + 1}次尝试失败: {e}")
                if attempt == max_retries - 1:
                    # 降级：返回默认描述
                    return "用户上传的图片"
        return "用户上传的图片"

    def _query_history(self, req: WorkflowRequest, limit: int = 20) -> List:
        """查询会话历史并转换为 LangGraph Messages（限制最近 N 条）

        Args:
            req: 请求对象
            limit: 最多查询多少条历史记录

        Returns:
            List[HumanMessage | AIMessage]: 历史消息列表，按时间正序排列
        """
        history_messages: List = []
        try:
            with pg_session(DBAlias.DB_ABROAD_AI.value) as session:
                result = session.execute(
                    text(
                        """
                        SELECT id, session_id, user_query
                        FROM n8n_user_query_message
                        WHERE session_id = :session_id
                        ORDER BY id DESC
                        LIMIT :limit
                    """
                    ),
                    {"session_id": req.session_id, "limit": limit},
                )
                rows = list(result.mappings().all())

                # 反转顺序，保持时间正序
                for row in reversed(rows):
                    query_text = row["user_query"]

                    # 根据前缀判断消息类型
                    if query_text.startswith("human:"):
                        content = query_text[6:].strip()  # 去掉 "human:" 前缀
                        history_messages.append(HumanMessage(content=content))
                    elif query_text.startswith("ai:"):
                        content = query_text[3:].strip()  # 去掉 "ai:" 前缀
                        history_messages.append(AIMessage(content=content))
                    else:
                        # 兼容旧数据：无前缀时默认为用户消息
                        logger.warning(f"[会话历史] 消息 {row['id']} 缺少前缀，默认为用户消息")
                        history_messages.append(HumanMessage(content=query_text))

                logger.info(f"[会话历史] 加载了 {len(history_messages)} 条历史消息")

        except Exception as e:
            logger.warning(f"[会话历史] 查询失败: {e}")

        return history_messages

    def _build_messages(
        self,
        req: WorkflowRequest,
        image_content: Optional[str],
        history_messages: List,
        upload_image_ids: Optional[List[str]] = None,
    ) -> List:
        """构建初始消息：System + History + Current

        Args:
            req: 请求对象
            image_content: 图片描述文本
            history_messages: 历史消息列表（HumanMessage/AIMessage）
            upload_image_ids: 上传图片的 artifact IDs

        Returns:
            List: 完整的消息列表
        """
        tool_names = ", ".join([tool.name for tool in ALL_TOOLS])

        system_prompt = f"""你是知小衣，一个由知衣科技开发的智能服装选品与设计助手。

## 核心能力

1. **商品搜索**：淘宝（select_zhiyi）、抖音（select_douyi）、海外平台
2. **设计生成**：AI 生图（create_image）、AI 改图（edit_image）
3. **内容搜索**：INS 博主、小红书内容
4. **任务管理**：定时监控任务

## 工作流程

1. **理解需求**：分析用户想要什么，识别关键信息（平台、类目、风格等）
2. **选择工具**：根据需求决定调用哪些工具，可并行调用多个
3. **执行并返回**：调用工具获取结果，给出简洁清晰的回复

## 资产系统（内部使用，不要向用户展示）

工具返回的数据以「资产」形式存储，系统会自动记录。
- 用户提到"上一轮"/"刚才"的结果时，从历史消息中找到对应的 artifact_id
- 需要查看资产详情时，调用 `inspect_artifact(artifact_id)`
- ⚠️ 不要向用户展示 artifact_id，用户只需要知道"找到了多少件"

## 可用工具

{tool_names}

## 回复要求

1. 只输出给用户看的内容，不要输出内部思考过程
2. 不要展示 artifact_id 给用户
3. 回复简洁自然，像正常对话一样

示例：
- ❌ "资产ID: select_zhiyi_xxx"
- ✅ "已为您在淘宝找到10件T恤，需要看具体款式吗？"
"""

        images = getattr(req, "images", []) or []
        image_url = getattr(req, "image_url", None)
        input_images = getattr(req, "input_images", None)
        image_ref = "（[Image N] 对应下方链接第 N 张）" if images else ""
        query_references = getattr(req, "query_references", None) or []
        if query_references:
            query_references_payload = [
                ref.model_dump() if hasattr(ref, "model_dump") else ref for ref in query_references
            ]
            query_references_text = json.dumps(query_references_payload, ensure_ascii=False)
        else:
            query_references_text = "(无)"

        # 当前轮次的用户请求
        user_prompt = (
            f"### 用户请求\n{req.user_query}\n\n"
            f"### 系统字段\n"
            f"- team_id: {req.team_id}\n"
            f"- user_id: {req.user_id}\n"
            f"- session_id: {req.session_id}\n"
            f"- message_id: {req.message_id}\n\n"
            f"### 业务上下文\n"
            f"- 平台偏好: {getattr(req, 'preferred_entity', '无')}\n"
            f"- 行业: {getattr(req, 'industry', '无')}\n"
            f"- 用户偏好: {getattr(req, 'user_preferences', '无')}\n"
            f"- abroad_type: {getattr(req, 'abroad_type', '无')}\n"
            f"- 是否参考监控数据: {getattr(req, 'is_monitored', '无')}\n"
            f"- 是否参考用户画像: {getattr(req, 'is_user_preferences', '无')}\n\n"
            f"### 关联实体\n{query_references_text}\n\n"
            f"### 视觉输入 {image_ref}\n{image_content or '(无)'}\n"
            f"- 图片链接: {images if images else '无'}\n"
            f"- image_url: {image_url or '无'}\n"
            f"- input_images: {input_images or '无'}\n"
        )

        # 组装消息：System + History + Current
        messages = [SystemMessage(content=system_prompt)]

        # 添加历史消息（如果有）
        if history_messages:
            messages.extend(history_messages)
            logger.info(f"[消息构建] 已添加 {len(history_messages)} 条历史消息")

        # 添加当前请求
        messages.append(HumanMessage(content=user_prompt))

        return messages

    def _should_continue(self, state: MainOrchestratorState) -> Literal["continue", "end"]:
        """判断是否继续 ReAct 循环（条件边逻辑）"""
        messages = state.get("messages") or []

        # 安全检查：防止无限循环
        tool_call_turns = sum(
            1 for msg in messages if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None)
        )

        if tool_call_turns >= self._max_agent_iterations:
            logger.warning(f"[Agent] 已达最大迭代上限 {self._max_agent_iterations} 次")
            return "end"

        # 检查最后一条消息是否需要调用工具
        last_message = messages[-1] if messages else None
        if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
            return "continue"  # LLM 决定调用工具 → 继续 ReAct 循环

        return "end"  # LLM 给出最终回复 → 结束循环

    # ==================== 后置处理节点 ====================

    def _postprocess_node(self, state: MainOrchestratorState) -> Dict[str, Any]:
        """后置处理节点：提取 artifacts + 推送结果 + 保存回复"""
        req = state["request"]
        messages = state.get("messages") or []

        # 1. 从 ToolMessage 中提取 artifact_id 和元数据（实际数据在 Redis，State 只保存引用）
        new_artifacts = {}
        for message in messages:
            if isinstance(message, ToolMessage):
                try:
                    content = json.loads(message.content)
                    if isinstance(content, dict) and content.get("status") == "success":
                        data = content.get("data")
                        if data and isinstance(data, dict) and "artifact_id" in data:
                            artifact_id = data["artifact_id"]
                            new_artifacts[artifact_id] = {
                                "type": data.get("type"),
                                "description": data.get("description"),
                                "meta": data.get("meta"),
                            }
                except Exception:
                    continue

        # 2. 从最后一个纯文本 AIMessage 中提取最终回复
        summary_text = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                summary_text = msg.content
                break

        if not summary_text:
            summary_text = "抱歉，处理过程中出现问题，请稍后重试。"

        # 2.1 构建资产摘要（附加到保存的历史中，方便多轮对话引用）
        artifact_summary = ""
        if new_artifacts:
            artifact_lines = [
                f"- {aid}: {info.get('description', '无描述')}"
                for aid, info in new_artifacts.items()
            ]
            artifact_summary = "\n\n📦 本轮生成的资产:\n" + "\n".join(artifact_lines)

        # 3. 推送结果到 Redis 队列（供前端实时展示，内部调用时跳过）
        if not getattr(req, "suppress_messages", False):
            finish_message = BaseRedisMessage(
                session_id=req.session_id,
                reply_message_id=req.message_id,
                reply_id=f"reply_{req.message_id}",
                reply_seq=0,
                operate_id="结果",
                status="END",
                content_type=1,
                content=TextMessageContent(text=summary_text),
                create_ts=int(round(time.time() * 1000)),
            )
            redis_client.list_left_push(
                RedisMessageKeyName.AI_CONVERSATION_MESSAGE_QUEUE.value,
                finish_message.model_dump_json(),
            )

        # 4. 异步保存回复到数据库（包含资产摘要，方便多轮对话引用）
        self._save_ai_response(req, summary_text + artifact_summary)

        # 5. 封装返回结果
        response = WorkflowResponse(select_result=summary_text, relate_data=None)
        result = {"summary_text": summary_text, "workflow_response": response}

        # 更新 State 中的 artifacts
        if new_artifacts:
            current_artifacts = state.get("artifacts") or {}
            current_artifacts.update(new_artifacts)
            result["artifacts"] = current_artifacts

        return result

    def _save_ai_response(self, req: WorkflowRequest, summary_text: str) -> None:
        """保存 AI 回复"""

        def insert_response() -> None:
            try:
                with pg_session(DBAlias.DB_ABROAD_AI.value) as session:
                    session.execute(
                        text(
                            """
                            INSERT INTO n8n_user_query_message(session_id, user_query)
                            VALUES (:session_id, :user_query)
                        """
                        ),
                        {"session_id": req.session_id, "user_query": f"ai:{summary_text}"},
                    )
            except Exception as e:
                logger.error(f"[主工作流] 保存AI回复失败: {e}")

        thread_pool.submit_with_context(insert_response)


__all__ = ["MainOrchestratorGraph"]


if __name__ == "__main__":
    """
    交互式终端对话模式

    使用方法:
        python -m app.service.chains.workflow.main_orchestrator_graph

    输入 'quit' 或 'exit' 退出
    """
    import uuid
    import sys

    print("=" * 60)
    print("知小衣 AI 助手 - 交互式对话模式")
    print("=" * 60)
    print("输入您的问题，输入 'quit' 或 'exit' 退出\n")

    # 初始化 graph
    graph = MainOrchestratorGraph()

    # 生成会话 ID（整个对话会话共享）
    session_id = f"cli_{uuid.uuid4().hex[:8]}"
    print(f"会话 ID: {session_id}\n")

    try:
        while True:
            # 读取用户输入
            try:
                user_input = input("您: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n再见！")
                sys.exit(0)

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                print("\n再见！")
                break

            # 创建请求
            request = WorkflowRequest(
                team_id=1,
                user_id=1,
                session_id=session_id,
                message_id=f"msg_{uuid.uuid4().hex[:8]}",
                user_query=user_input,
            )

            # 运行工作流
            print("\n[AI 思考中...]")
            try:
                state = graph.run(request)
                # 打印 AI 思考过程（从 messages 中提取）
                messages = state.get("messages") or []
                for i, msg in enumerate(messages):
                    if isinstance(msg, AIMessage):
                        content = (getattr(msg, "content", None) or "").strip()
                        if content:
                            print(f"\n[思考] {content}")
                        tool_calls = getattr(msg, "tool_calls", None)
                        if tool_calls:
                            for tc in tool_calls:
                                name = tc.get("name", "?")
                                args = tc.get("args") or {}
                                print(f"[调用工具] {name} 参数: {args}")
                    elif isinstance(msg, ToolMessage):
                        try:
                            content = (
                                json.loads(msg.content)
                                if isinstance(msg.content, str)
                                else msg.content
                            )
                            status = content.get("status", "?")
                            brief = content.get("message", str(content)[:80])
                            print(f"[工具结果] {msg.name} -> {status}: {brief}")
                        except Exception:
                            print(f"[工具结果] {msg.name} -> (见上方日志)")

                # 提取回复
                summary_text = state.get("summary_text")
                if not summary_text:
                    # 从 messages 中提取最后一条 AI 消息
                    messages = state.get("messages") or []
                    for msg in reversed(messages):
                        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                            summary_text = msg.content
                            break

                if summary_text:
                    print(f"\n知小衣: {summary_text}\n")
                else:
                    print("\n知小衣: 抱歉，未能生成回复。\n")

            except Exception as e:
                logger.exception(f"运行工作流时出错: {e}")
                print(f"\n[错误] {e}\n")

    except KeyboardInterrupt:
        print("\n\n再见！")
        sys.exit(0)
