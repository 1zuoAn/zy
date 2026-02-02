import json
import os
import asyncio
from typing import Optional, List, Dict, Any, AsyncGenerator
from uuid import uuid4
from datetime import datetime
from contextlib import asynccontextmanager

from pydantic import ValidationError
from openai import AsyncOpenAI

# ReMe & FlowLLM SDK
from reme_ai import ReMeApp
from flowllm.core.embedding_model.openai_compatible_embedding_model import OpenAICompatibleEmbeddingModel
from flowllm.core.vector_store.es_vector_store import EsVectorStore
from flowllm.core.schema import VectorNode

# 引入同级配置和Schema
from . import config, prompt_template
from .patches import apply_monkey_patches
from app.schemas.request.intent_request import (
    ClassifyRequest, FeedbackRequest, MemoryMaintainRequest, IntentEnum
)
from app.schemas.response.intent_response import (
    ClassifyResult, ClassifyResponse, CLASSIFY_JSON_SCHEMA
)
from ...config import settings
from ...core.config.constants import LlmModelName, EmbeddingModelName

# 1. 应用 Monkey Patches (最优先执行)
apply_monkey_patches()

class IntentService:
    def __init__(self):
        # 全局单例状态
        self.reme_app: Optional[ReMeApp] = None
        self.openai_client: Optional[AsyncOpenAI] = None
        self.maintenance_vs: Optional[EsVectorStore] = None
        self._is_ready = False

        # 模型回退链配置
        self.MODEL_FALLBACK_CHAIN = [
            LlmModelName.OPENROUTER_GEMINI_3_FLASH_PREVIEW.value,
            LlmModelName.OPENROUTER_GEMINI_2_5_FLASH.value,
            LlmModelName.OPENROUTER_GPT_4O.value
        ]

    # ===========================================================================
    # 1. 生命周期管理 (Startup / Shutdown)
    # ===========================================================================
    
    async def startup(self):
        """对应原 main.py 的 lifespan startup 部分"""
        print("🚀 [IntentService] Initializing...")

        # --- A. 初始化 OpenAI Client ---
        self.openai_client = AsyncOpenAI(
            base_url=settings.openrouter_api_base,
            api_key=settings.openrouter_api_key,
        )

        # --- B. 初始化维护用向量库 (Maintenance VS) ---
        print("🔧 [IntentService] Initializing Standalone Maintenance Vector Store...")
        
        # 独立初始化 Embedding 模型
        maintenance_embedding = OpenAICompatibleEmbeddingModel(
            model_name=EmbeddingModelName.DASHSCOPE_TEXT_EMBEDDING_V4.value,
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_api_base,
        )

        # 独立初始化 ES 连接
        try:
            # 简单处理 ES_HOST (移除 http:// 前缀如果存在，因为 EsVectorStore 可能自动加)
            # 这里按照你原始代码逻辑保持一致
            hosts = [f"http://{settings.es_host}"]
            
            self.maintenance_vs = EsVectorStore(
                hosts=hosts,
                basic_auth=(settings.es_user, settings.es_password),
                embedding_model=maintenance_embedding
            )
            print("✅ [IntentService] Maintenance Connection Ready.")
        except Exception as e:
            print(f"⚠️ [IntentService] Maintenance VS Init Failed: {e}")

        # --- C. 初始化 ReMe App ---
        print(f"🚀 [IntentService] Initializing ReMe (Backend: Elasticsearch)...")

        # 注入 ReMe 所需的环境变量
        if settings.openrouter_api_key:
            os.environ["FLOW_LLM_API_KEY"] = settings.openrouter_api_key
        os.environ["FLOW_LLM_BASE_URL"] = settings.openrouter_api_base

        if settings.dashscope_api_key:
            os.environ["FLOW_EMBEDDING_API_KEY"] = settings.dashscope_api_key
        else:
            print("⚠️ 警告: 未检测到 DASHSCOPE_API_KEY")
            os.environ["FLOW_EMBEDDING_API_KEY"] = "dummy"

        os.environ["FLOW_EMBEDDING_BASE_URL"] = settings.dashscope_api_base

        # 构造 ES 参数
        es_url = f"http://{settings.es_user}:{settings.es_password}@{settings.es_host}" if settings.es_user else None
        es_params_json = json.dumps({"hosts": es_url})

        self.reme_app = ReMeApp(
            f"llm.default.api_key={settings.openrouter_api_key}",
            f"llm.default.base_url={settings.openrouter_api_base}",
            f"llm.default.model_name={LlmModelName.OPENROUTER_GEMINI_3_FLASH_PREVIEW.value}",
            "llm.default.backend=openai_compatible",
            f"embedding_model.default.model_name={EmbeddingModelName.DASHSCOPE_TEXT_EMBEDDING_V4.value}",
            "embedding_model.default.backend=openai_compatible",
            "vector_store.default.backend=elasticsearch",
            f"vector_store.default.params={es_params_json}",
            "init_logger=false",  # 禁用 flowllm 的日志初始化，保留应用自己的日志配置
        )
        
        await self.reme_app.async_start()
        self._is_ready = True
        print("✅ [IntentService] Service Fully Started.")

    async def shutdown(self):
        """对应原 main.py 的 lifespan shutdown 部分"""
        if self.reme_app:
            await self.reme_app.async_stop()
        if self.maintenance_vs:
            await self.maintenance_vs.async_close()
            print("🛑 [IntentService] Maintenance Connection Closed")
        self._is_ready = False
        print("🛑 [IntentService] Stopped")

    # ===========================================================================
    # 2. 内部私有辅助方法 (原 main.py 中的独立函数)
    # ===========================================================================

    def _construct_standard_node(
        self,
        workspace_id: str,
        unique_id: str,
        when_to_use: str,   # 触发条件
        answer: str,        # 实际回答
        tags: List[str],
        author: str = "manual"
    ) -> VectorNode:
        """
        构造符合 ReMe/ES 标准的 VectorNode 结构 (原样保留)
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 1. 构造内层 Metadata
        inner_meta_dict = {
            "when_to_use": when_to_use,
            "experience": answer,
            "tags": tags,
            "confidence": 1.0,
            "step_type": "decision",
            "tools_used": []
        }
        
        # 2. 构造外层 Metadata
        outer_meta = {
            "memory_type": "task",
            "content": answer,
            "score": 1.0,
            "time_created": now_str,
            "time_modified": now_str,
            "author": author,
            "metadata": json.dumps(inner_meta_dict, ensure_ascii=False) 
        }

        return VectorNode(
            unique_id=unique_id,
            workspace_id=workspace_id,
            content=when_to_use,    # Trigger 用于 Embedding
            metadata=outer_meta,
            vector=None             # 稍后计算
        )

    async def _batch_insert(self, nodes: List[VectorNode]):
        """批量插入辅助函数 (原样保留)"""
        if not self.maintenance_vs: return
        try:
            texts = [n.content for n in nodes]
            # 使用 maintenance_vs 自带的 embedding model
            embeddings = self.maintenance_vs.embedding_model.get_embeddings(texts)
            if not embeddings: 
                raise ValueError("Embeddings generation returned empty")
            
            for i, node in enumerate(nodes):
                node.vector = embeddings[i] 
                
            await self.maintenance_vs.async_insert(nodes, workspace_id=nodes[0].workspace_id)
            print(f"   ✅ Inserted batch of {len(nodes)}")
        except Exception as e:
            print(f"   ⚠️ Batch insert failed: {e}")
            raise e

    async def _llm_call_with_retry(self, messages: List[dict]) -> ClassifyResult:
        """带重试机制的 LLM 调用 (原样保留逻辑)"""
        last_exception = None
        unique_models = []
        seen = set()
        
        # 构建去重后的模型列表
        for m in self.MODEL_FALLBACK_CHAIN:
            if m and m not in seen:
                unique_models.append(m)
                seen.add(m)

        for model_name in unique_models:
            print(f"🤖 [IntentService] Trying model: {model_name}...")
            try:
                if not self.openai_client: raise ValueError("OpenAI Client not initialized")
                
                completion = await self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    timeout=30,
                    temperature=0,
                    extra_body={
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": CLASSIFY_JSON_SCHEMA
                        }
                    }
                )
                usage_stats = {
                    "input_tokens": completion.usage.prompt_tokens if completion.usage else 0,
                    "output_tokens": completion.usage.completion_tokens if completion.usage else 0
                }
                if not completion.choices or not completion.choices[0].message.content:
                    raise ValueError("Empty response from LLM")
                content = completion.choices[0].message.content.strip()
                
                # 清洗 markdown
                if content.startswith("```"):
                    content = content.replace("```json", "").replace("```", "").strip()
                
                try:
                    result = ClassifyResult.model_validate_json(content)
                except ValidationError:
                    try:
                        temp = json.loads(content)
                        if isinstance(temp, str):
                            result = ClassifyResult.model_validate_json(temp)
                        else:
                            result = ClassifyResult.model_validate(temp)
                    except Exception:
                        raise ValueError(f"Failed to parse JSON: {content[:100]}...")
                return result, usage_stats
            except Exception as e:
                print(f"⚠️ Model {model_name} failed: {e}")
                last_exception = e
                continue
        
        raise Exception(f"All models failed. Last error: {last_exception}")

    # ===========================================================================
    # 3. 核心业务方法 (对应原 API 接口逻辑)
    # ===========================================================================

    async def predict_intent(self, req: ClassifyRequest) -> ClassifyResponse:
        """对应原 /classify 接口"""
        if not self._is_ready:
            raise RuntimeError("IntentService not initialized. Check lifespan.")

        def _extract_user_query_text(raw_query: str) -> str:
            """仅用于判断用户是否在 query 明确提到关键字，排除「数据源」拼接内容。"""
            if not raw_query:
                return ""
            # 仅保留“数据源：”之前的内容，避免前端勾选项影响判断
            return raw_query.split("数据源", 1)[0].strip()

        memory_context = ""
        search_query = req.query
        if req.history:
            search_query = req.history + req.query

        # 1. 检索 ReMe
        try:
            res = await self.reme_app.async_execute(
                name="retrieve_task_memory_simple",
                workspace_id=config.UNIFIED_WORKSPACE_ID,
                query=search_query,
                top_k=5
            )
            if isinstance(res, dict) and "answer" in res:
                memory_context = res["answer"]
            elif hasattr(res, "result"):
                memory_context = str(res.result)
            if memory_context is None: memory_context = ""
            
            print(f"🧠 Retrieved Context ({len(memory_context)} chars): {memory_context[:50]}...")
        except Exception as e:
            print(f"⚠️ Memory Retrieve Skipped: {e}")

        # 2. 构造 Prompt
        display_context = memory_context if memory_context else "暂无相关历史经验"
        full_prompt = prompt_template.N8N_SYSTEM_PROMPT.format(memory_context=display_context)
        formatted_history = req.history if req.history else "No History"

        user_input = f"""
        <user_context>
            <preferred_entity_selection>
                {req.preferred_entity or "None (User did not select)"}
            </preferred_entity_selection>
            <conversation_history>
                {formatted_history}
            </conversation_history>
        </user_context>
        <current_query>
            {req.query}
        </current_query>
        <instruction>
            Please classify the intent of the content in <current_query>.
            Note: If the intent in <current_query> conflicts with <preferred_entity_selection>, trust the explicit intent in <current_query>.
        </instruction>
        """
        
        messages = [
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": user_input}
        ]

        # 3. 调用 LLM
        try:
            result, token_stats = await self._llm_call_with_retry(messages)
            user_query_text = _extract_user_query_text(req.query).lower()
            if "ins" in user_query_text and result.category == IntentEnum.SELECTION:
                print(f"🔄 [IntentService] Auto-correct: 'ins' detected. SELECTION -> MEDIA.")
                result.category = IntentEnum.MEDIA
                result.reasoning = f"(自动纠正) 检测到关键字 'ins'，强制从选品纠正为媒体。原因为: {result.reasoning}"
            # 组装 Response
            return ClassifyResponse(
                category=result.category,
                reasoning=result.reasoning,
                memory_used=bool(memory_context),
                retrieved_context=display_context,
                input_tokens=token_stats["input_tokens"],   
                output_tokens=token_stats["output_tokens"]  
            )
        except Exception as e:
            print(f"🔥 ALL RETRIES FAILED: {e}")
            # 兜底返回
            return ClassifyResponse(
                category=IntentEnum.CHATBOT,
                reasoning=f"System Fallback: {str(e)}",
                memory_used=False,
                retrieved_context="Error"
            )

    async def process_feedback(self, req: FeedbackRequest):
        """对应原 /feedback 接口的后台任务逻辑"""
        if not self._is_ready: return
        try:
            print(f"🧠 Learning: {req.query} -> {req.correct_category.value}")
            await self.reme_app.async_execute(
                name="summary_task_memory",
                workspace_id=config.UNIFIED_WORKSPACE_ID,
                trajectories=[{
                    "messages": [
                        {"role": "user", "content": req.query},
                        {"role": "assistant", "content": f"Category: {req.correct_category.value}\nReason: {req.reason}"}
                    ],
                    "score": 1.0
                }]
            )
            print("✅ Memory Saved.")
        except Exception as e:
            print(f"❌ Learning Failed: {e}")

    # ===========================================================================
    # 4. 维护相关方法 (对应原 Maintenance 接口逻辑)
    # ===========================================================================

    async def import_memories_from_text(self, lines: List[str], workspace_id_override: Optional[str] = None) -> int:
        """
        对应原 /maintenance/import_jsonl 接口逻辑
        注意：这里接收的是字符串列表 (lines)，文件读取步骤放在 Endpoint 层处理
        """
        if not self.maintenance_vs:
            raise RuntimeError("Maintenance Service not initialized")

        batch_nodes = []
        processed = 0
        
        for line in lines:
            if not line.strip(): continue
            try:
                raw = json.loads(line)
                uid = raw.get("unique_id", uuid4().hex)
                wid = workspace_id_override or raw.get("workspace_id", config.UNIFIED_WORKSPACE_ID)
                
                # --- 智能判断逻辑 (原样保留) ---
                raw_meta = raw.get("metadata", {})
                
                # 情况 A: 标准复杂格式
                if "metadata" in raw_meta and isinstance(raw_meta["metadata"], str):
                     node = VectorNode(
                        unique_id=uid,
                        workspace_id=wid,
                        content=raw.get("content"), 
                        metadata=raw_meta,
                        vector=None
                    )
                
                # 情况 B: 简单格式 -> 升级为标准格式
                else:
                    answer = raw_meta.get("content") or raw.get("answer") or "No Content"
                    tags = raw_meta.get("tags", [])
                    
                    node = self._construct_standard_node(
                        workspace_id=wid,
                        unique_id=uid,
                        when_to_use=raw.get("content"), 
                        answer=answer,
                        tags=tags,
                        author="batch_import"
                    )
                
                batch_nodes.append(node)
                
                if len(batch_nodes) >= 10: 
                    await self._batch_insert(batch_nodes)
                    processed += len(batch_nodes)
                    batch_nodes = []
                    
            except Exception as e:
                print(f"❌ Error line: {e}")

        if batch_nodes:
            await self._batch_insert(batch_nodes)
            processed += len(batch_nodes)
        
        return processed

    async def upsert_memory(self, req: MemoryMaintainRequest) -> str:
        """对应原 /maintenance/memory 接口"""
        if not self.maintenance_vs: 
            raise RuntimeError("Maintenance Service not initialized")
        
        try:
            final_uid = req.unique_id if req.unique_id else uuid4().hex

            # 使用 _construct_standard_node 确保结构一致
            node = self._construct_standard_node(
                workspace_id=req.workspace_id,
                unique_id=final_uid,
                when_to_use=req.when_to_use,
                answer=req.content,
                tags=req.tags,
                author="api_manual"
            )
            
            # 生成向量
            emb = self.maintenance_vs.embedding_model.get_embeddings([node.content])
            if not emb: 
                raise Exception("Embedding failed")
            node.vector = emb[0]
            
            # 插入
            await self.maintenance_vs.async_insert([node], workspace_id=req.workspace_id)
            print(f"✅ Manual Memory Saved (Standardized): {final_uid}")
            return final_uid
            
        except Exception as e:
            print(f"❌ Upsert Error: {e}")
            raise e

    async def list_memories(self, workspace_id: str, limit: int) -> Dict[str, Any]:
        """对应原 /maintenance/list 接口"""
        if not self.maintenance_vs:
            raise RuntimeError("Maintenance Service not initialized")
        
        nodes = await self.maintenance_vs.async_list_workspace_nodes(workspace_id=workspace_id, max_size=limit)
        
        result = []
        for node in nodes:
            node_dict = node.model_dump()
            node_dict.pop("vector", None) # 隐藏向量
            result.append(node_dict)
            
        return {
            "workspace_id": workspace_id,
            "total_retrieved": len(result),
            "items": result
        }

    async def clear_workspace(self, workspace_id: str) -> Dict[str, str]:
        """对应原 /maintenance/clear 接口"""
        if not self.maintenance_vs:
             raise RuntimeError("Maintenance Service not initialized")
        
        exists = await self.maintenance_vs.async_exist_workspace(workspace_id)
        if not exists:
            return {"status": "skipped", "message": f"Workspace {workspace_id} does not exist."}
            
        await self.maintenance_vs.async_delete_workspace(workspace_id)
        print(f"🔥 Workspace {workspace_id} deleted.")
        
        await self.maintenance_vs.async_create_workspace(workspace_id)
        print(f"✅ Workspace {workspace_id} recreated.")
        
        return {"status": "success", "message": f"All memories in {workspace_id} have been cleared."}

    async def export_memories_generator(self, workspace_id: str) -> AsyncGenerator[str, None]:
        """
        对应原 /maintenance/export 接口的核心逻辑
        返回一个 AsyncGenerator，供 Endpoint 封装为 StreamingResponse
        """
        if not self.maintenance_vs:
            raise RuntimeError("Maintenance Service not initialized")

        # 1. 拉取数据 (保留原代码逻辑 max_size=10000)
        nodes = await self.maintenance_vs.async_list_workspace_nodes(workspace_id=workspace_id, max_size=10000)
        print(f"📤 Exporting {len(nodes)} nodes from {workspace_id}...")

        # 2. 生成器逻辑
        for node in nodes:
            node_dict = node.model_dump()
            node_dict["vector"] = [] # 清空向量
            yield json.dumps(node_dict, ensure_ascii=False) + "\n"

# 实例化单例
intent_service = IntentService()