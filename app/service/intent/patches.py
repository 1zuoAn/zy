import warnings
from typing import List
from loguru import logger
from elasticsearch import NotFoundError
from datetime import datetime

# 导入原本的库
from flowllm.core.vector_store.es_vector_store import EsVectorStore
from reme_ai.summary.task.memory_deduplication_op import MemoryDeduplicationOp
from reme_ai.vector_store.recall_vector_store_op import RecallVectorStoreOp
from reme_ai.schema.memory import vector_node_to_memory, BaseMemory
from flowllm.core.schema import VectorNode

warnings.filterwarnings("ignore")
def apply_monkey_patches():
    print("🔧 Applying Monkey Patches...")
    async def async_exist_workspace_patched(self, workspace_id: str) -> bool:
        try:
            await self._async_client.indices.get(index=workspace_id)
            return True
        except NotFoundError:
            return False
        except Exception as e:
            if "index_not_found" in str(e):
                return False
            logger.warning(f"⚠️ [Patch] Check index failed: {e}")
            raise e

    print("🔧 Applying Monkey Patch 1: EsVectorStore (HEAD -> GET)...")
    EsVectorStore.async_exist_workspace = async_exist_workspace_patched
    print("✅ Patch 1 Applied.")


    # --- 补丁 2: MemoryDeduplicationOp (保持不变) ---
    async def _get_existing_task_memory_embeddings_patched(self, workspace_id: str) -> List[List[float]]:
        try:
            if not hasattr(self, "vector_store") or not self.vector_store or not workspace_id:
                return []

            logger.debug(f"Fetching existing nodes via iterator for workspace: {workspace_id}...")
            existing_nodes = await self.vector_store.async_iter_workspace_nodes(workspace_id=workspace_id)

            existing_embeddings = []
            for node in existing_nodes:
                if hasattr(node, "embedding") and node.embedding:
                    existing_embeddings.append(node.embedding)
                elif hasattr(node, "vector") and node.vector: 
                    existing_embeddings.append(node.vector)

            max_memories = self.op_params.get("max_existing_task_memories", 1000)
            if len(existing_embeddings) > max_memories:
                existing_embeddings = existing_embeddings[:max_memories]

            logger.debug(f"Retrieved {len(existing_embeddings)} existing task memory embeddings (Patched)")
            return existing_embeddings

        except Exception as e:
            logger.warning(f"Failed to retrieve existing task memory embeddings: {e}")
            return []

    print("🔧 Applying Monkey Patch 2: MemoryDeduplicationOp (No Script)...")
    MemoryDeduplicationOp._get_existing_task_memory_embeddings = _get_existing_task_memory_embeddings_patched
    print("✅ Patch 2 Applied.")


    # --- 补丁 3: RecallVectorStoreOp  ---

    async def async_execute_recall_patched(self):
        """[Patched] Execute recall using strict KNN search for ES Serverless"""
        try:
            recall_key: str = self.op_params.get("recall_key", "query")
            top_k: int = self.context.get("top_k", 3)
            query: str = self.context.get(recall_key)
            workspace_id: str = self.context.workspace_id

            if not query:
                logger.warning("Query is empty, skipping recall.")
                self.context.response.metadata["memory_list"] = []
                return

            # 1. 手动生成 Query Embedding
            if hasattr(self.vector_store, "embedding_model") and self.vector_store.embedding_model:
                embeddings = self.vector_store.embedding_model.get_embeddings([query])
                if not embeddings:
                    logger.warning("Failed to generate embedding for query.")
                    self.context.response.metadata["memory_list"] = []
                    return
                query_vector = embeddings[0]
            else:
                logger.error("No embedding model found in vector store.")
                self.context.response.metadata["memory_list"] = []
                return

            # 2. 构造 ES 8.x 标准 KNN 查询
            search_body = {
                "knn": {
                    "field": "vector",
                    "query_vector": query_vector,
                    "k": top_k,
                    "num_candidates": max(100, top_k * 10)
                },
                "_source": True 
            }

            # 3. 执行原生搜索
            logger.info(f"Executing Patched KNN Search (field='vector') in workspace: {workspace_id}")
            resp = await self.vector_store._async_client.search(
                index=workspace_id,
                body=search_body,
                size=top_k
            )

            # 4. 解析结果 (🚑 关键修复逻辑)
            memory_list: List[BaseMemory] = []
            hits = resp.get("hits", {}).get("hits", [])
            
            for hit in hits:
                source = hit.get("_source", {})
                try:
                    # [Fix] 提取并清洗 metadata
                    meta = source.get("metadata", {})
                    if not isinstance(meta, dict): 
                        meta = {}
                    
                    # 🚑 关键修复: 强制补全缺失的 time_created
                    # ReMe 的 vector_node_to_memory 强依赖此字段
                    if "time_created" not in meta:
                        meta["time_created"] = datetime.now().isoformat()
                    
                    # 🚑 可选修复: 补全 memory_type 默认为 task
                    if "memory_type" not in meta:
                        meta["memory_type"] = "task"

                    # 步骤 A: 还原 VectorNode
                    node = VectorNode(
                        unique_id=hit.get("_id"), 
                        workspace_id=source.get("workspace_id"),
                        content=source.get("content"),
                        metadata=meta, # <--- 传入修复后的 meta
                        vector=None 
                    )
                    
                    # 步骤 B: 使用 ReMe 官方转换函数
                    memory = vector_node_to_memory(node)
                    
                    # 步骤 C: 补上相关性分数
                    memory.score = hit.get("_score")
                    
                    memory_list.append(memory)
                    
                except Exception as e:
                    # 打印更详细的错误日志以便排查
                    logger.warning(f"Failed to parse memory hit: {e}. Source keys: {source.keys()}")

            # 5. 设置上下文
            logger.info(f"Patched Recall retrieved {len(memory_list)} memories.")
            self.context.response.metadata["memory_list"] = memory_list

        except Exception as e:
            logger.error(f"Error in Patched RecallVectorStoreOp: {e}")
            self.context.response.metadata["memory_list"] = []

    print("🔧 Applying Monkey Patch 3: RecallVectorStoreOp (KNN 'vector' + Metadata Fix)...")
    RecallVectorStoreOp.async_execute = async_execute_recall_patched
    print("✅ Patch 3 Applied.")
    # --- 补丁 4: EsVectorStore 缺失方法修复 (async_list_workspace_nodes) ---
    async def async_list_workspace_nodes_patched(self, workspace_id: str, max_size: int = 10000, **kwargs) -> List[VectorNode]:
        try:
            # 1. 检查是否存在 (利用已有的 Patch 1)
            if not await self.async_exist_workspace(workspace_id=workspace_id):
                logger.warning(f"workspace_id={workspace_id} does not exist!")
                return []

            # 2. 执行全量搜索
            # 注意: max_size 默认 10000，ES 默认窗口限制也是 10000。如果超过需用 Scroll API (暂不实现)
            resp = await self._async_client.search(
                index=workspace_id,
                body={"query": {"match_all": {}}, "size": max_size}
            )
            
            # 3. 解析结果 (内联 doc2node 逻辑以防万一)
            nodes = []
            hits = resp.get("hits", {}).get("hits", [])
            for hit in hits:
                source = hit.get("_source", {})
                node = VectorNode(
                    unique_id=hit.get("_id"),
                    workspace_id=workspace_id,
                    content=source.get("content"),
                    metadata=source.get("metadata", {}),
                    vector=source.get("vector") 
                )
                nodes.append(node)
                
            return nodes
        except Exception as e:
            logger.error(f"Failed to list nodes (Patched): {e}")
            raise e

    print("🔧 Applying Monkey Patch 4: EsVectorStore (Add async_list_workspace_nodes)...")
    EsVectorStore.async_list_workspace_nodes = async_list_workspace_nodes_patched
    print("✅ Patch 4 Applied.")