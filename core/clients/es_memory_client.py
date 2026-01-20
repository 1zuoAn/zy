# -*- coding: utf-8 -*-
"""
ES 记忆客户端 - 用于存储和召回用户历史查询
"""

from datetime import datetime, timezone
from typing import List, Optional

from elasticsearch import Elasticsearch
from flowllm.core.embedding_model.openai_compatible_embedding_model import OpenAICompatibleEmbeddingModel
from loguru import logger

from app.config import settings
from app.core.config.constants import EmbeddingModelName

INDEX_NAME = "zxy_completion_memory"

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "user_id": {"type": "keyword"},
            "clean_query": {"type": "text"},
            "query_vector": {
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "cosine",
            },
            "timestamp": {"type": "date"},
            "context": {"type": "keyword"},
        }
    }
}


class EsMemoryClient:
    """ES 记忆客户端"""

    def __init__(self):
        self._es: Optional[Elasticsearch] = None
        self._embedding: Optional[OpenAICompatibleEmbeddingModel] = None
        self._index_checked = False

    def _get_es(self) -> Elasticsearch:
        if not self._es:
            self._es = Elasticsearch(
                hosts=[f"http://{settings.es_host}"],
                basic_auth=(settings.es_user, settings.es_password),
            )
        return self._es

    def _get_embedding(self) -> OpenAICompatibleEmbeddingModel:
        if not self._embedding:
            self._embedding = OpenAICompatibleEmbeddingModel(
                model_name=EmbeddingModelName.DASHSCOPE_TEXT_EMBEDDING_V4.value,
                api_key=settings.dashscope_api_key,
                base_url=settings.dashscope_api_base,
            )
        return self._embedding

    def _ensure_index(self):
        """确保索引存在（首次调用时检查）"""
        if self._index_checked:
            return
        try:
            es = self._get_es()
            if not es.indices.exists(index=INDEX_NAME):
                es.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)
                logger.info(f"[ES记忆] 索引 {INDEX_NAME} 创建成功")
            self._index_checked = True
        except Exception as e:
            logger.warning(f"[ES记忆] 索引检查/创建失败: {e}")

    def store_query(self, user_id: str, clean_query: str, context: str = "") -> bool:
        """存储用户查询（同步，失败不抛异常）"""
        try:
            self._ensure_index()
            vector = self._get_embedding().get_embeddings([clean_query])[0]
            doc = {
                "user_id": user_id,
                "clean_query": clean_query,
                "query_vector": vector,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "context": context,
            }
            self._get_es().index(index=INDEX_NAME, document=doc)
            return True
        except Exception as e:
            logger.warning(f"[ES记忆] 存储失败: {e}")
            return False

    def recall_queries(self, user_id: str, query: str, top_k: int = 5) -> List[str]:
        """召回相似历史查询"""
        try:
            self._ensure_index()
            vector = self._get_embedding().get_embeddings([query])[0]
            body = {
                "knn": {
                    "field": "query_vector",
                    "query_vector": vector,
                    "k": 50,
                    "num_candidates": 100,
                    "filter": {"term": {"user_id": user_id}},
                },
                "rescore": {
                    "window_size": 50,
                    "query": {
                        "rescore_query": {
                            "function_score": {
                                "query": {"match_all": {}},
                                "functions": [
                                    {
                                        "gauss": {
                                            "timestamp": {
                                                "origin": "now",
                                                "scale": "7d",
                                                "offset": "1d",
                                                "decay": 0.5,
                                            }
                                        }
                                    }
                                ],
                                "boost_mode": "multiply",
                            }
                        },
                        "query_weight": 1.0,
                        "rescore_query_weight": 1.0,
                    },
                },
                "_source": ["clean_query"],
            }
            resp = self._get_es().search(index=INDEX_NAME, body=body, size=top_k)
            return [hit["_source"]["clean_query"] for hit in resp["hits"]["hits"]]
        except Exception as e:
            logger.warning(f"[ES记忆] 召回失败: {e}")
            return []


es_memory_client = EsMemoryClient()
