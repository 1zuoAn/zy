from typing import Optional
from pydantic import BaseModel, Field
from app.schemas.request.intent_request import IntentEnum

# 给 LLM 用 (Json Schema)
class ClassifyResult(BaseModel):
    category: IntentEnum = Field(..., description="必须是预定义类别之一")
    reasoning: str = Field(..., description="简短的推理过程")

# 给 API 返回用 (包含 RAG 元数据)
class ClassifyResponse(ClassifyResult):
    memory_used: bool
    retrieved_context: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0

# 定义 JSON Schema 用于 LLM Structured Output
CLASSIFY_JSON_SCHEMA = {
    "name": "classify_intent",
    "schema": ClassifyResult.model_json_schema(),
    "strict": True 
}