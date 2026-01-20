from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field

class IntentEnum(str, Enum):
    IMAGE_SEARCH = "图搜"
    SELECTION = "选品"
    IMAGE_DESIGN = "生图改图"
    TRENDS = "趋势报告"
    MEDIA = "媒体"
    SHOP = "店铺"
    SCHEDULE = "定时任务"
    CHATBOT = "聊天机器人"

class ClassifyRequest(BaseModel):
    query: str
    preferred_entity: Optional[str] = None
    history: Optional[str] = None

class FeedbackRequest(BaseModel):
    query: str
    correct_category: IntentEnum
    reason: str

class MemoryMaintainRequest(BaseModel):
    workspace_id: str = Field(..., description="工作区ID，例如 intent_router_v2")
    unique_id: Optional[str] = Field(None, description="记忆ID，如果不传则新建")
    when_to_use: str = Field(..., description="检索触发条件 (Query/Key)")
    content: str = Field(..., description="具体的记忆内容/回答")
    category: str = Field("task", description="记忆类型: task, personal, tool")
    score: float = Field(1.0, description="置信度分数")
    tags: List[str] = Field(default_factory=list, description="标签")