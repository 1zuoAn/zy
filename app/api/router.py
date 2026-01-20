"""
API 路由聚合
"""
from fastapi import APIRouter

from app.api.endpoints import (
    completion_endpoint,
    health,
    media_workflow_endpoint,
    suggestion_endpoint,
    workflow_graph_endpoint,
    intent_endpoint,
)

# 创建主路由
api_router = APIRouter()

# 包含各个模块的路由
api_router.include_router(health.router, prefix="/health", tags=["健康检查"])

# 工作流入口
api_router.include_router(media_workflow_endpoint.router, prefix="/workflow", tags=["工作流入口"])

# 输入补全
api_router.include_router(completion_endpoint.router, prefix="/completion", tags=["输入补全"])

# 工作流图可视化
api_router.include_router(workflow_graph_endpoint.router, prefix="/workflow/graph", tags=["工作流可视化"])

# 建议生成
api_router.include_router(suggestion_endpoint.router, prefix="/suggestion", tags=["建议生成"])

# 意图识别
api_router.include_router(intent_endpoint.router, prefix="/intent", tags=["意图识别"])