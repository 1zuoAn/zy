import json
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Query
from fastapi.responses import StreamingResponse

# 引入 Schema
from app.schemas.request.intent_request import (
    ClassifyRequest, 
    FeedbackRequest, 
    MemoryMaintainRequest
)
from app.schemas.response.common import CommonResponse
from app.schemas.response.intent_response import ClassifyResponse

# 引入 Service 和 Config
from app.service.intent import intent_service
from app.service.intent import config

router = APIRouter()

# ===========================================================================
# 核心业务接口
# ===========================================================================

@router.post("/classify", response_model=CommonResponse[ClassifyResponse])
async def classify_intent(request: ClassifyRequest):
    """
    意图识别
    """
    try:
        # Service 直接返回 ClassifyResponse 对象 (包含 category, reasoning, memory_used 等)
        result = await intent_service.predict_intent(request)
        return CommonResponse(result=result)
    except Exception as e:
        # 建议记录日志 logger.error(f"Intent Error: {e}")
        raise HTTPException(status_code=500, detail=f"Intent classification failed: {str(e)}")

@router.post("/feedback")
async def feedback_intent(request: FeedbackRequest, background_tasks: BackgroundTasks):
    """
    意图反馈：异步学习 (对应原 /feedback)
    """
    # 将耗时操作放入后台任务
    background_tasks.add_task(intent_service.process_feedback, request)
    return CommonResponse(result="Feedback received and processing in background")

# ===========================================================================
# 维护接口 (Maintenance)
# ===========================================================================

@router.post("/maintenance/memory")
async def upsert_memory(request: MemoryMaintainRequest):
    """
    手动新增或修改一条记忆规则
    """
    try:
        uid = await intent_service.upsert_memory(request)
        return CommonResponse(result={"id": uid, "status": "success"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upsert failed: {str(e)}")

@router.post("/maintenance/import_jsonl")
async def import_jsonl(
    file: UploadFile = File(...), 
    workspace_id_override: Optional[str] = Query(None, description="可选：覆盖默认工作区ID")
):
    """
    批量导入 JSONL 文件
    """
    try:
        # 1. Endpoint 层负责读取文件流
        content = await file.read()
        
        # 2. 将 bytes 转换为 lines 列表
        lines = [line for line in content.decode("utf-8").strip().split("\n") if line.strip()]
        
        if not lines:
            return CommonResponse(result={"status": "empty_file", "imported": 0})

        # 3. 调用 Service 处理业务逻辑
        count = await intent_service.import_memories_from_text(lines, workspace_id_override)
        
        return CommonResponse(result={"status": "success", "imported": count})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")

@router.get("/maintenance/export")
async def export_memories(
    workspace_id: str = Query(default=config.UNIFIED_WORKSPACE_ID)
):
    """
    导出记忆为 JSONL 文件 (流式下载)
    注意：此接口返回二进制流，不使用 CommonResponse 包装
    """
    try:
        # 1. 获取 Service 提供的异步生成器
        data_generator = intent_service.export_memories_generator(workspace_id)
        
        # 2. 构造文件名
        filename = f"backup_{workspace_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        # 3. 返回流式响应
        return StreamingResponse(
            data_generator,
            media_type="application/x-ndjson",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.get("/maintenance/list")
async def list_memories(
    workspace_id: str = Query(default=config.UNIFIED_WORKSPACE_ID),
    limit: int = Query(default=100, ge=1, le=1000)
):
    """
    查看当前记忆列表
    """
    try:
        data = await intent_service.list_memories(workspace_id, limit)
        return CommonResponse(result=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List failed: {str(e)}")

@router.post("/maintenance/clear")
async def clear_workspace(
    workspace_id: str = Query(default=config.UNIFIED_WORKSPACE_ID)
):
    """
    🔥 清空指定工作区
    """
    try:
        result = await intent_service.clear_workspace(workspace_id)
        return CommonResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")