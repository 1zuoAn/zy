# app/core/tools/business_tools.py

from typing import Annotated, Optional, Literal
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from loguru import logger

from app.core.config.constants import WorkflowType
from app.service.chains.workflow.workflow_delegate import get_delegate
from app.schemas.entities.workflow.graph_state import MainOrchestratorState

delegate = get_delegate()

# --- 辅助映射函数 (保持不变) ---
def _map_platform_to_workflow_type(platform: str, intent: str) -> WorkflowType:
    p = platform.lower() if platform else ""
    if intent == "selection":
        if "抖" in p: return WorkflowType.SELECT_DOUYI
        if "亚马逊" in p or "amazon" in p or "temu" in p or "海外" in p: return WorkflowType.SELECT_ABROAD_GOODS
        return WorkflowType.SELECT_ZHIYI
    if intent == "media":
        if "海外" in p or "ins" in p: return WorkflowType.MEDIA_ABROAD_INS
        if "知款" in p: return WorkflowType.MEDIA_ZHIKUAN_INS
        return WorkflowType.MEDIA_ZXH_XHS
    if intent == "shop": return WorkflowType.SHOP
    return WorkflowType.CHAT

# ================= 纯粹能力导向的工具定义 =================

@tool
def search_products(
    query: str, 
    platform: str, 
    sort: Optional[str] = "default",
    state: Annotated[MainOrchestratorState, InjectedState] = None
) -> str:
    """
    通过文本关键词搜索商品数据。
    支持功能：找款、查价格、看销量、筛选商品。
    
    Args:
        query: 搜索关键词 (如 "红色连衣裙")
        platform: 目标平台 (如 "知衣", "抖音", "亚马逊")
        sort: 排序方式 (如 "销量倒序", "价格升序")
    """
    logger.info(f"[Tool] 文本选品: {query}")
    req = state["request"]
    req.user_query = f"{query} {sort}"
    req.preferred_entity = platform
    
    # 映射逻辑：将通用选品请求路由到具体业务线
    wf_type = _map_platform_to_workflow_type(platform, intent="selection")
    res = delegate.execute(wf_type, req)
    return f"选品结果: {res.output}" if res.success else f"选品失败: {res.output}"

@tool
def search_products_by_image(
    image_url: str,
    platform: str,
    category: Optional[str] = None,
    state: Annotated[MainOrchestratorState, InjectedState] = None
) -> str:
    """
    通过图片搜索相似商品（以图搜图）。
    
    Args:
        image_url: 图片的 HTTP URL 地址
        platform: 目标平台 (如 "知衣", "淘宝")
        category: (可选) 图片所属类目，辅助提高准确度
    """
    logger.info(f"[Tool] 图搜: {image_url}")
    req = state["request"]
    req.images = [image_url] # 关键：设置图片参数
    req.user_query = category or "同款"
    req.preferred_entity = platform
    
    wf_type = _map_platform_to_workflow_type(platform, intent="selection")
    res = delegate.execute(wf_type, req)
    return f"图搜结果: {res.output}" if res.success else f"图搜失败: {res.output}"

@tool
def search_social_media(
    query: str, 
    platform: str,
    state: Annotated[MainOrchestratorState, InjectedState] = None
) -> str:
    """
    搜索社交媒体内容（帖子、笔记、博主）。
    注意：此工具返回的是内容灵感，而非商品购买链接。  # TODO 需要更改工具能力边界的描述
    
    Args:
        query: 搜索关键词
        platform: 平台 (如 "小红书", "Instagram", "知款")
    """
    logger.info(f"[Tool] 社媒搜索: {query}")
    req = state["request"]
    req.user_query = query
    req.preferred_entity = platform
    
    wf_type = _map_platform_to_workflow_type(platform, intent="media")
    res = delegate.execute(wf_type, req)
    return res.output

@tool
def generate_or_edit_image(
    prompt: str,
    operation: Literal["generate", "edit"],
    reference_image_url: Optional[str] = None,
    state: Annotated[MainOrchestratorState, InjectedState] = None
) -> str:
    """
    执行 AI 图像生成或编辑任务。
    
    Args:
        prompt: 图像描述或修改指令
        operation: 'generate' (文生图) 或 'edit' (图生图/改款)
        reference_image_url: 当 operation='edit' 时必须提供原图 URL
    """
    logger.info(f"[Tool] AI生图: {operation} - {prompt}")
    req = state["request"]
    req.user_query = prompt
    if operation == "edit" and reference_image_url:
        req.images = [reference_image_url]
        
    res = delegate.execute(WorkflowType.IMAGE_DESIGN, req)
    return res.output

@tool
def deep_research_trend(
    topic: str,
    state: Annotated[MainOrchestratorState, InjectedState] = None
) -> str:
    """
    执行深度行业趋势调研，生成分析报告。
    调用外部搜索和知识库，适合回答宏观趋势、市场洞察类问题。
    
    Args:
        topic: 调研主题 (如 "2025秋冬女装色彩趋势")
    """
    logger.info(f"[Tool] 深度调研: {topic}")
    req = state["request"]
    req.user_query = topic
    
    res = delegate.execute(WorkflowType.TRENDS, req)
    return res.output

@tool
def analyze_shop_data(
    shop_name: str,
    platform: str,
    state: Annotated[MainOrchestratorState, InjectedState] = None
) -> str:
    """
    查询特定店铺的经营数据（销量、排名、增长率）。
    
    Args:
        shop_name: 店铺名称
        platform: 平台 (如 "淘宝", "天猫")
    """
    req = state["request"]
    req.user_query = shop_name
    req.preferred_entity = platform
    res = delegate.execute(WorkflowType.SHOP, req)
    return res.output

@tool
def manage_schedule_task(
    instruction: str,
    state: Annotated[MainOrchestratorState, InjectedState] = None
) -> str:
    """
    管理定时任务（创建、删除、查询）。
    
    Args:
        instruction: 具体的自然语言指令 (如 "每天早上8点推送")
    """
    req = state["request"]
    req.user_query = instruction
    res = delegate.execute(WorkflowType.SCHEDULE, req)
    return res.output

# 导出工具列表
ALL_TOOLS = [
    search_products,
    search_products_by_image,
    search_social_media,
    generate_or_edit_image,
    deep_research_trend,
    analyze_shop_data,
    manage_schedule_task
]