# @Author   : kiro
# @Time     : 2025/12/14
# @File     : graph_state.py

"""
LangGraph 工作流状态类型定义

使用 TypedDict 定义状态结构，提供类型安全和 IDE 支持。
"""
from collections import defaultdict
from typing import Any, TypedDict

from app.schemas.entities.workflow.context import SubWorkflowResult
from app.schemas.entities.workflow.llm_output import IntentClassifyResult
from app.schemas.request.workflow_request import WorkflowRequest


class MainOrchestratorState(TypedDict, total=False):
    """主编排工作流状态"""

    # 输入
    request: WorkflowRequest

    # 图片提取结果
    image_content: str | None

    # 会话上下文
    conversation_history: str

    # 意图分类结果
    intent: str
    intent_result: IntentClassifyResult

    # 子工作流执行结果
    sub_workflow_result: SubWorkflowResult

    # 最终输出
    summary_text: str
    workflow_response: Any


class MediaWorkflowState(TypedDict, total=False):
    """媒体子工作流通用状态（AbroadIns, ZhikuanIns, ZxhXhs）"""

    # 输入
    request: WorkflowRequest

    # 问题关联实体
    query_ref_dict: defaultdict[str, dict]

    # 筛选项数据
    selection_dict: dict[str, Any]

    # LLM 解析结果
    param_result: Any  # 具体类型由子工作流决定
    sort_param_result: Any

    # API 调用结果
    api_request: Any
    api_resp: Any
    result_count: int

    # 分支结果
    has_query_result: bool
    entity_simple_data: list[dict[str, Any]]

    # 最终输出
    workflow_response: Any


class ZxhXhsWorkflowState(MediaWorkflowState, total=False):
    """知小红小红书工作流状态（扩展 RAG 相关字段）"""

    recall_topic_list: list[str] | None
    cleaned_topic_list: list[str] | None
    is_monitor_query: bool

    # 浏览笔记数
    browsed_count: int


class AbroadInsWorkflowState(MediaWorkflowState, total=False):
    """海外探款INS工作流状态"""

    # 监控达人模式
    is_monitor_streamer: bool

    # 浏览商品数
    browsed_count: int


class ZhikuanInsWorkflowState(MediaWorkflowState, total=False):
    """知款INS工作流状态"""

    # 监控博主模式
    is_monitor_blogger: bool

    # 浏览帖子数
    browsed_count: int


class ChatWorkflowState(TypedDict, total=False):
    """闲聊工作流状态"""

    request: WorkflowRequest
    chat_response: str
    workflow_response: Any


class SelectionWorkflowState(TypedDict, total=False):
    """选品工作流基类状态（Douyi, Zhiyi, AbroadGoods）"""

    # 输入
    request: WorkflowRequest

    # 查询维表数据
    selection_dict: dict[str, Any]

    # LLM 参数解析结果
    param_result: Any  # 具体类型由子工作流决定

    # API 调用结果
    api_request: Any
    api_resp: Any
    result_count: int
    browsed_count: int  # 浏览商品总数

    # 最终输出
    has_query_result: bool
    entity_simple_data: list[dict[str, Any]]
    workflow_response: Any  # 工作流响应


class DouyiWorkflowState(SelectionWorkflowState, total=False):
    """抖衣(抖音)选品工作流状态"""

    # 属性解析结果
    property_list: list[list[str]] | None
    user_filters: Any | None
    user_tags: list[str] | None
    sort_param_result: Any  # 排序解析结果
    shop_id: str | None  # 店铺ID（来自查询引用）

    # API 调用状态
    api_success: bool  # API 是否调用成功

    # API 调用结果（兜底调用）
    fallback_api_request: Any
    fallback_api_resp: Any
    fallback_result_count: int | None
    fallback_api_success: bool
    used_fallback: bool

    # 后处理结果
    final_result_count: int | None


class ZhiyiWorkflowState(SelectionWorkflowState, total=False):
    """知衣选品工作流状态"""

    # 品类/活动维表数据
    category_data: list[dict[str, Any]] | None
    activity_data: list[dict[str, Any]] | None
    style_data: list[dict[str, Any]] | None  # 风格数据由 LLM 解析
    kb_content: str | None  # 知识库检索内容
    category_vector_content: list[str] | None  # 类目向量检索结果

    # 排序解析结果
    sort_result: Any  # ZhiyiSortResult

    # 并行解析结果 (parallel_parse)
    style_list: list[str] | None  # 风格解析
    brand_list: list[str] | None  # 品牌列表
    property_list: list[dict[str, str]] | None  # 属性列表 (n8n propertyList)
    shop_id: str | None  # 品牌店铺 ID
    user_style_values: str | None  # 用户画像风格提取结果
    user_filter_tags: Any | None  # 用户画像 filters（原始值）

    # 路由控制字段
    is_brand_item: int  # 1=品牌查询, 0=非品牌查询 (对齐 n8n If11)
    flag: int  # 1=监控店铺, 2=全网数据 (对齐 n8n 判断查询数据类型)
    sale_type: str  # 热销 / 新品 (对齐 n8n Switch/Switch2)

    # API 分支标识
    api_branch: str  # brand / monitor_hot / monitor_new / all_hot / all_new

    # API 调用结果
    api_success: bool | None  # 主 API 是否调用成功
    goods_list: list[dict[str, Any]] | None  # 商品列表
    fallback_api_request: Any  # 兜底 API 请求
    fallback_result_count: int | None  # 兜底结果数量
    fallback_api_success: bool | None  # 兜底 API 是否调用成功
    fallback_goods_list: list[dict[str, Any]] | None  # 兜底商品列表
    used_fallback: bool | None  # 是否使用了兜底结果
    merged_goods_list: list[dict[str, Any]] | None  # 汇聚后商品列表

    # 后处理结果
    goods_id_list: list[str] | None  # 商品 ID 列表
    processed_goods_list: list[dict[str, Any]] | None  # 处理后的商品列表


class AbroadGoodsWorkflowState(SelectionWorkflowState, total=False):
    """海外探款商品工作流状态"""

    # 查询模式
    is_new_goods: bool  # 是否新品模式（vs 热销）

    # 并行解析结果
    sort_param_result: Any  # AbroadGoodsSortResult
    style_list: list[str] | None
    brand_tags: list[str] | None  # 品牌召回标签
    user_filters: Any | None

    # 引用信息
    site_name: str | None  # 站点引用名称
    site_id: str | None  # 站点引用ID
    shop_id: str | None  # 店铺引用ID（Amazon/Temu）
    platform_type_override: int | str | None  # 平台类型覆盖（Amazon/Temu）
    platform_name_override: str | None  # 平台名称覆盖（Amazon/Temu）

    # API 调用状态
    api_success: bool  # API 是否调用成功

    # 站点匹配结果
    tag_text: str | None  # 站点匹配 tag_text
    match_tags: list[str] | None  # 站点匹配结果
    platform_types: list[str] | None  # 平台类型列表
    is_single_platform: bool  # 是否单站点
    site_count: int  # 站点数量
    has_sites: bool  # 是否有站点

    # 兜底 API 结果
    fallback_api_request: Any
    fallback_api_resp: Any
    fallback_result_count: int
    fallback_api_success: bool
    fallback_request_path: str | None
    fallback_request_body: str | None
    fallback_attempted: bool

    # 合并后的结果
    merged_api_resp: Any  # 合并后的 API 响应
    merged_result_count: int  # 合并后的结果数量

    # 请求路径/参数（用于前端展示）
    request_path: str | None
    request_body: str | None
    final_request_path: str | None
    final_request_body: str | None

    # 后处理结果
    processed_goods_list: list[dict[str, Any]] | None
    goods_labels: list[dict[str, Any]] | None

    # 进度推送器
    progress_pusher: Any


# ============================================================
# Amazon/Temu 子工作流状态（SubGraph）
# ============================================================


class AmazonSubGraphState(TypedDict, total=False):
    """Amazon 子工作流状态 - 对应 n8n amazon选品子工作流v2"""

    # === 输入（从主工作流传入）===
    user_id: int
    team_id: int
    user_query: str
    param_result: Any  # AbroadGoodsParseParam
    sort_type: int  # 排序类型数字
    flag: int  # 1=监控台, 2=商品库
    new_type: str  # "新品" / "热销"
    title: str  # 任务标题
    shop_id: str | None  # 店铺ID（Amazon/Temu）
    platform_name_override: str | None  # 平台名称覆盖（Amazon/Temu）

    # === 站点判断（LLM）===
    platform_list: list[dict[str, Any]] | None  # 可选站点列表
    amazon_platform_type: int | None  # LLM 判断的站点类型

    # === 类目匹配（LLM）===
    origin_category_data: list[dict[str, Any]] | None  # 原站类目数据
    matched_category_id_list: list[list[str]] | None  # LLM 匹配的类目路径
    param_category_id_list: list[list[str]] | None  # 主工作流传入的类目

    # === API 调用 ===
    api_params: dict[str, Any] | None  # 构建的 API 参数
    api_resp: Any  # API 响应
    result_count: int
    api_success: bool
    request_path: str  # API 路径（用于前端展示）
    result_threshold: int  # 结果数阈值
    search_kind: str  # 搜索类型：monitor_new/monitor_hot/goods_list

    # === 兜底 ===
    fallback_api_resp: Any
    fallback_result_count: int
    fallback_params: dict[str, Any] | None
    fallback_request_path: str

    # === 输出 ===
    goods_list: list[dict[str, Any]] | None
    final_params: dict[str, Any] | None
    final_request_path: str


class IndependentSitesSubGraphState(TypedDict, total=False):
    """独立站子工作流状态"""

    # === 输入（从主工作流传入）===
    user_id: int
    team_id: int
    user_query: str
    param_result: Any  # AbroadGoodsParseParam
    sort_type: int
    title: str
    brand_tags: list[str] | None  # 品牌召回标签

    # === 查询解析（LLM）===
    parsed_query: Any | None  # QueryParseResult
    tag_text: str | None  # 构建的tag_text
    platform_type: str | None  # 单个平台类型

    # === 站点匹配 ===
    available_sites: list[dict[str, Any]] | None  # 可用站点列表
    match_tags: list[str] | None  # 匹配的站点标签列表
    platform_types: list[str] | None  # 平台类型列表

    # === 站点判断 ===
    site_count: int  # 站点数量
    has_sites: bool  # 是否有站点
    is_single_site: bool  # 是否单个站点

    # === API 调用 ===
    primary_params: dict[str, Any] | None  # 主 API 参数
    primary_api_resp: Any  # 主 API 响应
    primary_result_count: int
    primary_success: bool
    primary_request_path: str

    secondary_params: dict[str, Any] | None  # 兜底 API 参数
    secondary_api_resp: Any  # 兜底 API 响应
    secondary_result_count: int
    secondary_success: bool
    secondary_request_path: str

    # === 结果合并 ===
    merged_result_list: list[Any] | None  # 合并的结果列表
    merged_result_count: int  # 合并的结果计数
    used_primary_fallback: bool  # 是否使用了主API兜底

    # === 输出 ===
    goods_list: list[dict[str, Any]] | None
    final_params: dict[str, Any] | None
    final_request_path: str
    final_result_count: int  # 最终结果数（主请求或兜底请求）
    final_api_resp: Any  # 最终API响应（主请求或兜底请求）
    final_success: bool  # 最终是否成功
    used_fallback: bool  # 是否使用兜底
    fallback_attempted: bool  # 是否触发兜底


class TemuSubGraphState(TypedDict, total=False):
    """Temu 子工作流状态 - 对应 n8n temu选品子工作流v2"""

    # === 输入（从主工作流传入）===
    user_id: int
    team_id: int
    user_query: str
    param_result: Any  # AbroadGoodsParseParam
    sort_type: int
    flag: int  # 1=监控台, 2=商品库
    new_type: str  # "新品" / "热销"
    title: str
    shop_id: str | None  # 店铺ID（Amazon/Temu）
    platform_name_override: str | None  # 平台名称覆盖（Amazon/Temu）

    # === 站点判断（LLM）===
    platform_list: list[dict[str, Any]] | None
    temu_platform_type: int | None

    # === 类目匹配（LLM）===
    origin_category_data: list[dict[str, Any]] | None
    matched_category_id_list: list[list[str]] | None
    param_category_id_list: list[list[str]] | None  # 主工作流传入的类目

    # === API 调用 ===
    api_params: dict[str, Any] | None
    api_resp: Any
    result_count: int
    api_success: bool
    request_path: str
    result_threshold: int  # 结果数阈值
    search_kind: str  # 搜索类型：monitor_new/monitor_hot/goods_list

    # === 兜底 ===
    fallback_api_resp: Any
    fallback_result_count: int
    fallback_params: dict[str, Any] | None
    fallback_request_path: str

    # === 输出 ===
    goods_list: list[dict[str, Any]] | None
    final_params: dict[str, Any] | None
    final_request_path: str


class ImageWorkflowState(TypedDict, total=False):
    """图片生成工作流状态（文生图 / 图生图）"""

    # 输入
    request: WorkflowRequest

    # 文生图参数
    image_prompt: str | None  # 用户输入的提示词

    # 图生图参数
    input_images: list[str] | None  # 输入图片URL列表
    raw_input_images: str | list[str] | None  # 原始输入图片字段
    edit_prompt: str | None  # 编辑提示词

    # LLM 优化后的提示词
    optimized_prompt: str | None

    # 流程标识
    cost_id: str | None

    # 图片生成结果
    generated_image_base64: str | None  # Gemini 返回的 base64 图片
    output_image_url: str | None  # OSS 上传后的图片 URL

    # 重试控制
    retry_count: int  # 当前重试次数
    max_retries: int  # 最大重试次数

    # 最终输出
    workflow_response: Any


__all__ = [
    "MainOrchestratorState",
    "MediaWorkflowState",
    "ZxhXhsWorkflowState",
    "AbroadInsWorkflowState",
    "ZhikuanInsWorkflowState",
    "ChatWorkflowState",
    "DouyiWorkflowState",
    "ZhiyiWorkflowState",
    "SelectionWorkflowState",
    "AbroadGoodsWorkflowState",
    "AmazonSubGraphState",
    "TemuSubGraphState",
    "IndependentSitesSubGraphState",
    "ImageWorkflowState",
]
