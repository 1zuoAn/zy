from dataclasses import dataclass
from enum import Enum


class WorkflowType(str, Enum):
    """工作流类型枚举"""
    SELECT = "select"
    IMAGE_DESIGN = "image_design"
    TRENDS = "trends"
    MEDIA = "media"
    MEDIA_ABROAD_INS = "media_abroad_ins"
    MEDIA_ZHIKUAN_INS = "media_zhikuan_ins"
    MEDIA_ZXH_XHS = "media_zxh_xhs"
    SELECT_ZHIYI = "select_zhiyi"  # 知衣选品
    SELECT_ABROAD_GOODS = "select_abroad_goods"  # 海外探款商品
    SELECT_DOUYI = "select_douyi"  # 抖衣(抖音)选品
    SHOP = "shop"
    SCHEDULE = "schedule"
    CHAT = "chat"
    # 图片设计工作流
    IMAGE_CREATE = "image_create"  # 文生图
    IMAGE_EDIT = "image_edit"  # 图生图

class WorkflowMessageContentType(int, Enum):
    # 处理过程
    PROCESSING = 1
    # 预处理文本
    PRE_TEXT = 2
    # 文本
    TEXT = 3
    # 思考过程
    THINKING = 4
    # 结果
    RESULT = 5
    # 筛选项
    SELECTION = 6
    # 多选
    MULTI_SELECTION = 7
    # 查询数据来源
    QUERY_DATA_SOURCE = 8
    # 任务状态
    TASK_STATUS = 9
    # 扣费退款
    COST_REFOUND = 10

class WorkflowEntityType(Enum):
    TAOBAO_ITEM = (1, "淘宝商品")
    DY_ITEM = (2, "抖衣商品")
    ABROAD_ITEM = (3, "海外探款商品")
    ABROAD_INS = (6, "海外探款INS帖子")
    ZHIKUAN_INS = (7, "知款INS帖子")
    XHS_NOTE = (8, "小红书笔记")
    TB_SHOP_RANK = (9, "淘宝店铺排名")
    IMAGE_SEARCH = (10, "知衣图搜")
    ABROAD_IMAGE_SEARCH = (11, "海外探款图搜")

    def __init__(self, code: int, desc: str):
        self.code = code
        self.desc = desc


class LlmProvider(Enum):
    """
    大模型api供应商枚举
    """
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    HUANXIN = "huanxin"

class LlmModelName(Enum):
    """
    供应商内部定义的大模型调用名称
    """
    # open router模型
    OPENROUTER_GPT_4_1 = "openai/gpt-4.1"
    OPENROUTER_GPT_4O = "openai/gpt-4o"
    OPENROUTER_GPT_4O_MINI = "openai/gpt-4o-mini"
    OPENROUTER_GPT_O3 = "openai/o3"
    OPENROUTER_CLAUDE_3_5_SONNET = "anthropic/claude-3-5-sonnet"
    OPENROUTER_GEMINI_3_FLASH_PREVIEW = "google/gemini-3-flash-preview"
    OPENROUTER_GEMINI_3_PRO_PREVIEW = "google/gemini-3-pro-preview"
    OPENROUTER_GEMINI_2_5_PRO = "google/gemini-2.5-pro"
    OPENROUTER_GEMINI_2_5_FLASH = "google/gemini-2.5-flash"
    OPENROUTER_GEMINI_2_5_FLASH_LITE = "google/gemini-2.5-flash-lite"
    OPENROUTER_GEMINI_3_PRO_IMAGE = "google/gemini-3-pro-image-preview"

    # 寰信模型
    HUANXIN_GPT_4O = "gpt-4o"
    HUANXIN_GPT_5 = "gpt-5"
    HUANXIN_O3 = "o3"
    HUANXIN_GEMINI_2_5_FLASH = "gemini-2.5-flash"
    HUANXIN_GEMINI_3_FLASH_PREVIEW = "gemini-3-flash-preview"
    HUANXIN_GROK_4_FAST_NON_REASONING = "grok-4-fast-non-reasoning"
    HUANXON_GROK_4_1_FAST_REASONING = "grok-4-1-fast-reasoning"
    HUANXIN_GROK_4_1_FAST_NON_REASONING = "grok-4-1-fast-non-reasoning"


class EmbeddingModelName(Enum):
    """
    供应商内部定义的向量模型名称
    """
    DASHSCOPE_TEXT_EMBEDDING_V4 = "text-embedding-v4"


class DBAlias(Enum):
    """
    数据源枚举
    """
    OLAP_ZXY_AGENT = "OLAP_ZXY_AGENT"
    B = "b"
    DB_ABROAD_AI = "DB_ABROAD_AI"


class RedisMessageKeyName(Enum):
    """
    redis消息推送key名称
    """
    AI_CONVERSATION_MESSAGE_QUEUE = "ai-conversation-agent-message-queue"

class CozePromptHubKey(Enum):
    # 主工作流
    MAIN_INTENT_CLASSIFY_PROMPT = "main_intent_classify_prompt"
    MAIN_SUMMARY_PROMPT = "main_summary_prompt"
    
    # 跨境ins
    ABROAD_INS_THINK_PROMPT = 'abroad_ins_think_prompt'
    ABROAD_INS_MAIN_PARAM_PARSE_PROMPT = "abroad_ins_main_param_parse_prompt"
    ABROAD_INS_SORT_TYPE_PARSE_PROMPT = "abroad_ins_sort_type_parse_prompt"

    # 知款ins
    ZHIKUAN_INS_THINK_PROMPT = "zhikuan_ins_think_prompt"
    ZHIKUAN_INS_MAIN_PARAM_PARSE_PROMPT = "zhikuan_ins_main_parse_prompt"
    ZHIKUAN_INS_SORT_TYPE_PARSE_PROMPT = "zhikuan_ins_sort_type_parse_prompt"

    # 小红书
    ZXH_XHS_THINK_PROMPT = 'zxh_xhs_think_prompt'
    ZXH_XHS_MAIN_PARAM_PARSE_PROMPT = 'zxh_xhs_main_param_parse_prompt'
    ZXH_XHS_SORT_TYPE_PARSE_PROMPT = 'zxh_xhs_sort_type_parse_prompt'
    ZXH_XHS_TOPIC_RAG_CLEAN_PROMPT = 'zxh_xhs_topic_rag_clean_prompt'

    # 抖衣
    DOUYI_THINK_PROMPT = 'douyi_think_prompt'
    DOUYI_MAIN_PARAM_PARSE_PROMPT = 'douyi_main_param_parse_prompt'
    DOUYI_SORT_TYPE_PARSE_PROMPT = 'douyi_sort_type_parse_prompt'
    DOUYI_USER_TAG_PARSE_PROMPT = 'douyi_user_tag_parse_prompt'
    DOUYI_PROPERTY_WRAP_PROMPT = 'douyi_property_wrap_prompt'
    DOUYI_CATEGORY_VECTOR_CLEAN_PROMPT = 'douyi_category_vector_clean_prompt'
    DOUYI_PROPERTY_CLEAN_PROMPT = "douyi_property_clean_prompt"

    # 知衣
    ZHIYI_THINK_PROMPT = 'zhiyi_think_prompt'
    ZHIYI_MAIN_PARAM_PARSE_PROMPT = 'zhiyi_main_param_parse_prompt'
    ZHIYI_SORT_TYPE_PARSE_PROMPT = 'zhiyi_sort_type_parse_prompt'
    ZHIYI_CATEGORY_VECTOR_CLEAN_PROMPT = 'zhiyi_category_vector_clean_prompt'
    ZHIYI_KB_FILTER_PROMPT = 'zhiyi_kb_filter_prompt'
    ZHIYI_USER_TAG_PARSE_PROMPT = 'zhiyi_user_tag_parse_prompt'
    ZHIYI_PROPERTY_CLEAN_PROMPT = "zhiyi_property_clean_prompt"
    ZHIYI_SHOP_CLEAN_PROMPT = "zhiyi_shop_clean_prompt"

    # 海外探款商品
    ABROAD_GOODS_THINK_PROMPT = 'abroad_goods_think_prompt'
    ABROAD_GOODS_MAIN_PARAM_PARSE_PROMPT = 'abroad_goods_main_param_parse_prompt'
    ABROAD_GOODS_SORT_TYPE_PARSE_PROMPT = 'abroad_goods_sort_type_parse_prompt'
    ABROAD_GOODS_STYLE_PARSE_PROMPT = 'abroad_goods_style_parse_prompt'
    ABROAD_GOODS_SITE_CLEAN_PROMPT = 'abroad_goods_site_clean_prompt'
    ABROAD_GOODS_BRAND_CLEAN_PROMPT = 'abroad_goods_brand_clean_prompt'

    # 图片设计
    IMAGE_PROMPT_OPTIMIZE = 'image_prompt_optimize'



class ZhiyiDataSourceKey(str, Enum):
    """知衣数据源Key - 对应n8n的zy-item-*系列

    映射规则:
    - zy-item-hot: 热销 + 全网数据 (flag=2, type=热销)
    - zy-item-monitor-hot: 热销 + 监控店铺 (flag=1, type=热销)
    - zy-item-all: 新品 + 全网数据 (flag=2, type=新品)
    - zy-item-monitor-new: 新品 + 监控店铺 (flag=1, type=新品)
    - zy-item-shop: 品牌店铺查询，需要带query_params (shopId)
    """
    HOT = "zy-item-hot"
    MONITOR_HOT = "zy-item-monitor-hot"
    ALL = "zy-item-all"
    MONITOR_NEW = "zy-item-monitor-new"
    SHOP = "zy-item-shop"


# 火山引擎知识库服务配置
class VolcKnowledgeServiceId(str, Enum):
    """火山知识库服务别名"""
    # 通用
    ZHIYI_KNOWLEDGE = "kb-service-928d084fab43cacb"  # 知小衣知识库检索
    # 知小红
    XHS_TOPIC_KNOWLEDGE = "kb-service-301c9b98ec29a9"
    # 知衣
    ZHIYI_SHOP_KNOWLEDGE = "kb-service-df48b3874ba53282" # 知衣店铺向量检索
    ZHIYI_CATEGORY_VECTOR = "kb-service-e74e6348d0eb338b"  # 知衣类目向量检索
    ZHIYI_PROPERTIES_VECTOR = "kb-service-6569280e7104c306" # 知衣属性向量检索
    # 抖衣
    DOUYI_CATEGORY_VECTOR = "kb-service-5a86eb6285a42b39"  # 抖衣类目向量检索
    DOUYI_PROPERTIES_VECTOR = "kb-service-c9f6fab4f3104255" # 抖衣属性向量检索
    # 海外探款
    ABROAD_CATEGORY_VECTOR = "kb-service-12566035fe3cd7bf"  # 海外探款通用类目向量检索
    ABROAD_SITE_VECTOR = "kb-service-506e215b9f9293ef"  # 海外探款已上线站点检索
    ABROAD_SITE_BRAND_VECTOR = "kb-service-c22c942c77fac32d" # 海外探款
    AMAZON_CATEGORY_VECTOR = "kb-service-8d5619997b85abec"  # Amazon 原站类目向量检索
    TEMU_CATEGORY_VECTOR = "kb-service-db419d13fbc98b14"  # Temu 原站类目向量检索
