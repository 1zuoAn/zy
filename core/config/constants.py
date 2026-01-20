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
    IMAGE_SEARCH = "image_search"  # 图搜
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
    ABROAD_INS_CATEGORY_PARSE_PROMPT = "abroad_ins_category_parse_prompt"
    ABROAD_INS_TIME_PARSE_PROMPT = "abroad_ins_time_parse_prompt"
    ABROAD_INS_BLOGGER_PARSE_PROMPT = "abroad_ins_blogger_parse_prompt"
    ABROAD_INS_STYLE_PARSE_PROMPT = "abroad_ins_style_parse_prompt"
    ABROAD_INS_MISC_PARSE_PROMPT = "abroad_ins_misc_parse_prompt"
    ABROAD_INS_SORT_TYPE_PARSE_PROMPT = "abroad_ins_sort_type_parse_prompt"

    # 知款ins
    ZHIKUAN_INS_THINK_PROMPT = "zhikuan_ins_think_prompt"
    ZHIKUAN_INS_CATEGORY_PARSE_PROMPT = "zhikuan_ins_category_parse_prompt"
    ZHIKUAN_INS_TIME_PARSE_PROMPT = "zhikuan_ins_time_parse_prompt"
    ZHIKUAN_INS_BLOGGER_PARSE_PROMPT = "zhikuan_ins_blogger_parse_prompt"
    ZHIKUAN_INS_STYLE_PARSE_PROMPT = "zhikuan_ins_style_parse_prompt"
    ZHIKUAN_INS_MISC_PARSE_PROMPT = "zhikuan_ins_misc_parse_prompt"
    ZHIKUAN_INS_SORT_TYPE_PARSE_PROMPT = "zhikuan_ins_sort_type_parse_prompt"

    # 小红书
    ZXH_XHS_THINK_PROMPT = 'zxh_xhs_think_prompt'
    ZXH_XHS_MAIN_PARAM_PARSE_PROMPT = 'zxh_xhs_main_param_parse_prompt'
    ZXH_XHS_SORT_TYPE_PARSE_PROMPT = 'zxh_xhs_sort_type_parse_prompt'
    ZXH_XHS_TOPIC_RAG_CLEAN_PROMPT = 'zxh_xhs_topic_rag_clean_prompt'

    # 抖衣
    DOUYI_THINK_PROMPT = 'douyi_think_prompt'
    DOUYI_MAIN_PARAM_PARSE_PROMPT = 'douyi_main_param_parse_prompt'
    DOUYI_CATEGORY_PARSE_PROMPT = 'douyi_category_parse_prompt'
    DOUYI_TIME_PARSE_PROMPT = 'douyi_time_parse_prompt'
    DOUYI_NUMERIC_PARSE_PROMPT = 'douyi_numeric_parse_prompt'
    DOUYI_SALES_FLAG_PARSE_PROMPT = 'douyi_sales_flag_parse_prompt'
    DOUYI_SORT_INTENT_PARSE_PROMPT = 'douyi_sort_intent_parse_prompt'
    DOUYI_PROPERTIES_PARSE_PROMPT = 'douyi_properties_parse_prompt'
    DOUYI_MISC_PARSE_PROMPT = 'douyi_misc_parse_prompt'
    DOUYI_SORT_TYPE_PARSE_PROMPT = 'douyi_sort_type_parse_prompt'
    DOUYI_USER_TAG_PARSE_PROMPT = 'douyi_user_tag_parse_prompt'
    DOUYI_PROPERTY_WRAP_PROMPT = 'douyi_property_wrap_prompt'
    DOUYI_CATEGORY_VECTOR_CLEAN_PROMPT = 'douyi_category_vector_clean_prompt'
    DOUYI_PROPERTY_CLEAN_PROMPT = "douyi_property_clean_prompt"

    # 知衣
    ZHIYI_THINK_PROMPT = 'zhiyi_think_prompt'
    ZHIYI_CATEGORY_PARSE_PROMPT = 'zhiyi_category_parse_prompt'
    ZHIYI_TIME_PARSE_PROMPT = 'zhiyi_time_parse_prompt'
    ZHIYI_NUMERIC_PARSE_PROMPT = 'zhiyi_numeric_parse_prompt'
    ZHIYI_ROUTE_PARSE_PROMPT = 'zhiyi_route_parse_prompt'
    ZHIYI_BRAND_PHRASE_PARSE_PROMPT = 'zhiyi_brand_phrase_parse_prompt'
    ZHIYI_PROPERTIES_PARSE_PROMPT = 'zhiyi_properties_parse_prompt'
    ZHIYI_SORT_TYPE_PARSE_PROMPT = 'zhiyi_sort_type_parse_prompt'
    ZHIYI_CATEGORY_VECTOR_CLEAN_PROMPT = 'zhiyi_category_vector_clean_prompt'
    ZHIYI_KB_FILTER_PROMPT = 'zhiyi_kb_filter_prompt'
    ZHIYI_USER_TAG_PARSE_PROMPT = 'zhiyi_user_tag_parse_prompt'
    ZHIYI_PROPERTY_CLEAN_PROMPT = "zhiyi_property_clean_prompt"
    ZHIYI_SHOP_CLEAN_PROMPT = "zhiyi_shop_clean_prompt"

    # 海外探款商品
    ABROAD_GOODS_THINK_PROMPT = 'abroad_goods_think_prompt'
    ABROAD_GOODS_MAIN_PARAM_PARSE_PROMPT = 'abroad_goods_main_param_parse_prompt'
    ABROAD_GOODS_CATEGORY_PARSE_PROMPT = 'abroad_goods_category_parse_prompt'
    ABROAD_GOODS_ATTRIBUTE_PARSE_PROMPT = 'abroad_goods_attribute_parse_prompt'
    ABROAD_GOODS_BRAND_PARSE_PROMPT = 'abroad_goods_brand_parse_prompt'
    ABROAD_GOODS_TIME_PARSE_PROMPT = 'abroad_goods_time_parse_prompt'
    ABROAD_GOODS_NUMERIC_PARSE_PROMPT = 'abroad_goods_numeric_parse_prompt'
    ABROAD_GOODS_PLATFORM_ROUTE_PARSE_PROMPT = 'abroad_goods_platform_route_parse_prompt'
    ABROAD_GOODS_REGION_PARSE_PROMPT = 'abroad_goods_region_parse_prompt'
    ABROAD_GOODS_TEXT_TITLE_PARSE_PROMPT = 'abroad_goods_text_title_parse_prompt'
    ABROAD_GOODS_SORT_TYPE_PARSE_PROMPT = 'abroad_goods_sort_type_parse_prompt'
    ABROAD_GOODS_STYLE_PARSE_PROMPT = 'abroad_goods_style_parse_prompt'
    ABROAD_GOODS_SITE_CLEAN_PROMPT = 'abroad_goods_site_clean_prompt'
    ABROAD_GOODS_BRAND_CLEAN_PROMPT = 'abroad_goods_brand_clean_prompt'
    ABROAD_GOODS_USER_TAG_PARSE_PROMPT = 'abroad_goods_user_tag_parse_prompt'

    # 图片设计
    IMAGE_PROMPT_OPTIMIZE = 'image_prompt_optimize'
    # 主工作流-选品/媒体路由
    MAIN_SELECT_AGENT_PROMPT = "main_select_agent_prompt"
    MAIN_MEDIA_AGENT_PROMPT = "main_media_agent_prompt"
    MAIN_SHOP_AGENT_PROMPT = "main_shop_agent_prompt"
    MAIN_CHAT_PROMPT = "main_chat_prompt"
    MAIN_SCHEDULE_PROMPT = "main_schedule_prompt"
    MAIN_IMAGE_DESIGN_PROMPT = "main_image_design_prompt"
    # 店铺排行
    SHOP_CATEGORY_TIME_PARSE_PROMPT = "shop_category_time_parse_prompt"
    SHOP_PLATFORM_TYPE_PROMPT = "shop_platform_type_prompt"
    SHOP_LABEL_TYPE_PROMPT = "shop_label_type_prompt"
    SHOP_SORT_PROMPT = "shop_sort_prompt"
    SHOP_STYLE_PROMPT = "shop_style_prompt"
    SHOP_SEARCH_KEY_PROMPT = "shop_search_key_prompt"

    # 数据洞察-知衣
    DR_ZHIYI_THINKING_PARAM_PARSE_PROMPT = "deepresearch_zhiyi_thinking_param_parse_prompt"
    DR_ZHIYI_THINKING_SHOP_CLEAN_PROMPT = "deepresearch_zhiyi_thinking_shop_clean_prompt"
    DR_ZHIYI_THINKING_PROPERTY_EXTRACT_PROMPT = "deepresearch_zhiyi_thinking_property_extract_prompt"
    DR_ZHIYI_THINKING_PRICE_EXTRACT_PROMPT = "deepresearch_zhiyi_thinking_price_extract_prompt"
    DR_ZHIYI_THINKING_SHOP_CATEGORY_REPORT_PROMPT = "deepresearch_zhiyi_thinking_shop_category_report_prompt"
    DR_ZHIYI_THINKING_SHOP_NORMAL_REPORT_PROMPT = "deepresearch_zhiyi_thinking_shop_normal_report_prompt"

    # 数据洞察-知衣大盘分析
    DR_ZHIYI_HYDC_DIMENSION_ANALYZE_PROMPT = "deepresearch_zhiyi_hydc_dimension_analyze_prompt"  # 维度分析(0=属性/1=颜色/2=品牌)
    DR_ZHIYI_HYDC_PROPERTY_EXTRACT_PROMPT = "deepresearch_zhiyi_hydc_property_extract_prompt"  # 大盘属性提取
    DR_ZHIYI_HYDC_CATEGORY_REPORT_PROMPT = "deepresearch_zhiyi_hydc_category_report_prompt"  # 大盘品类报告
    DR_ZHIYI_HYDC_NORMAL_REPORT_PROMPT = "deepresearch_zhiyi_hydc_normal_report_prompt"  # 大盘通用报告

    # 数据洞察-抖衣
    DR_DOUYI_MAIN_PARAM_PARSE_PROMPT = "deepresearch_douyi_main_param_parse_prompt"
    DR_DOUYI_PROPERTY_EXTRACT_PROMPT = "deepresearch_douyi_property_extract_prompt"  # 属性提取
    DR_DOUYI_PRICE_EXTRACT_PROMPT = "deepresearch_douyi_price_extract_prompt"  # 价格带提取
    DR_DOUYI_REPORT_GENERATE_PROMPT = "deepresearch_douyi_report_generate_prompt"  # 报告生成
    DR_DOUYI_NONTHINKING_REPORT_GENERATE_PROMPT = "deepresearch_douyi_nonthinking_report_generate_prompt" # 非深度思考报告生成

    # 数据洞察-海外探款
    DR_ABROAD_MAIN_PARAM_PARSE_PROMPT = "deepresearch_abroad_main_param_parse_prompt"  # 参数解析
    DR_ABROAD_SITE_JUDGE_PROMPT = "deepresearch_abroad_site_judge_prompt"  # 站点判断
    DR_ABROAD_DIMENSION_ANALYZE_PROMPT = "deepresearch_abroad_dimension_analyze_prompt"  # 维度分析
    DR_ABROAD_PROPERTY_EXTRACT_PROMPT = "deepresearch_abroad_property_extract_prompt"  # 属性提取
    DR_ABROAD_CATEGORY_REPORT_GENERATE_PROMPT = "deepresearch_abroad_category_report_generate_prompt"
    DR_ABROAD_NORMAL_REPORT_GENERATE_PROMPT = "deepresearch_abroad_normal_report_generate_prompt"  # 报告生成
    DR_ABROAD_CATEGORY_NONTHINKING_REPORT_GENERATE_PROMPT = "deepresearch_abroad_category_nonthinking_report_generate_prompt"
    DR_ABROAD_NORMAL_NONTHINKING_REPORT_GENERATE_PROMPT = "deepresearch_abroad_normal_nonthinking_report_generate_prompt"



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
