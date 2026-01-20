# @Author   : kiro
# @Time     : 2025/12/22
# @File     : amazon_subgraph.py

"""
Amazon 子工作流 - LangGraph SubGraph

对应 n8n: amazon选品子工作流v2

完整版：包含 LLM 站点判断和类目匹配
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from app.core.config.constants import LlmModelName, LlmProvider, VolcKnowledgeServiceId
from app.core.tools import llm_factory
from app.schemas.entities.workflow.graph_state import AmazonSubGraphState
from app.service.rpc.abroad_api import get_abroad_api
from app.utils.abroad_helpers import (
    extract_json_from_llm_output,
    normalize_category_list,
    normalize_label_value,
)
from app.service.rpc.volcengine_kb_api import (
    KBDocFilter,
    KBMessage,
    KBQueryParam,
    get_volcengine_kb_api,
)

# ==================== LLM 输出结构 ====================


class PlatformTypeResult(BaseModel):
    """站点类型判断结果"""

    model_config = ConfigDict(populate_by_name=True)

    amazon_platform_type: int = Field(
        default=4, alias="amazonPlatformType", description="Amazon 站点类型 ID"
    )


def _normalize_platform_name(name: str) -> str:
    value = str(name or "").strip().lower()
    if not value:
        return ""
    value = value.replace("亚马逊", "amazon")
    value = re.sub(r"[\s\-_\\/()（）\[\]{}]+", "", value)
    return value


def _match_platform_type_by_name(
    platform_list: list[dict[str, Any]], platform_name: str
) -> int | None:
    target = _normalize_platform_name(platform_name)
    if not target:
        return None
    for item in platform_list:
        name = item.get("platform_name") or ""
        if _normalize_platform_name(name) == target:
            try:
                return int(item.get("platform_type"))
            except (TypeError, ValueError):
                return None
    best_type = None
    best_score = 0
    for item in platform_list:
        name = item.get("platform_name") or ""
        normalized = _normalize_platform_name(name)
        if not normalized:
            continue
        if target in normalized or normalized in target:
            score = len(normalized)
            if score > best_score:
                try:
                    best_type = int(item.get("platform_type"))
                except (TypeError, ValueError):
                    best_type = None
                best_score = score
    return best_type


class CategoryMatchResult(BaseModel):
    """类目匹配结果"""

    model_config = ConfigDict(populate_by_name=True)

    category_id_list: list[list[str]] = Field(
        default_factory=list,
        alias="categoryIdList",
        description="匹配的类目 ID 路径列表",
    )


# ==================== SubGraph ====================


class AmazonSubGraph:
    """Amazon 子工作流 - 完整版"""

    RESULT_THRESHOLD = 10  # 默认结果数阈值

    def __init__(self) -> None:
        self._graph: CompiledStateGraph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        """构建子工作流图"""
        graph = StateGraph(AmazonSubGraphState)  # type: ignore[arg-type]

        # === 节点 ===
        graph.add_node("init", self._init_node)  # type: ignore[arg-type]
        graph.add_node("llm_platform", self._llm_platform_node)  # type: ignore[arg-type]
        graph.add_node("llm_category", self._llm_category_node)  # type: ignore[arg-type]
        graph.add_node("monitor_new", self._monitor_new_node)  # type: ignore[arg-type]
        graph.add_node("monitor_hot", self._monitor_hot_node)  # type: ignore[arg-type]
        graph.add_node("goods_list", self._goods_list_node)  # type: ignore[arg-type]
        graph.add_node("fallback", self._fallback_node)  # type: ignore[arg-type]
        graph.add_node("output", self._output_node)  # type: ignore[arg-type]

        # === 边 ===
        graph.set_entry_point("init")
        graph.add_edge("init", "llm_platform")
        graph.add_edge("llm_platform", "llm_category")

        # 路由：flag=1 监控台, flag=2 商品库
        graph.add_conditional_edges(
            "llm_category",
            self._route_by_flag,
            {
                "monitor": "route_monitor_type",
                "goods": "goods_list",
            },
        )

        # 监控台子路由
        graph.add_node("route_monitor_type", lambda s: s)
        graph.add_conditional_edges(
            "route_monitor_type",
            self._route_by_new_type,
            {"new": "monitor_new", "hot": "monitor_hot"},
        )

        # 结果检查 → 兜底或输出
        for node in ["monitor_new", "monitor_hot", "goods_list"]:
            graph.add_conditional_edges(
                node,
                self._check_result,
                {"ok": "output", "fallback": "fallback"},
            )

        graph.add_edge("fallback", "output")
        graph.add_edge("output", END)

        return graph.compile()

    def run(self, state: AmazonSubGraphState) -> AmazonSubGraphState:
        """执行子工作流"""
        return self._graph.invoke(state)  # type: ignore[return-value]

    # ==================== 路由函数 ====================

    def _route_by_flag(self, state: AmazonSubGraphState) -> str:
        flag = state.get("flag", 2)
        return "monitor" if flag == 1 else "goods"

    def _route_by_new_type(self, state: AmazonSubGraphState) -> str:
        new_type = state.get("new_type", "")
        return "new" if new_type == "新品" else "hot"

    def _check_result(self, state: AmazonSubGraphState) -> str:
        result_count = state.get("result_count", 0)
        api_success = state.get("api_success", True)
        threshold = state.get("result_threshold", self.RESULT_THRESHOLD)
        if not api_success or result_count < threshold:
            return "fallback"
        return "ok"

    # ==================== 节点实现 ====================

    def _init_node(self, state: AmazonSubGraphState) -> dict[str, Any]:
        """初始化 - 获取可选站点列表"""
        api = get_abroad_api()
        platform_list = api.get_amazon_platforms()
        logger.debug(f"[Amazon-SubGraph] 获取到 {len(platform_list)} 个可选站点")
        return {"platform_list": platform_list}

    def _llm_platform_node(self, state: AmazonSubGraphState) -> dict[str, Any]:
        """LLM 判断站点类型 - 对齐 n8n 的 '判断amazon站点' 节点"""
        override = state.get("amazon_platform_type")
        if override is not None:
            override_str = str(override).strip()
            if override_str:
                platform_type = int(override_str) if override_str.isdigit() else override
                logger.debug(f"[Amazon-SubGraph] 使用引用站点类型: {platform_type}")
                return {"amazon_platform_type": platform_type}
        platform_list = state.get("platform_list", [])
        platform_name_override = state.get("platform_name_override")
        if platform_name_override and platform_list:
            matched_type = _match_platform_type_by_name(platform_list, platform_name_override)
            if matched_type is not None:
                logger.debug(f"[Amazon-SubGraph] 使用引用站点名称匹配: {matched_type}")
                return {"amazon_platform_type": matched_type}
        user_query = state.get("user_query", "")

        # 如果没有站点列表或用户查询，使用默认值
        if not platform_list or not user_query:
            return {"amazon_platform_type": 4}  # 默认美国站

        try:
            llm: BaseChatModel = llm_factory.get_llm(
                LlmProvider.OPENROUTER.name, LlmModelName.OPENROUTER_GEMINI_2_5_FLASH.value
            )
            structured_llm = llm.with_structured_output(PlatformTypeResult)

            # 对齐 n8n 的 system prompt
            system_prompt = """## 目标
你的任务是从提供的站点列表中匹配用户输入中最匹配的那一个。

## 输入结构
【可选站点列表】中表示的是可以与用户的输入进行匹配的候选站点列表，其中platform_type表示站点的id，platform_name表示站点的名称。
【用户输入】表示的是用户的选择倾向，其中可能包含不相关的信息，只需要关注用户对于站点的偏好即可。

## 默认行为
如果用户的输入与候选的站点无关或者没有表达站点的倾向，则参考【默认选择】中的信息。

## 输出结构
以json结构输出。
输出一个，且仅有一个最匹配的站点id，输出为amazonPlatformType字段。不可以为空。
{{"amazonPlatformType": 4}}"""

            platforms_str = json.dumps(platform_list, ensure_ascii=False)
            user_message = f"""# 可选站点列表
{platforms_str}

# 用户输入
{user_query}

# 默认选择
platform_type：4
platform_name：Amazon美国站"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]

            result = structured_llm.invoke(messages)
            platform_type = result.amazon_platform_type if result else 4
            logger.debug(f"[Amazon-SubGraph] LLM 判断站点类型: {platform_type}")
            return {"amazon_platform_type": platform_type}

        except Exception as e:
            logger.warning(f"[Amazon-SubGraph] LLM 站点判断失败: {e}")
            return {"amazon_platform_type": 4}

    def _llm_category_node(self, state: AmazonSubGraphState) -> dict[str, Any]:
        """LLM 匹配原站类目 - 对齐 n8n 的 '重新匹配原站类目' 节点

        n8n 逻辑:
        1. 调用 '海外探款amazon原站类目检索' 子工作流 (kb-service-8d5619997b85abec)
        2. 传入 tag_text, origin_query, platform_type
        3. 用 LLM 清洗召回结果
        4. 用 LLM 匹配类目
        """
        user_query = state.get("user_query", "")
        param_result = state.get("param_result")
        platform_type = state.get("amazon_platform_type", 4)

        param_category_id_list = param_result.category_id_list if param_result else []

        if not user_query:
            return {
                "matched_category_id_list": [],
                "param_category_id_list": param_category_id_list,
            }

        # Step 1: 调用火山知识库向量检索 - 对齐 n8n 的 '海外探款amazon原站类目检索' 子工作流
        try:
            kb_client = get_volcengine_kb_api()
            messages = [KBMessage(role="user", content=user_query)]
            # 按 platform_type 过滤
            doc_filter = KBDocFilter(op="must", field="platform_type", conds=[platform_type])
            query_param = KBQueryParam(doc_filter=doc_filter)

            response = kb_client.chat(
                messages=messages,
                service_resource_id=VolcKnowledgeServiceId.AMAZON_CATEGORY_VECTOR.value,
                query_param=query_param,
            )

            if not response.data or not response.data.result_list:
                logger.debug("[Amazon-SubGraph] 向量检索无结果")
                return {
                    "matched_category_id_list": [],
                    "param_category_id_list": param_category_id_list,
                }

            # Step 2: 提取检索结果 - 对齐 n8n 的 '提取检索结果2' 节点
            content_list = []
            for item in response.data.result_list:
                if item.table_chunk_fields:
                    key, value = "", ""
                    for chunk in item.table_chunk_fields:
                        field_name = chunk.get("field_name", "") if isinstance(chunk, dict) else ""
                        field_value = (
                            chunk.get("field_value", "") if isinstance(chunk, dict) else ""
                        )
                        if field_name == "key":
                            key = field_value
                        elif field_name == "value":
                            value = field_value
                    if key or value:
                        content_list.append(f"{key}#{value}")

            if not content_list:
                logger.debug("[Amazon-SubGraph] 提取检索结果为空")
                return {
                    "matched_category_id_list": [],
                    "param_category_id_list": param_category_id_list,
                }

            # Step 3: LLM 清洗召回文本 - 对齐 n8n 的 '清洗召回文本' 节点
            llm: BaseChatModel = llm_factory.get_llm(
                LlmProvider.OPENROUTER.name, LlmModelName.OPENROUTER_GEMINI_2_5_FLASH.value
            )

            clean_system_prompt = """# 角色
你是一个用户意图分析专家。

# 目标
分析用户的原始查询意图，对召回的标签文本进行分析，清洗去除用户不需要或者排除的内容，只保留用户明确需要的内容。

# 重要
召回文本的格式是"类目名称路径#类目ID路径"，你必须保留完整的原始格式，不要拆分或修改。

# 输出格式（JSON）
请输出一个json对象，不要包含其他无关内容
{
  "content_list": [
    "女装,女士T恤#28,1891",
    "女装#28"
  ]
}

# 输出限制
只输出json，不要输出其他的内容。"""

            clean_user_prompt = f"""用户原始查询意图：
{user_query}
召回文本：
{json.dumps(content_list, ensure_ascii=False)}"""

            clean_result = llm.invoke(
                [
                    SystemMessage(content=clean_system_prompt),
                    HumanMessage(content=clean_user_prompt),
                ]
            )

            # 解析清洗结果
            clean_content = (
                clean_result.content if hasattr(clean_result, "content") else str(clean_result)
            )
            extracted_list = extract_json_from_llm_output(clean_content, "content_list")
            if extracted_list and isinstance(extracted_list, list):
                content_list = extracted_list

            if not content_list:
                return {
                    "matched_category_id_list": [],
                    "param_category_id_list": param_category_id_list,
                }

            # Step 4: LLM 匹配类目 - 对齐 n8n 的 '重新匹配原站类目' 节点
            structured_llm = llm.with_structured_output(CategoryMatchResult)

            match_system_prompt = """# 角色
你是一个商品类目分析大师，能够精确的分析用户对于商品类目的需求。

# 任务
在【候选类目】中包含了一个电商站点所有的可选类目。每个元素是一个类目路径，#前是类目的名称路径，#后是类目的id路径。在【用户输入】中是用户对于选款的需求，可能不包含类目信息，你不需要关注其中与类目信息无关的部分，如品牌、热销等内容，均与你的任务无关。
你的任务是从【候选类目】中匹配【用户输入】中包含的一个或多个最匹配的类目，并将类目id路径按指定结构输出。整体输出到categoryIdList字段中，每个匹配到类目路径作为二维数组中的子输出单独输出。
如果用户输入中没有表达对类目的需求信息，则输出为空数组。
注意匹配的精准性，不要输出无关的类目结果。

# 技巧
- 在查找具体品类的时候，可以适当的忽视行业条件，例如【男士羽绒服】，可以出现在【体育服装-男士-羽绒衣】下，也是可选的类目。不强制要求比如为【男士-羽绒服】

# 输出结构
以json结构输出，不包含无关内容。
{{
	"categoryIdList": [
		["fashion", "7147440011", "1040660"],
		["fashion", "7147440011", "7192394011", "7454926011"]
	]
}}"""

            categories_str = json.dumps(content_list, ensure_ascii=False)
            match_user_message = f"""# 候选类目
{categories_str}

# 用户输入
{user_query}"""

            result = structured_llm.invoke(
                [
                    SystemMessage(content=match_system_prompt),
                    HumanMessage(content=match_user_message),
                ]
            )

            category_list = result.category_id_list if result else []
            logger.debug(f"[Amazon-SubGraph] LLM匹配类目: {category_list}")
            return {
                "matched_category_id_list": category_list,
                "param_category_id_list": param_category_id_list,
            }

        except Exception as e:
            logger.warning(f"[Amazon-SubGraph] 类目向量检索/匹配失败: {e}")
            return {
                "matched_category_id_list": [],
                "param_category_id_list": param_category_id_list,
            }

    def _build_api_params(
        self, state: AmazonSubGraphState, category_id_list_override: list[list[str]] | None = None
    ) -> dict[str, Any]:
        """构建 API 请求参数 - 对齐 n8n 的 'amazon 商品库参数' 节点

        参数包含:
        - designElementIndustry: "CLOTH"
        - categoryIdList: 优先使用 LLM 匹配的类目，其次使用参数解析的类目
        - 其他筛选参数: text, label, minSprice, maxSprice, brandList 等

        兜底时的参数简化由 _fallback_node 统一处理。
        """
        param_result = state.get("param_result")
        sort_type = state.get("sort_type", 8)
        platform_type = state.get("amazon_platform_type", 4)
        matched_category = (
            category_id_list_override
            if category_id_list_override is not None
            else state.get("matched_category_id_list")
        )

        on_sale_flag = getattr(param_result, "on_sale_flag", None) if param_result else None

        params = {
            "sortType": sort_type,
            "pageSize": 40,
            "pageNo": 1,
            "platformType": platform_type,
            "categoryType": "origin",
            "designElementIndustry": "CLOTH",
            "resultCountLimit": 6000,
            "onSaleFlag": on_sale_flag,
        }

        # 设置 categoryIdList: 优先使用传入的 override（已根据 prefer_matched 计算好的类目）
        # matched_category 可能是 None（未传入）、[]（空列表）或有值（非空列表）
        if matched_category is not None:
            params["categoryIdList"] = matched_category
        elif param_result and param_result.category_id_list is not None:
            params["categoryIdList"] = normalize_category_list(param_result.category_id_list)
        else:
            params["categoryIdList"] = []

        if param_result:
            if param_result.min_sprice is not None:
                params["minSprice"] = int(param_result.min_sprice * 100)
            if param_result.max_sprice is not None:
                params["maxSprice"] = int(param_result.max_sprice * 100)
            if param_result.put_on_sale_start_date:
                params["putOnSaleStartDate"] = param_result.put_on_sale_start_date
                params["startDate"] = param_result.put_on_sale_start_date
            elif param_result.start_date:
                params["startDate"] = param_result.start_date
            if param_result.put_on_sale_end_date:
                params["putOnSaleEndDate"] = param_result.put_on_sale_end_date
                params["endDate"] = param_result.put_on_sale_end_date
            elif param_result.end_date:
                params["endDate"] = param_result.end_date
            params["label"] = normalize_label_value(param_result.label)
            params["text"] = param_result.text if param_result.text is not None else ""
            # 添加brand参数支持 - 用于品牌筛选（API期望brandList数组）
            if param_result.brand:
                if isinstance(param_result.brand, list):
                    brand_list = [str(item).strip() for item in param_result.brand if str(item).strip()]
                else:
                    brand_list = [str(param_result.brand).strip()]
                if brand_list:
                    params["brandList"] = brand_list
            if param_result.limit:
                params["resultCountLimit"] = param_result.limit

        return params

    def _get_category_id_list(
        self, state: AmazonSubGraphState, prefer_matched: bool = True
    ) -> list:
        """获取类目 ID 列表，用于 API 调用

        prefer_matched=True 时优先 matched_category_id_list，否则回退到 param_category_id_list。
        """
        param_category = state.get("param_category_id_list")
        matched_category = state.get("matched_category_id_list")
        if prefer_matched:
            return matched_category if matched_category is not None else (param_category or [])
        return param_category if param_category is not None else (matched_category or [])

    def _call_api_node(
        self,
        state: AmazonSubGraphState,
        api_method: str,
        request_path: str,
        result_threshold: int,
        search_kind: str,
        prefer_matched: bool = True,
    ) -> dict[str, Any]:
        """通用 API 调用节点

        prefer_matched=True 时优先使用 LLM 匹配的类目，False 时优先使用参数解析的类目。
        如果 API 调用失败或结果数不足，会触发兜底逻辑。
        """
        user_id, team_id = state.get("user_id"), state.get("team_id")
        category_id_list = self._get_category_id_list(state, prefer_matched=prefer_matched)
        params = self._build_api_params(state, category_id_list_override=category_id_list)
        shop_id = state.get("shop_id")
        if search_kind == "goods_list" and shop_id:
            shop_id_str = str(shop_id).strip()
            if shop_id_str:
                params["shopId"] = int(shop_id_str) if shop_id_str.isdigit() else shop_id_str
        logger.debug(f"[Amazon-SubGraph] {search_kind}, params={params}")

        api = get_abroad_api()
        try:
            api_func = getattr(api, api_method)
            result = api_func(str(user_id), str(team_id), params)
            count = result.result_count or len(result.result_list)
            logger.debug(f"[Amazon-SubGraph] {search_kind} API返回: count={count}, result_list长度={len(result.result_list)}")
            return {
                "api_params": params,
                "api_resp": result,
                "result_count": count,
                "api_success": True,
                "request_path": request_path,
                "result_threshold": result_threshold,
                "search_kind": search_kind,
            }
        except Exception as e:
            logger.error(f"[Amazon-SubGraph] {search_kind} 失败: {e}")
            return {
                "api_params": params,
                "api_resp": None,
                "result_count": 0,
                "api_success": False,
                "request_path": request_path,
                "result_threshold": result_threshold,
                "search_kind": search_kind,
            }

    def _monitor_new_node(self, state: AmazonSubGraphState) -> dict[str, Any]:
        """监控店铺 - 上新 - 对齐 n8n 的 'amazon监控店铺-上新' 节点

        对应 n8n 节点: "amazon监控店铺-上新"
        使用 prefer_matched=False，优先使用参数解析的类目（param_category_id_list）。
        对应 n8n 的 "补充监控参数" 节点逻辑。
        """
        return self._call_api_node(
            state,
            api_method="amazon_monitor_new",
            request_path="amazon/goods/monitor-shop-new-list",
            result_threshold=5,
            search_kind="monitor_new",
            prefer_matched=False,
        )

    def _monitor_hot_node(self, state: AmazonSubGraphState) -> dict[str, Any]:
        """监控店铺 - 热销 - 对齐 n8n 的 'amazon监控店铺-热销' 节点

        对应 n8n 节点: "amazon监控店铺-热销"
        使用 prefer_matched=False，优先使用参数解析的类目（param_category_id_list）。
        对应 n8n 的 "补充监控参数" 节点逻辑。
        """
        return self._call_api_node(
            state,
            api_method="amazon_monitor_hot",
            request_path="amazon/goods/monitor-shop-hot-list",
            result_threshold=10,
            search_kind="monitor_hot",
            prefer_matched=False,
        )

    def _goods_list_node(self, state: AmazonSubGraphState) -> dict[str, Any]:
        """商品库 - 对齐 n8n 的 'amazon商品库接口' 节点

        对应 n8n 节点: "amazon商品库接口"
        使用 prefer_matched=True，优先使用 LLM 匹配的类目。
        """
        return self._call_api_node(
            state,
            api_method="amazon_goods_list",
            request_path="amazon/goods/list",
            result_threshold=10,
            search_kind="goods_list",
            prefer_matched=True,
        )

    def _fallback_node(self, state: AmazonSubGraphState) -> dict[str, Any]:
        """兜底查询 - 对齐 n8n 的 '商品库参数化简' 节点

        对应 n8n 节点: "商品库参数化简" -> "amazon商品库接口1"

        兜底逻辑: 保留核心参数与类目/时间/brandList，移除 designElementIndustry、text、label、
        minSprice/maxSprice 等复杂筛选后重试。
        """
        user_id, team_id = state.get("user_id"), state.get("team_id")
        original_params = state.get("api_params", {})
        search_kind = state.get("search_kind", "goods_list")

        # 兜底参数: 保留核心参数，移除 designElementIndustry/text/label 等复杂筛选
        params = {
            "pageSize": 40,
            "pageNo": 1,
            "platformType": original_params.get("platformType", 4),
            "categoryType": "origin",
            "resultCountLimit": original_params.get("resultCountLimit", 6000),
            "onSaleFlag": original_params.get("onSaleFlag"),
        }

        # 保留类目和时间范围（对齐 n8n "商品库参数化简" 节点）
        if "categoryIdList" in original_params:
            params["categoryIdList"] = original_params.get("categoryIdList")
        if original_params.get("putOnSaleStartDate"):
            params["putOnSaleStartDate"] = original_params["putOnSaleStartDate"]
        if original_params.get("putOnSaleEndDate"):
            params["putOnSaleEndDate"] = original_params["putOnSaleEndDate"]
        if original_params.get("startDate"):
            params["startDate"] = original_params["startDate"]
        if original_params.get("endDate"):
            params["endDate"] = original_params["endDate"]
        if original_params.get("brandList"):
            params["brandList"] = original_params["brandList"]
        if original_params.get("shopId") is not None:
            params["shopId"] = original_params["shopId"]

        logger.debug(f"[Amazon-SubGraph] 兜底查询, params={params}")

        api = get_abroad_api()
        try:
            if search_kind == "monitor_new":
                result = api.amazon_monitor_new(str(user_id), str(team_id), params)
                request_path = "amazon/goods/monitor-shop-new-list"
            elif search_kind == "monitor_hot":
                result = api.amazon_monitor_hot(str(user_id), str(team_id), params)
                request_path = "amazon/goods/monitor-shop-hot-list"
            else:
                result = api.amazon_goods_list(str(user_id), str(team_id), params)
                request_path = "amazon/goods/list"
            count = result.result_count or len(result.result_list)
            return {
                "fallback_api_resp": result,
                "fallback_result_count": count,
                "fallback_params": params,
                "fallback_request_path": request_path,
            }
        except Exception as e:
            logger.error(f"[Amazon-SubGraph] 兜底失败: {e}")
            return {
                "fallback_api_resp": None,
                "fallback_result_count": 0,
                "fallback_params": params,
            }

    def _output_node(self, state: AmazonSubGraphState) -> dict[str, Any]:
        """输出节点"""
        api_resp = state.get("api_resp")
        fallback_resp = state.get("fallback_api_resp")
        fallback_params = state.get("fallback_params")
        fallback_attempted = fallback_params is not None
        request_params = fallback_params if fallback_attempted else state.get("api_params")
        request_path = (
            state.get("fallback_request_path") if fallback_attempted else state.get("request_path")
        )

        resp = fallback_resp if fallback_attempted else api_resp
        if resp and resp.result_list:
            goods_list = [
                {
                    "productId": g.goods_id,
                    "productName": g.goods_name,
                    "picUrl": g.goods_img,
                    "sprice": g.sprice,
                }
                for g in resp.result_list
            ]
        else:
            goods_list = []

        return {
            "goods_list": goods_list,
            "request_params": request_params,
            "request_path": request_path,
        }


# 单例
_amazon_subgraph: AmazonSubGraph | None = None


def get_amazon_subgraph() -> AmazonSubGraph:
    global _amazon_subgraph
    if _amazon_subgraph is None:
        _amazon_subgraph = AmazonSubGraph()
    return _amazon_subgraph


__all__ = ["AmazonSubGraph", "get_amazon_subgraph"]
