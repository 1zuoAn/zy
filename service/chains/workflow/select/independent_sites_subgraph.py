# @Author   : kiro
# @Time     : 2025/12/22
# @File     : independent_sites_subgraph.py

"""
独立站子工作流 - LangGraph SubGraph

对应 n8n: 主工作流中的独立站分支

完整版：包含向量知识库站点检索、LLM 清洗、双 API 调用策略（与 n8n 逻辑完全一致）
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from app.core.clients.coze_loop_client import coze_loop_client_provider
from app.core.config.constants import (
    CozePromptHubKey,
    LlmModelName,
    LlmProvider,
    VolcKnowledgeServiceId,
)
from app.utils.abroad_helpers import extract_json_from_llm_output
from app.core.tools import llm_factory
from app.schemas.entities.workflow.graph_state import IndependentSitesSubGraphState
from app.service.rpc.abroad_api import get_abroad_api
from app.service.rpc.volcengine_kb_api import KBMessage, get_volcengine_kb_api

# ==================== SubGraph ====================


class IndependentSitesSubGraph:
    """独立站子工作流 - 与 n8n 逻辑完全一致"""

    RESULT_THRESHOLD = 50  # 结果数阈值（n8n 中明确为 50）

    def __init__(self) -> None:
        self._graph: CompiledStateGraph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        """构建子工作流图 - 严格按照 n8n 节点顺序"""
        graph = StateGraph(IndependentSitesSubGraphState)  # type: ignore[arg-type]

        # === 节点（与 n8n 一一对应）===
        graph.add_node("match_sites", self._match_sites_node)  # type: ignore[arg-type]
        graph.add_node("process_platforms", self._process_platforms_node)  # type: ignore[arg-type]
        graph.add_node("call_primary_api", self._call_primary_api_node)  # type: ignore[arg-type]
        graph.add_node("call_secondary_api", self._call_secondary_api_node)  # type: ignore[arg-type]
        graph.add_node("output", self._output_node)  # type: ignore[arg-type]

        # === 边 ===
        graph.set_entry_point("match_sites")
        graph.add_edge("match_sites", "process_platforms")
        graph.add_edge("process_platforms", "call_primary_api")

        # 主 API 结果判断 → 兜底或输出
        graph.add_conditional_edges(
            "call_primary_api",
            self._check_primary_result,
            {"ok": "output", "fallback": "call_secondary_api"},
        )

        # 兜底后直接进入输出
        graph.add_edge("call_secondary_api", "output")

        graph.add_edge("output", END)

        return graph.compile()

    def run(self, state: IndependentSitesSubGraphState) -> IndependentSitesSubGraphState:
        """执行子工作流"""
        return self._graph.invoke(state)  # type: ignore[return-value]

    # ==================== 路由函数 ====================

    def _check_primary_result(self, state: IndependentSitesSubGraphState) -> str:
        """判断主 API 结果 - 对齐 n8n 的 '详情结果判断' 节点

        对应 n8n 节点: "详情结果判断"
        n8n 条件: resultCountMax < 50 OR success == false → 触发兜底

        返回:
            "ok": 结果充足，直接输出
            "fallback": 结果不足或失败，触发兜底 API
        """
        result_count = state.get("primary_result_count", 0)
        api_success = state.get("primary_success", True)

        if not api_success or result_count < self.RESULT_THRESHOLD:
            logger.debug(
                f"[IndependentSites-SubGraph] 主API结果不足，触发兜底: {result_count} < {self.RESULT_THRESHOLD}"
            )
            return "fallback"

        logger.debug(f"[IndependentSites-SubGraph] 主API结果充足: {result_count}")
        return "ok"

    # ==================== 节点实现 ====================

    def _match_sites_node(self, state: IndependentSitesSubGraphState) -> dict[str, Any]:
        """站点匹配 - 对齐 n8n 的 'Call 海外探款已上线站点检索' 节点

        对应 n8n 节点: "Call '海外探款已上线站点检索'"

        n8n 逻辑流程:
        1. 构建 tag_text: type_platform_countryList
        2. 调用火山知识库向量检索 (kb-service-12566035fe3cd7bf)
        3. 提取结果: 从 table_chunk_fields 中提取 platform_type 和 platform_name
        4. LLM 清洗召回文本: 使用 CozePromptHubKey.ABROAD_GOODS_SITE_CLEAN_PROMPT
        5. 返回 match_tags: 格式为 "platformType,platformName" 的列表

        返回:
            match_tags: 匹配到的站点列表，格式为 ["platformType,platformName", ...]
            tag_text: 用于检索的标签文本
        """
        param_result: Any = state.get("param_result")

        # 如果已有 match_tags，直接返回
        match_tags_cached: Any = state.get("match_tags")
        if match_tags_cached:
            return {
                "match_tags": match_tags_cached,
            }

        # 优先使用传入的 tag_text（来自主工作流）
        tag_text: str = state.get("tag_text", "")
        if not tag_text:
            # 回退：从 param_result 构建 tag_text
            # 对应 n8n: $json.type_$json.platform_$json.countryList
            platform: str = ""
            country_list: str = ""
            type_value: str = "独立站"
            if param_result:
                platform = str(getattr(param_result, "platform", "") or "")
                # 使用 country_list (对应 n8n 的 countryList)
                country_list: Any = getattr(param_result, "country_list", []) or []
                type_value = str(getattr(param_result, "type", "") or "独立站")

            if isinstance(country_list, str):
                country_list_str = country_list
            elif isinstance(country_list, list):
                country_list_str = ",".join(country_list) if country_list else ""
            else:
                country_list_str = ""

            tag_text = f"{type_value}_{platform}_{country_list_str}"

        logger.debug(f"[IndependentSites-SubGraph] 构建 tag_text: {tag_text}")

        try:
            # Step 1: 调用火山知识库向量检索 - 对齐 n8n 的 '海外探款已上线站点检索' 子工作流
            kb_client = get_volcengine_kb_api()
            messages = [KBMessage(role="user", content=tag_text)]

            response = kb_client.chat(
                messages=messages,
                service_resource_id=VolcKnowledgeServiceId.ABROAD_SITE_VECTOR.value,
            )

            if not response.data or not response.data.result_list:
                logger.debug("[IndependentSites-SubGraph] 向量检索无结果")
                return {
                    "match_tags": [],
                    "tag_text": tag_text,
                }

            # Step 2: 提取检索结果 - 站点知识库返回 platform_type 和 platform_name
            content_list = []
            for item in response.data.result_list:
                if item.table_chunk_fields:
                    platform_type, platform_name = "", ""
                    for chunk in item.table_chunk_fields:
                        field_name = chunk.get("field_name", "")
                        field_value = chunk.get("field_value", "")
                        if field_name == "key":
                            platform_name = str(field_value)
                        elif field_name == "value":
                            platform_type = str(field_value)
                    if platform_type:
                        # 格式: "platformType,platformName" 与 n8n 一致
                        content_list.append(f"{platform_type},{platform_name}")

            if not content_list:
                logger.debug("[IndependentSites-SubGraph] 提取检索结果为空")
                return {
                    "match_tags": [],
                    "tag_text": tag_text,
                }

            # Step 3: LLM 清洗召回文本 - 对齐 n8n 的 '清洗召回文本' 节点
            llm: BaseChatModel = llm_factory.get_llm(
                LlmProvider.OPENROUTER.name, LlmModelName.OPENROUTER_GEMINI_2_5_FLASH.value
            )

            recall_text = "#".join(content_list)
            messages = coze_loop_client_provider.get_langchain_messages(
                prompt_key=CozePromptHubKey.ABROAD_GOODS_SITE_CLEAN_PROMPT.value,
                variables={"tag_text": tag_text, "recall_text": recall_text},
            )
            clean_result = llm.invoke(messages)

            # 解析清洗结果
            clean_content = (
                clean_result.content if hasattr(clean_result, "content") else str(clean_result)
            )
            # 尝试提取 content_list 或 clean_tag_list
            extracted_list = extract_json_from_llm_output(clean_content, "content_list")
            if not extracted_list:
                extracted_list = extract_json_from_llm_output(clean_content, "clean_tag_list")
            if extracted_list and isinstance(extracted_list, list):
                content_list = extracted_list

            if not content_list:
                return {
                    "match_tags": [],
                    "tag_text": tag_text,
                }

            # Step 4: content_list 已是 "platformType,platformName" 格式，直接作为 match_tags
            match_tags = content_list

            logger.debug(f"[IndependentSites-SubGraph] 匹配到 {len(match_tags)} 个站点")
            return {
                "match_tags": match_tags,
                "tag_text": tag_text,
            }

        except Exception as e:
            logger.warning(f"[IndependentSites-SubGraph] 站点向量检索/匹配失败: {e}")
            return {
                "match_tags": [],
                "tag_text": tag_text,
            }

    def _process_platforms_node(self, state: IndependentSitesSubGraphState) -> dict[str, Any]:
        """平台处理 - 对齐 n8n 的 '单选站点处理' / '多选站点处理' 节点

        对应 n8n 节点: "单选站点处理" 或 "多选站点处理"
        从 match_tags 中提取 platformType（格式: "platformType,platformName"）
        返回站点数量、是否单选等信息。
        """
        match_tags: list[Any] = state.get("match_tags", [])

        if not match_tags:
            logger.warning("[IndependentSites-SubGraph] 无站点数据")
            return {
                "platform_types": [],
                "is_single_site": False,
                "site_count": 0,
                "has_sites": False,
            }

        # 提取所有 platformType - 对应 n8n 逻辑: map(item => item.split(',')[0])
        platform_types: list[str] = []
        for tag in match_tags:
            if isinstance(tag, str) and "," in tag:
                platform_types.append(tag.split(",")[0])

        site_count: int = len(platform_types)

        logger.debug(
            f"[IndependentSites-SubGraph] 平台处理: {site_count} 个站点, "
            f"is_single={site_count == 1}"
        )

        return {
            "platform_types": platform_types,
            "is_single_site": site_count == 1,
            "site_count": site_count,
            "has_sites": site_count > 0,
        }

    def _build_api_params(
        self, state: IndependentSitesSubGraphState, is_fallback: bool = False
    ) -> dict[str, Any]:
        """构建 API 参数 - 对齐 n8n 的 '站点详情参数' 节点

        对应 n8n 节点: "站点详情参数"
        - is_fallback=False: 对应 n8n 的 params1（主请求参数）
        - is_fallback=True: 对应 n8n 的 params2（兜底参数）

        参数包含:
        - 核心参数: sortType, pageSize, pageNo, onSaleFlag, designElementIndustry, resultCountLimit
        - platformType: 从 platform_types[0] 获取
        - 筛选参数: categoryIdList, minSprice, maxSprice, startDate, endDate, label, color, bodyType, text
        - 销量范围: minSaleVolume30Day, maxSaleVolume30Day

        Fallback 时:
        - 清空 style（风格筛选）
        - brand 仍按品牌意图保留（brand_tags/param_result.brand）

        注意: designElementIndustry 固定为 "CLOTH"，与 Amazon/Temu 不同，独立站 API 接受此参数。
        """
        platform_types: list[Any] = state.get("platform_types", [])
        param_result: Any = state.get("param_result")
        brand_tags: list[str] = state.get("brand_tags") or []
        sort_type: int = state.get("sort_type", 1)
        params: dict[str, Any] = {
            "sortType": sort_type,
            "pageSize": 40,
            "pageNo": 1,
            "onSaleFlag": 1,
            "designElementIndustry": "CLOTH",
            "resultCountLimit": 6000,
        }

        # 添加 platformType - 对应 n8n: params[5].output.platformType[0]
        if platform_types:
            platform_str: str = str(platform_types[0])
            params["platformType"] = int(platform_str) if platform_str.isdigit() else 0

        if param_result:
            # 价格
            params["minSprice"] = (
                int(param_result.min_sprice * 100) if param_result.min_sprice is not None else None
            )
            params["maxSprice"] = (
                int(param_result.max_sprice * 100) if param_result.max_sprice is not None else None
            )

            # 类目 - 需要转换为二维数组格式
            category_id_list_2d = []
            for item in param_result.category_id_list or []:
                if isinstance(item, str) and "," in item:
                    category_id_list_2d.append(item.split(","))
                elif isinstance(item, list):
                    category_id_list_2d.append(item)
                else:
                    category_id_list_2d.append([str(item)])
            params["categoryIdList"] = category_id_list_2d

            # 上架日期
            params["putOnSaleStartDate"] = param_result.put_on_sale_start_date or ""
            params["putOnSaleEndDate"] = param_result.put_on_sale_end_date or ""
            # 统计日期
            params["startDate"] = param_result.start_date or ""
            params["endDate"] = param_result.end_date or ""

            # 标签（n8n 为二维数组）
            label_items: list[str] = []
            if param_result.label:
                if isinstance(param_result.label, str):
                    label_items = [v.strip() for v in param_result.label.split(",") if v.strip()]
                elif isinstance(param_result.label, list):
                    label_items = [str(v).strip() for v in param_result.label if str(v).strip()]
            params["label"] = [label_items] if label_items else []

            # 销量范围
            params["minSaleVolume30Day"] = param_result.min_sale_volume_total
            params["maxSaleVolume30Day"] = param_result.max_sale_volume_total

            # 颜色
            colors: list[Any] = []
            if param_result.color:
                colors = (
                    [param_result.color]
                    if isinstance(param_result.color, str)
                    else param_result.color
                )
            params["color"] = colors

            # 体型
            body_types: list[Any] = []
            if param_result.body_type:
                body_types = (
                    [param_result.body_type]
                    if isinstance(param_result.body_type, str)
                    else param_result.body_type
                )
            params["bodyType"] = body_types

            # 文本搜索
            params["text"] = param_result.text or ""

            # 限制数量
            if param_result.limit:
                params["resultCountLimit"] = param_result.limit

            # 风格与品牌 - 兜底时清空 style（对齐 n8n params2 逻辑）
            if not is_fallback:
                # 主请求参数: 包含 style 和 brand
                styles: list[Any] = []
                if param_result.style:
                    styles = (
                        [param_result.style]
                        if isinstance(param_result.style, str)
                        else param_result.style
                    )
                params["style"] = styles

                # 品牌过滤策略: 有品牌意图则保留 brand
                brands: list[Any] = []
                if brand_tags:
                    brands = brand_tags
                elif param_result.brand:
                    brands = (
                        [param_result.brand]
                        if isinstance(param_result.brand, str)
                        else param_result.brand
                    )
                if brands:
                    logger.info(
                        f"[IndependentSites-SubGraph] 使用brand过滤: {brands}"
                    )
                params["brand"] = brands
            else:
                # 兜底参数 - 移除 style（对齐 n8n params2）
                params["style"] = []
                brands: list[Any] = []
                if brand_tags:
                    brands = brand_tags
                elif param_result.brand:
                    brands = (
                        [param_result.brand]
                        if isinstance(param_result.brand, str)
                        else param_result.brand
                    )
                params["brand"] = brands

        return params

    def _call_primary_api_node(self, state: IndependentSitesSubGraphState) -> dict[str, Any]:
        """调用主 API - 对齐 n8n 的 '站点详情接口' 节点

        对应 n8n 节点: "站点详情接口"
        API 路径: /external/for-zxy/site-goods-list
        使用参数: params1（包含 style/brand 等完整筛选参数）

        如果结果数 < RESULT_THRESHOLD (50) 或调用失败，会触发兜底逻辑。
        """
        user_id: Any = state.get("user_id")
        team_id: Any = state.get("team_id")
        params: dict[str, Any] = self._build_api_params(state, is_fallback=False)
        logger.debug(f"[IndependentSites-SubGraph] 主API调用, params={params}")

        api = get_abroad_api()
        try:
            result = api.external_site_goods_list(str(user_id), str(team_id), params)
            count = getattr(result, "result_count_max", 0) or len(
                getattr(result, "result_list", [])
            )
            logger.info(f"[IndependentSites-SubGraph] 主API结果: {count} 个商品")
            return {
                "primary_params": params,
                "primary_api_resp": result,
                "primary_result_count": count,
                "primary_success": True,
                "primary_request_path": "external/for-zxy/site-goods-list",
            }
        except Exception as e:
            logger.error(f"[IndependentSites-SubGraph] 主API调用失败: {e}")
            return {
                "primary_params": params,
                "primary_api_resp": None,
                "primary_result_count": 0,
                "primary_success": False,
                "primary_request_path": "external/for-zxy/site-goods-list",
            }

    def _call_secondary_api_node(self, state: IndependentSitesSubGraphState) -> dict[str, Any]:
        """调用兜底 API - 对齐 n8n 的 '站点详情接口再次请求' 节点

        对应 n8n 节点: "站点详情接口再次请求"
        API 路径: /goods-list/site-goods-list
        使用参数: params2（清空 style 的简化参数，brand 仍保留）

        兜底逻辑: 当主 API 结果不足（< 50）或失败时触发，移除 style 等复杂筛选参数后重试。
        sortType 降级: 如果 sortType=8（热销排序），降级到 sortType=1（最新上架）。
        """
        user_id: Any = state.get("user_id")
        team_id: Any = state.get("team_id")
        params: dict[str, Any] = self._build_api_params(state, is_fallback=True)
        
        # 兜底时检查sortType：独立站商品可能没有销量数据，sortType=8（热销排序）会返回0结果
        # 兜底策略：降级到最新上架排序（sortType=1），确保有数据返回
        if params.get("sortType") == 8:
            logger.info(f"[IndependentSites-SubGraph] 兜底模式: sortType从8(热销)降级为1(最新上架)")
            params["sortType"] = 1
        
        logger.debug(f"[IndependentSites-SubGraph] 兜底API调用, params={params}")

        api = get_abroad_api()
        try:
            result = api.site_goods_list(str(user_id), str(team_id), params)
            count = getattr(result, "result_count_max", 0) or len(
                getattr(result, "result_list", [])
            )
            logger.info(f"[IndependentSites-SubGraph] 兜底API结果: {count} 个商品")
            return {
                "secondary_params": params,
                "secondary_api_resp": result,
                "secondary_result_count": count,
                "secondary_success": True,
                "secondary_request_path": "goods-list/site-goods-list",
                "fallback_attempted": True,
            }
        except Exception as e:
            logger.error(f"[IndependentSites-SubGraph] 兜底API调用失败: {e}")
            return {
                "secondary_params": params,
                "secondary_api_resp": None,
                "secondary_result_count": 0,
                "secondary_success": False,
                "secondary_request_path": "goods-list/site-goods-list",
                "fallback_attempted": True,
            }

    def _output_node(self, state: IndependentSitesSubGraphState) -> dict[str, Any]:
        """输出节点 - 对齐 n8n 的输出逻辑

        优先使用主 API 结果，如果主 API 失败或结果不足，则使用兜底 API 结果。
        返回最终的商品列表、参数、请求路径等信息。
        """
        primary_resp: Any = state.get("primary_api_resp")
        secondary_resp: Any = state.get("secondary_api_resp")
        primary_count = state.get("primary_result_count", 0)
        secondary_count = state.get("secondary_result_count", 0)
        primary_success = state.get("primary_success", True)
        secondary_success = state.get("secondary_success", False)

        fallback_attempted = state.get("fallback_attempted", False) or (
            "secondary_params" in state
        )

        def _has_results(resp: Any, count: int | None) -> bool:
            if count is not None and count > 0:
                return True
            return bool(resp and getattr(resp, "result_list", None))

        if fallback_attempted:
            has_results = secondary_success and _has_results(secondary_resp, secondary_count)
            resp = secondary_resp if has_results else None
            final_result_count = secondary_count if has_results else 0
            request_params = state.get("secondary_params")
            request_path = state.get("secondary_request_path", "")
            final_success = secondary_success and has_results
            used_fallback = True
        else:
            has_results = primary_success and _has_results(primary_resp, primary_count)
            resp = primary_resp if has_results else None
            final_result_count = primary_count if has_results else 0
            request_params = state.get("primary_params")
            request_path = state.get("primary_request_path", "")
            final_success = primary_success and has_results
            used_fallback = False

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
            "final_result_count": final_result_count,
            "final_api_resp": resp,
            "final_success": final_success,
            "used_fallback": used_fallback,
            "fallback_attempted": fallback_attempted,
        }


# 单例
_independent_sites_subgraph: IndependentSitesSubGraph | None = None


def get_independent_sites_subgraph() -> IndependentSitesSubGraph:
    global _independent_sites_subgraph
    if _independent_sites_subgraph is None:
        _independent_sites_subgraph = IndependentSitesSubGraph()
    return _independent_sites_subgraph


__all__ = ["IndependentSitesSubGraph", "get_independent_sites_subgraph"]
