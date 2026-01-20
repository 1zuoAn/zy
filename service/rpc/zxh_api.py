# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2025/12/1 15:56
# @File     : zxh_api.py
"""
知小红 小红书业务 API 封装
"""
from __future__ import annotations

import json
from typing import Optional

import requests
from loguru import logger
from pydantic import BaseModel, Field

from app.config import settings
from app.core.errors import AppException, ErrorCode
from app.schemas.entities.workflow.llm_output import BoxLabelItem, ZxhXhsParseParam
from app.schemas.response.common import CommonResponse, PageResult


# ============================================================
# 知小红 - 小红书博文请求实体模型
# ============================================================

class XhsNoteSearchRequest(BaseModel):
    """小红书博文列表请求参数"""
    class Config:
        populate_by_name = True

    page_no: Optional[str] = Field(default="1", alias="pageNo")
    page_size: Optional[str] = Field(default="40", alias="pageSize")
    min_publish_time: Optional[str] = Field(default=None, alias="minPublishTime")
    max_publish_time: Optional[str] = Field(default=None, alias="maxPublishTime")
    root_seo_list: Optional[list[str]] = Field(default=None, alias="rootSeoList")
    second_seo_list: Optional[list[str]] = Field(default=None, alias="secondSeoList")
    industry: Optional[str] = Field(default=None, alias="industry")
    note_type: Optional[str] = Field(default=None, alias="noteType")
    blogger_skin_color_list: Optional[str] = Field(default=None, alias="bloggerSkinColorList")
    blogger_shapes: Optional[str] = Field(default=None, alias="bloggerShapes")
    sort_field: Optional[str] = Field(default=None, alias="sortField")
    sort_order: Optional[str] = Field(default=None, alias="sortOrder")
    limit: Optional[int] = Field(default=None, alias="limit")
    min_fan_num: Optional[int] = Field(default=None, alias="minFanNum")
    max_fan_num: Optional[int] = Field(default=None, alias="maxFanNum")
    box_label_color: Optional[str] = Field(default=None, alias="boxLabelColor")
    box_label_ret_list: Optional[list[BoxLabelItem]] = Field(default=None, alias="boxLabelRetList")
    exclude_user_black_flag: Optional[str] = Field(default=None, alias="excludeUserBlackFlag")
    exclude_user_black_word_flag: Optional[str] = Field(default=None, alias="excludeUserBlackWordFlag")
    min_agg_like_num: Optional[int] = Field(default=None, alias="minAggLikeNum")
    max_agg_like_num: Optional[int] = Field(default=None, alias="maxAggLikeNum")
    min_agg_comment_num: Optional[int] = Field(default=None, alias="minAggCommentNum")
    max_agg_comment_num: Optional[int] = Field(default=None, alias="maxAggCommentNum")
    min_agg_collect_num: Optional[int] = Field(default=None, alias="minAggCollectNum")
    max_agg_collect_num: Optional[int] = Field(default=None, alias="maxAggCollectNum")
    topic_list: Optional[list[str]] = Field(default=None, alias="topicList")
    style_list: Optional[list[str]] = Field(default=None, alias="styleList")
    metric_date: Optional[str] = Field(default=None, alias="metricDate")
    search_like: Optional[str] = Field(default=None, alias="searchLike")
    search_field_list: Optional[list[str]] = Field(default=None, alias="searchFieldList")

    @classmethod
    def from_parse_param(cls, param: ZxhXhsParseParam) -> "XhsNoteSearchRequest":
        data = param.model_dump(by_alias=True, exclude={"title", "user_data", "topicQueryText"})
        return cls.model_validate(data)

class XhsNoteMonitorRequest(XhsNoteSearchRequest):
    """小红书监控达人博文列表请求参数"""
    class Config:
        populate_by_name = True

    # 监控额外参数
    group_id_list: Optional[list[str]] = Field(default_factory=list, alias="groupIdList")
    monitor_type: Optional[str] = Field(default=None, alias="monitorType", description="监控类型：blogger-达人，note-笔记，brand-品牌，shop-店铺")


# ============================================================
# 知小红 - 小红书博文响应实体模型
# ============================================================

class XhsBlogEntity(BaseModel):
    """知小红 - 小红书博文实体"""
    class Config:
        populate_by_name = True

    # 笔记基础信息
    note_id: Optional[str] = Field(default=None, alias="noteId", description="笔记ID")
    pic_url: Optional[str] = Field(default=None, alias="picUrl", description="封面图URL")
    pic_url_list: Optional[list[str]] = Field(default=None, alias="picUrlList", description="图片列表")
    xsec_token: Optional[str] = Field(default=None, alias="xsecToken", description="安全token")
    title: Optional[str] = Field(default=None, alias="title", description="笔记标题")
    intro: Optional[str] = Field(default=None, alias="intro", description="笔记简介/标签")
    publish_time: Optional[str] = Field(default=None, alias="publishTime", description="发布时间")
    edit_time: Optional[str] = Field(default=None, alias="editTime", description="编辑时间")
    note_type: Optional[int] = Field(default=None, alias="noteType", description="笔记类型，0图文1视频")
    detail_status: Optional[int] = Field(default=None, alias="detailStatus", description="详情状态")
    platform_id: Optional[int] = Field(default=None, alias="platformId", description="平台ID")
    pic_change_status: Optional[int] = Field(default=None, alias="picChangeStatus", description="图片变更状态")
    # 类目信息
    root_seo_list: Optional[list[str]] = Field(default=None, alias="rootSeoList", description="一级类目列表")
    second_seo_list: Optional[list[str]] = Field(default=None, alias="secondSeoList", description="二级类目列表")
    # 博主信息
    xhs_user_id: Optional[str] = Field(default=None, alias="xhsUserId", description="小红书用户ID")
    nickname: Optional[str] = Field(default=None, alias="nickname", description="博主昵称")
    avatar: Optional[str] = Field(default=None, alias="avatar", description="博主头像")
    fan_num: Optional[int] = Field(default=None, alias="fanNum", description="粉丝数")
    phone_number: Optional[str] = Field(default=None, alias="phoneNumber", description="手机号")
    email_number: Optional[str] = Field(default=None, alias="emailNumber", description="邮箱")
    user_intro: Optional[str] = Field(default=None, alias="userIntro", description="博主简介")
    fans_distribution: Optional[str] = Field(default=None, alias="fansDistribution", description="粉丝分布")
    # # 品牌信息
    # brand_name_list: Optional[list[str]] = Field(default=None, alias="brandNameList", description="品牌名称列表")
    # business_brand_name_list: Optional[list[str]] = Field(default=None, alias="businessBrandNameList", description="商业品牌列表")
    # is_business: Optional[int] = Field(default=None, alias="isBusiness", description="是否商业笔记")
    # sentiment: Optional[str] = Field(default=None, alias="sentiment", description="情感倾向")
    # # 互动数据（聚合）
    # agg_like_num: Optional[int] = Field(default=None, alias="aggLikeNum", description="聚合点赞数")
    # agg_collect_num: Optional[int] = Field(default=None, alias="aggCollectNum", description="聚合收藏数")
    # agg_comment_num: Optional[int] = Field(default=None, alias="aggCommentNum", description="聚合评论数")
    # agg_share_num: Optional[int] = Field(default=None, alias="aggShareNum", description="聚合分享数")
    # agg_interaction_num: Optional[int] = Field(default=None, alias="aggInteractionNum", description="聚合互动数")
    # agg_interaction_ratio: Optional[float] = Field(default=None, alias="aggInteractionRatio", description="聚合互动率")
    # agg_avg_like_num: Optional[int] = Field(default=None, alias="aggAvgLikeNum", description="聚合平均点赞数")
    # agg_avg_collect_num: Optional[int] = Field(default=None, alias="aggAvgCollectNum", description="聚合平均收藏数")
    # agg_avg_comment_num: Optional[int] = Field(default=None, alias="aggAvgCommentNum", description="聚合平均评论数")
    # agg_avg_share_num: Optional[int] = Field(default=None, alias="aggAvgShareNum", description="聚合平均分享数")
    # agg_avg_interaction_num: Optional[int] = Field(default=None, alias="aggAvgInteractionNum", description="聚合平均互动数")
    # # 互动数据（实时）
    # like_num: Optional[int] = Field(default=None, alias="likeNum", description="点赞数")
    # collect_num: Optional[int] = Field(default=None, alias="collectNum", description="收藏数")
    # comment_num: Optional[int] = Field(default=None, alias="commentNum", description="评论数")
    # share_num: Optional[int] = Field(default=None, alias="shareNum", description="分享数")
    # interaction_num: Optional[int] = Field(default=None, alias="interactionNum", description="互动数")
    # imp_num: Optional[int] = Field(default=None, alias="impNum", description="曝光数")
    # read_num: Optional[int] = Field(default=None, alias="readNum", description="阅读数")
    # # 比率数据
    # like_fan_rate: Optional[float] = Field(default=None, alias="likeFanRate", description="点赞粉丝比")
    # like_collect_rate: Optional[float] = Field(default=None, alias="likeCollectRate", description="点赞收藏比")
    # # 监控状态
    # is_monitor_blogger: Optional[bool] = Field(default=None, alias="isMonitorBlogger", description="是否监控博主")
    # is_team_monitor_blogger: Optional[bool] = Field(default=None, alias="isTeamMonitorBlogger", description="是否团队监控博主")
    # is_monitor_note: Optional[bool] = Field(default=None, alias="isMonitorNote", description="是否监控笔记")
    # is_team_monitor_note: Optional[bool] = Field(default=None, alias="isTeamMonitorNote", description="是否团队监控笔记")
    # monitor_group_ids: Optional[list[str]] = Field(default=None, alias="monitorGroupIds", description="监控分组ID列表")
    # # 备注信息
    # user_remark_content: Optional[str] = Field(default=None, alias="userRemarkContent", description="用户备注")
    # note_remark_content: Optional[str] = Field(default=None, alias="noteRemarkContent", description="笔记备注")
    # # 黑名单状态
    # is_black: Optional[bool] = Field(default=None, alias="isBlack", description="是否黑名单")
    # is_team_black: Optional[bool] = Field(default=None, alias="isTeamBlack", description="是否团队黑名单")
    # is_compared: Optional[bool] = Field(default=None, alias="isCompared", description="是否已对比")
    # # 关联ID
    # black_id: Optional[str] = Field(default=None, alias="blackId", description="黑名单ID")
    # coo_note_id: Optional[str] = Field(default=None, alias="cooNoteId", description="协作笔记ID")
    # monitor_id: Optional[str] = Field(default=None, alias="monitorId", description="监控ID")
    # compare_id: Optional[str] = Field(default=None, alias="compareId", description="对比ID")
    # remark_id: Optional[str] = Field(default=None, alias="remarkId", description="备注ID")
    # remark_note_id: Optional[str] = Field(default=None, alias="remarkNoteId", description="备注笔记ID")
    # id: Optional[str] = Field(default=None, alias="id", description="记录ID")
    # # 商品信息
    # item_info_list: Optional[list] = Field(default=None, alias="itemInfoList", description="商品信息列表")
    # long_item_list: Optional[list] = Field(default=None, alias="longItemList", description="长商品列表")
    # oltp_connect_item_list: Optional[list] = Field(default=None, alias="oltpConnectItemList", description="OLTP关联商品列表")


# ============================================================
# 知小红 API 封装
# ============================================================

class ZxhAPI:
    """
    知小红 API
    封装对知小红后端服务的调用
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 20.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        user_id: str,
        team_id: str,
        **kwargs,
    ) -> dict:
        """
        发送请求

        Args:
            method: HTTP 方法
            path: 请求路径
            user_id: 用户 ID（header）
            team_id: 团队 ID（header）
            **kwargs: 其他 requests 参数

        Returns:
            响应 JSON
        """
        url = f"{self._base_url}{path}"
        headers = kwargs.pop("headers", {})
        headers.update({
            "USER-ID": str(user_id),
            "TEAM-ID": str(team_id),
            "Content-Type": "application/json",
        })

        logger.debug(f"[ZxhAPI] {method} {url}")
        if "json" in kwargs:
            logger.debug(f"[ZxhAPI] Request Body: {json.dumps(kwargs['json'], ensure_ascii=False)}")

        try:
            resp = requests.request(
                method=method,
                url=url,
                headers=headers,
                timeout=self._timeout,
                **kwargs,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            logger.error(f"[ZxhAPI] 请求超时: {url}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, "知小红服务请求超时")
        except Exception as e:
            logger.error(f"[ZxhAPI] 请求异常: {e}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, str(e))

    def common_search(
        self,
        user_id: str,
        team_id: str,
        params: XhsNoteSearchRequest,
    ) -> PageResult[XhsBlogEntity]:
        """
        搜索小红书博文列表
        """
        data = self._request(
            method="POST", path="/notes/common-search", user_id=user_id, team_id=team_id, json=params.model_dump(by_alias=True, exclude_none=True)
        )

        # 解析响应
        response = CommonResponse[PageResult[XhsBlogEntity]].model_validate(data)

        if not response.success:
            logger.warning(f"[ZxhAPI] 业务错误: {response.error_code} - {response.error_desc}")
            raise AppException(
                ErrorCode.EXTERNAL_API_ERROR,
                response.error_desc or "知小红服务返回错误",
            )

        return response.result or PageResult[XhsBlogEntity]()

    def note_monitor_search(
        self,
        user_id: str,
        team_id: str,
        params: XhsNoteMonitorRequest,
    ) -> PageResult[XhsBlogEntity]:
        """搜索小红书监控博主博文列表"""
        data = self._request(
            method="POST", path="/user/monitor/xhs-user/note-list", user_id=user_id, team_id=team_id, json=params.model_dump(by_alias=True, exclude_none=True)
        )

        # 解析响应
        response = CommonResponse[PageResult[XhsBlogEntity]].model_validate(data)

        if not response.success:
            logger.warning(f"[ZxhAPI] 业务错误: {response.error_code} - {response.error_desc}")
            raise AppException(
                ErrorCode.EXTERNAL_API_ERROR,
                response.error_desc or "知小红服务返回错误",
            )

        return response.result or PageResult[XhsBlogEntity]()

# ============================================================
# 延迟初始化单例
# ============================================================

_zxh_api_client: Optional[ZxhAPI] = None


def get_zxh_api() -> ZxhAPI:
    """
    获取知小红 API 实例（延迟初始化）

    首次调用时才读取配置并创建实例，避免模块加载时配置未就绪的问题。
    """
    global _zxh_api_client
    if _zxh_api_client is None:
        base_url = settings.zxh_api_url
        if not base_url:
            raise RuntimeError("zxh_api_url 未配置，请检查 Apollo 或环境变量")
        _zxh_api_client = ZxhAPI(
            base_url=base_url,
            timeout=settings.zxh_api_timeout,
        )
    return _zxh_api_client


class _LazyZxhAPIProxy:
    """延迟代理，保持 zxh_api_client 的使用方式不变"""

    def __getattr__(self, name):
        return getattr(get_zxh_api(), name)


zxh_api_client = _LazyZxhAPIProxy()
