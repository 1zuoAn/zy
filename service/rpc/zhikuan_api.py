# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2025/12/1 18:16
# @File     : zhikuan_api.py
"""
知款业务 API 封装
"""
from __future__ import annotations

import json
from typing import Optional

import requests
from loguru import logger
from pydantic import BaseModel, Field

from app.config import settings
from app.core.errors import AppException, ErrorCode
from app.schemas.entities.workflow.llm.zhikuan_ins_output import ZhikuanInsParseParam
from app.schemas.response.common import CommonResponse, PageResult


# ============================================================
# 知款 - 博客列表请求实体
# ============================================================

class ZhikuanInsBlogListRequest(BaseModel):
    """知款博客列表请求参数"""
    class Config:
        populate_by_name = True

    start: Optional[int] = 0
    page_size: Optional[int] = Field(default=60, alias="pageSize")
    blog_time_start: Optional[str] = Field(default=None, alias="blogTimeStart")
    blog_time_end: Optional[str] = Field(default=None, alias="blogTimeEnd")
    gender: Optional[str] = None
    category_list: Optional[list[str]] = Field(default=None, alias="categoryList")
    image_style_list: Optional[list[str]] = Field(default_factory=list, alias="imageStyleList")
    label_matrix: Optional[list] = Field(default_factory=list, alias="labelMatrix")
    blogger_skin_colors: Optional[list[str]] = Field(default_factory=list, alias="bloggerSkinColors")
    blogger_body_shapes: Optional[list[str]] = Field(default_factory=list, alias="bloggerBodyShapes")
    blogger_tags: Optional[list[str]] = Field(default_factory=list, alias="bloggerTags")
    min_fans_num: Optional[int] = Field(default=None, alias="minFansNum")
    max_fans_num: Optional[int] = Field(default=None, alias="maxFansNum")
    rank_status: Optional[str] = Field(default=None, alias="rankStatus", description="排序项")
    limit: Optional[int] = 80
    blogger_group_id_list: Optional[list[str]] = Field(default_factory=list, alias="bloggerGroupIdList")
    source_type: Optional[str] = Field(default=None, alias="sourceType")
    platform_id: Optional[str] = Field(default=None, alias="platformId")

    @classmethod
    def from_parse_param(cls, param: ZhikuanInsParseParam) -> "ZhikuanInsBlogListRequest":
        """
        从 LLM 解析结果转换为请求参数

        Args:
            param: LLM 解析出的参数 (ZhikuanInsParseParam)

        Returns:
            API 请求参数
        """
        # 排除不需要的字段（title, user_data, sort_type, is_monitor_blogger 等）
        data = param.model_dump(
            by_alias=True,
            exclude={"title", "user_data", "sort_type", "is_monitor_blogger"}
        )
        # blogger_group_id_list 需要转换为字符串列表
        if data.get("bloggerGroupIdList"):
            data["bloggerGroupIdList"] = [str(x) for x in data["bloggerGroupIdList"]]
        return cls.model_validate(data)



# ============================================================
# 知款 - 响应实体类（核心字段）
# ============================================================

class BoxLabelItem(BaseModel):
    """标签框项"""
    box_id: Optional[str] = Field(default=None, alias="boxId")
    label_array: Optional[list[str]] = Field(default=None, alias="labelArray")
    xmax: Optional[str] = None
    xmin: Optional[str] = None
    ymax: Optional[str] = None
    ymin: Optional[str] = None

    class Config:
        populate_by_name = True


class FinalLabelItem(BaseModel):
    """最终标签项（带颜色信息）"""
    root_tag: Optional[str] = Field(default=None, alias="rootTag")
    tag: Optional[str] = None
    type: Optional[int] = None
    color_number: Optional[str] = Field(default=None, alias="colorNumber")
    color: Optional[str] = None
    en_name: Optional[str] = Field(default=None, alias="enName")
    color_level: Optional[int] = Field(default=None, alias="colorLevel")

    class Config:
        populate_by_name = True


class ZkBoxLabelItem(BaseModel):
    """知款标签框项（带详细标签）"""
    box_id: Optional[str] = Field(default=None, alias="boxId")
    category: Optional[str] = None
    gender: Optional[str] = None
    root_category: Optional[str] = Field(default=None, alias="rootCategory")
    final_label_array: Optional[list[FinalLabelItem]] = Field(default=None, alias="finalLabelArray")
    xmax: Optional[float] = None
    xmin: Optional[float] = None
    ymax: Optional[float] = None
    ymin: Optional[float] = None

    class Config:
        populate_by_name = True


class ZhikuanBloggerInfo(BaseModel):
    """知款博主信息"""
    blogger_id: Optional[str] = Field(default=None, alias="bloggerId")
    nick_name: Optional[str] = Field(default=None, alias="nickName")
    full_name: Optional[str] = Field(default=None, alias="fullName")
    head_img: Optional[str] = Field(default=None, alias="headImg")
    region: Optional[str] = None
    fans_num: Optional[int] = Field(default=None, alias="fansNum")
    follow_num: Optional[int] = Field(default=None, alias="followNum")
    blog_num: Optional[int] = Field(default=None, alias="blogNum")
    introduction: Optional[str] = None
    is_verified: Optional[int] = Field(default=None, alias="isVerified")
    blogger_type: Optional[str] = Field(default=None, alias="bloggerType")
    clothing_types: Optional[str] = Field(default=None, alias="clothingTypes")
    sum_tags: Optional[str] = Field(default=None, alias="sumTags")

    class Config:
        populate_by_name = True


class ZhikuanInsDataDTO(BaseModel):
    """知款 Instagram 数据"""
    blog_type: Optional[int] = Field(default=None, alias="blogType")
    blog_url: Optional[str] = Field(default=None, alias="blogUrl")
    blogger_id: Optional[str] = Field(default=None, alias="bloggerId")
    blogger_name: Optional[str] = Field(default=None, alias="bloggerName")
    blogger_obj: Optional[ZhikuanBloggerInfo] = Field(default=None, alias="bloggerObj")
    blogger_tags: Optional[list[str]] = Field(default=None, alias="bloggerTags")
    comment_num: Optional[int] = Field(default=None, alias="commentNum")
    like_num: Optional[int] = Field(default=None, alias="likeNum")
    like_fans_ratio: Optional[float] = Field(default=None, alias="likeFansRatio")
    image_num: Optional[int] = Field(default=None, alias="imageNum")
    detail_urls: Optional[str] = Field(default=None, alias="detailUrls")
    text_content: Optional[str] = Field(default=None, alias="textContent")
    season: Optional[str] = None

    class Config:
        populate_by_name = True


class ZhikuanInsBlogEntity(BaseModel):
    """知款博客实体"""
    union_id: Optional[str] = Field(default=None, alias="unionId")
    image_group_entity_id: Optional[str] = Field(default=None, alias="imageGroupEntityId")
    main_url: Optional[str] = Field(default=None, alias="mainUrl")
    height: Optional[int] = None
    width: Optional[int] = None
    platform_id: Optional[int] = Field(default=None, alias="platformId")
    source_time: Optional[str] = Field(default=None, alias="sourceTime")
    sort_values: Optional[list] = Field(default=None, alias="sortValues")

    # 标签数据
    box_label_ret_list: Optional[list[BoxLabelItem]] = Field(default=None, alias="boxLabelRetList")
    zk_box_label_ret_list_origin: Optional[list[ZkBoxLabelItem]] = Field(
        default=None, alias="zkBoxLabelRetListOrigin"
    )

    # Instagram 数据
    ins_data_dto: Optional[ZhikuanInsDataDTO] = Field(default=None, alias="insDataDTO")

    # 图片过滤标签
    image_filters: Optional[list[str]] = Field(default=None, alias="imageFilters")

    class Config:
        populate_by_name = True


# ============================================================
# 知款 - 博主分组响应实体
# ============================================================

class UserPermItem(BaseModel):
    """用户权限项"""
    class Config:
        populate_by_name = True

    user_id: Optional[int] = Field(default=None, alias="userId")
    nick_name: Optional[str] = Field(default=None, alias="nickName")
    avatar: Optional[str] = None
    member_type: Optional[int] = Field(default=None, alias="memberType")
    owner_flag: Optional[bool] = Field(default=None, alias="ownerFlag")
    user_perm_list: Optional[list[int]] = Field(default=None, alias="userPermList")


class BloggerGroupItem(BaseModel):
    """博主分组项"""
    class Config:
        populate_by_name = True

    id: Optional[int] = None
    group_name: Optional[str] = Field(default=None, alias="groupName")
    blogger_num: Optional[int] = Field(default=None, alias="bloggerNum")
    user_num: Optional[int] = Field(default=None, alias="userNum")
    is_default: Optional[int] = Field(default=None, alias="isDefault")
    source_type: Optional[int] = Field(default=None, alias="sourceType")
    team_id: Optional[int] = Field(default=None, alias="teamId")
    user_id: Optional[int] = Field(default=None, alias="userId")
    user_name: Optional[str] = Field(default=None, alias="userName")
    status: Optional[int] = None
    cooperation: Optional[int] = None
    in_group: Optional[int] = Field(default=None, alias="inGroup")
    cover_img_url: Optional[str] = Field(default=None, alias="coverImgUrl")
    remark: Optional[str] = None
    created_at: Optional[str] = Field(default=None, alias="createdAt")
    updated_at: Optional[str] = Field(default=None, alias="updatedAt")
    user_perm_list: Optional[list[UserPermItem]] = Field(default=None, alias="userPermList")


class BloggerGroupListResult(BaseModel):
    """博主分组列表结果"""
    class Config:
        populate_by_name = True

    recent_view_group_entity_num: Optional[int] = Field(default=None, alias="recentViewGroupEntityNum")
    recent_view_group_flag: Optional[int] = Field(default=None, alias="recentViewGroupFlag")
    self_list: Optional[list[BloggerGroupItem]] = Field(default=None, alias="selfList")
    team_list: Optional[list[BloggerGroupItem]] = Field(default=None, alias="teamList")


# ============================================================
# 知款 API 封装
# ============================================================

class ZhikuanAPI:
    """
    知款 API
    封装对知款后端服务的调用
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
        """
        url = f"{self._base_url}{path}"
        headers = kwargs.pop("headers", {})
        headers.update({
            "USER-ID": str(user_id),
            "TEAM-ID": str(team_id),
            "Content-Type": "application/json",
        })

        logger.debug(f"[ZhikuanAPI] {method} {url}")
        if "json" in kwargs:
            logger.debug(f"[ZhikuanAPI] Request Body: {json.dumps(kwargs['json'], ensure_ascii=False)}")

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
            logger.error(f"[ZhikuanAPI] 请求超时: {url}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, "知款服务请求超时")
        except Exception as e:
            logger.error(f"[ZhikuanAPI] 请求异常: {e}")
            raise AppException(ErrorCode.EXTERNAL_API_ERROR, str(e))

    def list_blogs(
        self,
        user_id: str,
        team_id: str,
        params: ZhikuanInsBlogListRequest,
    ) -> PageResult[ZhikuanInsBlogEntity]:
        """
        获取知款博客列表
        """
        data = self._request(
            method="POST",
            path="/image-bus/label-selector/list-blog",
            user_id=user_id,
            team_id=team_id,
            json=params.model_dump(by_alias=True, exclude_none=True),
        )

        # 解析响应
        response = CommonResponse[PageResult[ZhikuanInsBlogEntity]].model_validate(data)

        if not response.success:
            logger.warning(f"[ZhikuanAPI] 业务错误: {response.error_code} - {response.error_desc}")
            raise AppException(
                ErrorCode.EXTERNAL_API_ERROR,
                response.error_desc or "知款服务返回错误",
            )

        return response.result or PageResult(result_list=[], result_count=0)

    def list_all_blogger_groups(
        self,
        user_id: str,
        team_id: str,
        source_type: Optional[int] = None,
        is_show_default: Optional[int] = None,
    ) -> BloggerGroupListResult:
        """
        获取所有博主分组列表

        Args:
            user_id: 用户 ID
            team_id: 团队 ID
            source_type: 数据来源类型
            is_show_default: 是否显示默认分组

        Returns:
            博主分组列表结果
        """
        params = {}
        if source_type is not None:
            params["sourceType"] = source_type
        if is_show_default is not None:
            params["isShowDefault"] = is_show_default

        data = self._request(
            method="GET",
            path="/ins/blogger/list-all-blogger-group",
            user_id=user_id,
            team_id=team_id,
            params=params,
        )

        # 解析响应
        response = CommonResponse[BloggerGroupListResult].model_validate(data)

        if not response.success:
            logger.warning(f"[ZhikuanAPI] 业务错误: {response.error_code} - {response.error_desc}")
            raise AppException(
                ErrorCode.EXTERNAL_API_ERROR,
                response.error_desc or "知款服务返回错误",
            )

        return response.result or BloggerGroupListResult()


# ============================================================
# 延迟初始化单例
# ============================================================

_zhikuan_api_client: Optional[ZhikuanAPI] = None


def get_zhikuan_api() -> ZhikuanAPI:
    """
    获取知款 API 实例（延迟初始化）

    首次调用时才读取配置并创建实例，避免模块加载时配置未就绪的问题。
    """
    global _zhikuan_api_client
    if _zhikuan_api_client is None:
        base_url = settings.zhikuan_api_url
        if not base_url:
            raise RuntimeError("zhikuan_api_url 未配置，请检查 Apollo 或环境变量")
        _zhikuan_api_client = ZhikuanAPI(
            base_url=base_url,
            timeout=settings.zhikuan_api_timeout,
        )
    return _zhikuan_api_client


class _LazyZhikuanAPIProxy:
    """延迟代理，保持 zhikuan_api_client 的使用方式不变"""

    def __getattr__(self, name):
        return getattr(get_zhikuan_api(), name)


zhikuan_api_client = _LazyZhikuanAPIProxy()
