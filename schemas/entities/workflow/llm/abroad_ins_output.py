# -*- coding: utf-8 -*-
from typing import List, Optional

from pydantic import BaseModel, Field


class AbroadInsCategoryParseResult(BaseModel):
    """Abroad INS category parse result."""

    class Config:
        populate_by_name = True

    category_list: List[List[str]] = Field(
        default_factory=list,
        alias="categoryList",
        description="Category path matrix, e.g. [['Womenswear', 'Tops']].",
    )
    region_list: List[List[str]] = Field(
        default_factory=list,
        alias="regionList",
        description="Region path matrix, e.g. [['North America', 'US']].",
    )


class AbroadInsTimeParseResult(BaseModel):
    """Abroad INS time parse result."""

    class Config:
        populate_by_name = True

    start_date: Optional[str] = Field(
        default=None,
        alias="startDate",
        description="Start date in yyyy-MM-dd format.",
    )
    end_date: Optional[str] = Field(
        default=None,
        alias="endDate",
        description="End date in yyyy-MM-dd format.",
    )


class AbroadInsBloggerParseResult(BaseModel):
    """Abroad INS blogger and monitor mode parse result."""

    class Config:
        populate_by_name = True

    blogger_skin_color_list: List[str] = Field(
        default_factory=list,
        alias="bloggerSkinColorList",
        description="Blogger skin color list.",
    )
    blogger_shapes: List[str] = Field(
        default_factory=list,
        alias="bloggerShapes",
        description="Blogger body shape list.",
    )
    min_fans_num: Optional[int] = Field(
        default=None,
        alias="minFansNum",
        description="Minimum fans count.",
    )
    max_fans_num: Optional[int] = Field(
        default=None,
        alias="maxFansNum",
        description="Maximum fans count.",
    )
    is_monitor_streamer: Optional[int] = Field(
        default=None,
        ge=0,
        le=1,
        alias="isMonitorStreamer",
        description="Monitor streamer mode flag (1 or 0).",
    )


class AbroadInsStyleParseResult(BaseModel):
    """Abroad INS style and detail parse result."""

    class Config:
        populate_by_name = True

    label: List[List[str]] = Field(
        default_factory=list,
        alias="label",
        description="Detail label matrix, e.g. [['Fabric', 'Jacquard']].",
    )
    style_list: List[str] = Field(
        default_factory=list,
        alias="styleList",
        description="Style tag list.",
    )


class AbroadInsMiscParseResult(BaseModel):
    """Abroad INS misc parse result."""

    class Config:
        populate_by_name = True

    limit: Optional[int] = Field(default=None, alias="limit", description="Result count limit.")
    sort_type: Optional[str] = Field(default=None, alias="sortType", description="Sort type.")
    search_content: Optional[str] = Field(
        default=None, alias="searchContent", description="Search content."
    )
    user_data: Optional[int] = Field(
        default=None,
        ge=0,
        le=1,
        alias="user_data",
        description="Use user preference flag (1 or 0).",
    )
    title: Optional[str] = Field(default=None, alias="title", description="Task title.")


class AbroadInsParseParam(BaseModel):
    """Abroad INS main parse parameter."""

    class Config:
        populate_by_name = True

    start_date: Optional[str] = Field(
        default=None, alias="startDate", description="Start date in yyyy-MM-dd format."
    )
    end_date: Optional[str] = Field(
        default=None, alias="endDate", description="End date in yyyy-MM-dd format."
    )
    category_list: List[List[str]] = Field(
        default_factory=list,
        alias="categoryList",
        description="Category path matrix.",
    )
    label: List[List[str]] = Field(
        default_factory=list,
        alias="label",
        description="Detail label matrix.",
    )
    style_list: List[str] = Field(
        default_factory=list,
        alias="styleList",
        description="Style tag list.",
    )
    region_list: List[List[str]] = Field(
        default_factory=list,
        alias="regionList",
        description="Region path matrix.",
    )
    blogger_skin_color_list: List[str] = Field(
        default_factory=list,
        alias="bloggerSkinColorList",
        description="Blogger skin color list.",
    )
    blogger_shapes: List[str] = Field(
        default_factory=list,
        alias="bloggerShapes",
        description="Blogger body shape list.",
    )
    min_fans_num: Optional[int] = Field(
        default=None, alias="minFansNum", description="Minimum fans count."
    )
    max_fans_num: Optional[int] = Field(
        default=None, alias="maxFansNum", description="Maximum fans count."
    )
    limit: int = Field(
        default=6000, le=6000, alias="limit", description="Result count limit (max 6000)."
    )
    sort_type: str = Field(default="默认", alias="sortType", description="Sort type.")
    search_content: Optional[str] = Field(default=None, alias="searchContent", description="Search content.")
    user_data: int = Field(
        default=0,
        ge=0,
        le=1,
        alias="user_data",
        description="Use user preference flag (1 or 0).",
    )
    is_monitor_streamer: int = Field(
        default=0,
        ge=0,
        le=1,
        alias="isMonitorStreamer",
        description="Monitor streamer mode flag (1 or 0).",
    )
    title: str = Field(
        ...,
        min_length=0,
        max_length=100,
        alias="title",
        description="Task title.",
    )


class AbroadInsSortTypeParseResult(BaseModel):
    """Abroad INS sort type parse result."""

    class Config:
        populate_by_name = True

    sort_type_final: str = Field(
        ...,
        alias="sortTypeFinal",
        description="Final sort type value.",
    )
    sort_type_final_name: str = Field(
        ...,
        alias="sortTypeFinalName",
        description="Final sort type display name.",
    )
