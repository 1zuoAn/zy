# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2025/01/19 10:00
# @File     : excel_export.py
"""
Excel 导出数据实体类

对应 Java 后端的 ExcelExportRequest 类结构，用于通过 Redis list 推送 Excel 导出数据。
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


# 列类型常量
EXPORT_COLUMN_TYPE_STRING: str = "string"
EXPORT_COLUMN_TYPE_IMAGE: str = "image"


class CellData(BaseModel):
    """单元格数据"""

    model_config = ConfigDict(populate_by_name=True)

    col_id: int = Field(alias="colId", description="列ID")
    cell_value: str = Field(alias="cellValue", description="单元格值")


class RowData(BaseModel):
    """行数据"""

    model_config = ConfigDict(populate_by_name=True)

    row_id: int = Field(alias="rowId", description="行ID")
    row_data: List[CellData] = Field(alias="rowData", default_factory=list, description="单元格数据列表")


class ColumnInfo(BaseModel):
    """列信息"""

    model_config = ConfigDict(populate_by_name=True)

    col_name: str = Field(alias="colName", description="列名")
    col_id: int = Field(alias="colId", description="列ID")
    col_type: str = Field(
        alias="colType",
        default="string",
        description="列类型: string/image",
    )


class SheetData(BaseModel):
    """Sheet 数据"""

    model_config = ConfigDict(populate_by_name=True)

    sheet_name: str = Field(alias="sheetName", description="Sheet名称")
    sheet_active: bool = Field(alias="sheetActive", default=False, description="Sheet是否激活")
    sheet_info: List[ColumnInfo] = Field(
        alias="sheetInfo",
        default_factory=list,
        description="列信息列表",
    )
    sheet_data: List[RowData] = Field(
        alias="sheetData",
        default_factory=list,
        description="行数据列表",
    )


class ExcelData(BaseModel):
    """Excel 文件数据"""

    model_config = ConfigDict(populate_by_name=True)

    xlsx_name: str = Field(alias="xlsxName", description="Excel文件名")
    xlsx_data: List[SheetData] = Field(
        alias="xlsxData",
        default_factory=list,
        description="Sheet数据列表",
    )


class ExcelExportRequest(BaseModel):
    """Excel 导出请求

    用于通过 Redis list 向后端推送 Excel 导出数据。
    对应 Java 后端的 ExcelExportRequest 类。
    """

    model_config = ConfigDict(populate_by_name=True)

    job_id: str = Field(alias="jobId", description="请求ID")
    interface_caller: str = Field(alias="interfaceCaller", description="FC调用者")
    data: Optional[ExcelData] = Field(default=None, description="Excel文件数据")

    def to_json_dict(self) -> dict:
        """转换为 JSON 字典（使用驼峰命名，用于序列化到 Redis）"""
        return self.model_dump(by_alias=True, exclude_none=True)


__all__ = [
    "CellData",
    "ColumnInfo",
    "ExcelData",
    "ExcelExportRequest",
    "EXPORT_COLUMN_TYPE_IMAGE",
    "EXPORT_COLUMN_TYPE_STRING",
    "RowData",
    "SheetData",
]
