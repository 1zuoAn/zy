# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2026/1/19 18:21
# @File     : excel_exporter.py
"""
海外探款数据洞察 Excel 导出器

将工作流聚合数据转换为 Excel 导出格式，通过 Redis list 推送给后端处理。
"""
from __future__ import annotations

from datetime import datetime
from typing import List

from app.schemas.entities.export import (
    CellData,
    ColumnInfo,
    ExcelData,
    EXPORT_COLUMN_TYPE_IMAGE,
    EXPORT_COLUMN_TYPE_STRING,
    RowData,
    SheetData,
)
from app.service.chains.workflow.deepresearch.abroad.schema import (
    AbroadAggregatedData,
    AbroadMainParseParam,
)


class AbroadExcelExporter:
    """海外探款数据洞察 Excel 导出器"""

    def __init__(self, aggregated_data: AbroadAggregatedData, param: AbroadMainParseParam, is_thinking: bool):
        self.data = aggregated_data
        self.param = param
        self.is_thinking = is_thinking

    def build_excel_data_list(self) -> List[ExcelData]:
        """构建多个 ExcelData，每个只包含一个 Sheet"""
        excel_list: List[ExcelData] = []

        # 根据 table_type 和数据可用性添加独立的 ExcelData
        table_type = self.data.table_type

        # 销售趋势（合并销量和销售额）
        if self.data.trend_data and self.data.trend_data.result:
            excel_list.append(self._build_sales_trend_excel())

        # 品类分布（table_type=1 时）
        if table_type == 1 and self.data.category_data and self.data.category_data.result:
            excel_list.append(self._build_category_excel())

        # 颜色分布（table_type=1 时）
        if table_type == 1 and self.data.color_data and self.data.color_data.result:
            excel_list.append(self._build_color_excel())

        # 价格带分布（table_type=1 时）
        if table_type == 1 and self.data.price_data and self.data.price_data.result:
            excel_list.append(self._build_price_excel())

        # 属性分布（table_type=2 时）
        if table_type == 2 and self.data.property_data and self.data.property_data.result:
            excel_list.append(self._build_property_excel())

        # Top10商品
        if self.is_thinking and self.data.top_goods and self.data.top_goods.result:
            excel_list.append(self._build_top_goods_excel())

        return excel_list

    def _generate_filename(self, data_type: str) -> str:
        """生成 Excel 文件名

        格式: {品类}{月份}{数据类型}.xlsx
        示例: 连衣裙11月销售趋势.xlsx
        """
        category_name = self.param.category_id_name or self.param.root_category_id_name or "品类"
        date_range = f"{self.param.start_date}~{self.param.end_date}"
        return f"{category_name}{date_range}{data_type}.xlsx"

    def _build_sales_trend_excel(self) -> ExcelData:
        """构建销售趋势 Excel（合并销量和销售额）"""
        sheet = self._build_sales_trend_sheet()
        sheet.sheet_active = True
        return ExcelData(
            xlsx_name=self._generate_filename("销售趋势"),
            xlsx_data=[sheet],
        )

    def _build_category_excel(self) -> ExcelData:
        """构建品类分布 Excel"""
        sheet = self._build_category_sheet()
        sheet.sheet_active = True
        return ExcelData(
            xlsx_name=self._generate_filename("品类分布"),
            xlsx_data=[sheet],
        )

    def _build_color_excel(self) -> ExcelData:
        """构建颜色分布 Excel"""
        sheet = self._build_color_sheet()
        sheet.sheet_active = True
        return ExcelData(
            xlsx_name=self._generate_filename("颜色分布"),
            xlsx_data=[sheet],
        )

    def _build_price_excel(self) -> ExcelData:
        """构建价格带分布 Excel"""
        sheet = self._build_price_sheet()
        sheet.sheet_active = True
        return ExcelData(
            xlsx_name=self._generate_filename("价格带分布"),
            xlsx_data=[sheet],
        )

    def _build_property_excel(self) -> ExcelData:
        """构建属性分布 Excel"""
        sheet = self._build_property_sheet()
        sheet.sheet_active = True
        return ExcelData(
            xlsx_name=self._generate_filename("属性分布"),
            xlsx_data=[sheet],
        )

    def _build_top_goods_excel(self) -> ExcelData:
        """构建 Top10 商品 Excel"""
        sheet = self._build_top_goods_sheet()
        sheet.sheet_active = True
        return ExcelData(
            xlsx_name=self._generate_filename("Top10商品"),
            xlsx_data=[sheet],
        )

    def _build_sales_trend_sheet(self) -> SheetData:
        """构建销售趋势 Sheet（合并销量和销售额）

        列结构: 日期 | 销量 | 销售额(USD)
        数据来源: trend_data.result.sale_volume_trend 和 sale_amount_trend
        通过 record_date 合并两个列表的数据
        """
        columns = [
            ColumnInfo(col_name="日期", col_id=1, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销量", col_id=2, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销售额(USD)", col_id=3, col_type=EXPORT_COLUMN_TYPE_STRING),
        ]

        rows: List[RowData] = []

        # 获取销量趋势和销售额趋势数据
        result = self.data.trend_data.result
        volume_trend = result.sale_volume_trend if result else []
        amount_trend = result.sale_amount_trend if result else []

        # 构建日期到销售额的映射
        amount_map: dict[str, int] = {}
        for item in amount_trend:
            if item.record_date:
                amount_map[item.record_date] = item.sale_amount

        # 遍历销量趋势，合并销售额数据
        for row_idx, item in enumerate(volume_trend, start=1):
            date_str = item.record_date or ""
            sale_volume = item.sale_volume
            sale_amount = amount_map.get(date_str, 0)

            cells = [
                CellData(col_id=1, cell_value=date_str),
                CellData(col_id=2, cell_value=str(sale_volume)),
                CellData(col_id=3, cell_value=str(sale_amount)),
            ]
            rows.append(RowData(row_id=row_idx, row_data=cells))

        return SheetData(
            sheet_name="销售趋势",
            sheet_active=False,
            sheet_info=columns,
            sheet_data=rows,
        )

    def _build_category_sheet(self) -> SheetData:
        """构建品类分布 Sheet

        列结构: 品类 | 销量 | 销售额(USD) | 销量占比 | 销售额占比
        数据来源: category_data.result.items
        """
        columns = [
            ColumnInfo(col_name="品类", col_id=1, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销量", col_id=2, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销售额(USD)", col_id=3, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销量占比", col_id=4, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销售额占比", col_id=5, col_type=EXPORT_COLUMN_TYPE_STRING),
        ]

        rows: List[RowData] = []
        category_items = (
            self.data.category_data.result.items if self.data.category_data.result else []
        )

        for row_idx, item in enumerate(category_items, start=1):
            # 格式化占比
            volume_ratio_str = (
                f"{item.sale_volume_ratio * 100:.2f}%" if item.sale_volume_ratio else "-"
            )
            amount_ratio_str = (
                f"{item.sale_amount_ratio * 100:.2f}%" if item.sale_amount_ratio else "-"
            )

            cells = [
                CellData(col_id=1, cell_value=item.dimension_info or ""),
                CellData(col_id=2, cell_value=str(item.sale_volume)),
                CellData(col_id=3, cell_value=str(item.sale_amount)),
                CellData(col_id=4, cell_value=volume_ratio_str),
                CellData(col_id=5, cell_value=amount_ratio_str),
            ]
            rows.append(RowData(row_id=row_idx, row_data=cells))

        return SheetData(
            sheet_name="品类分布",
            sheet_active=False,
            sheet_info=columns,
            sheet_data=rows,
        )

    def _build_color_sheet(self) -> SheetData:
        """构建颜色分布 Sheet

        列结构: 颜色 | 销量 | 销售额(USD) | 占比
        数据来源: color_data.result
        """
        columns = [
            ColumnInfo(col_name="颜色", col_id=1, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销量", col_id=2, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销售额(USD)", col_id=3, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="占比", col_id=4, col_type=EXPORT_COLUMN_TYPE_STRING),
        ]

        rows: List[RowData] = []
        color_items = self.data.color_data.result if self.data.color_data else []

        for row_idx, item in enumerate(color_items, start=1):
            # 格式化占比
            rate_str = f"{item.rate * 100:.2f}%" if item.rate else "-"

            cells = [
                CellData(col_id=1, cell_value=item.property_value or ""),
                CellData(col_id=2, cell_value=str(item.sales_volume)),
                CellData(col_id=3, cell_value=str(item.sales_amount)),
                CellData(col_id=4, cell_value=rate_str),
            ]
            rows.append(RowData(row_id=row_idx, row_data=cells))

        return SheetData(
            sheet_name="颜色分布",
            sheet_active=False,
            sheet_info=columns,
            sheet_data=rows,
        )

    def _build_price_sheet(self) -> SheetData:
        """构建价格带分布 Sheet

        列结构: 价格区间(USD) | 销量 | 占比
        数据来源: price_data.result
        """
        columns = [
            ColumnInfo(col_name="价格区间(USD)", col_id=1, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销量", col_id=2, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="占比", col_id=3, col_type=EXPORT_COLUMN_TYPE_STRING),
        ]

        rows: List[RowData] = []
        price_items = self.data.price_data.result if self.data.price_data else []

        for row_idx, item in enumerate(price_items, start=1):
            # 价格已存储为分，需要转换为美元
            left_usd = item.left_price / 100 if item.left_price else 0
            right_usd = item.right_price / 100 if item.right_price else None

            # 构建价格区间字符串
            if right_usd is not None:
                price_range = f"${left_usd:.2f}-${right_usd:.2f}"
            else:
                price_range = f"${left_usd:.2f}以上"

            # 格式化占比
            rate_str = f"{item.rate * 100:.2f}%" if item.rate else "-"

            cells = [
                CellData(col_id=1, cell_value=price_range),
                CellData(col_id=2, cell_value=str(item.sales_volume)),
                CellData(col_id=3, cell_value=rate_str),
            ]
            rows.append(RowData(row_id=row_idx, row_data=cells))

        return SheetData(
            sheet_name="价格带分布",
            sheet_active=False,
            sheet_info=columns,
            sheet_data=rows,
        )

    def _build_property_sheet(self) -> SheetData:
        """构建属性分布 Sheet

        列结构: 属性值 | 销量 | 销售额(USD) | 占比
        数据来源: property_data.result
        """
        columns = [
            ColumnInfo(col_name="属性值", col_id=1, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销量", col_id=2, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销售额(USD)", col_id=3, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="占比", col_id=4, col_type=EXPORT_COLUMN_TYPE_STRING),
        ]

        rows: List[RowData] = []
        property_items = self.data.property_data.result if self.data.property_data else []

        for row_idx, item in enumerate(property_items, start=1):
            # 格式化占比
            rate_str = f"{item.rate * 100:.2f}%" if item.rate else "-"

            cells = [
                CellData(col_id=1, cell_value=item.property_value or ""),
                CellData(col_id=2, cell_value=str(item.sales_volume)),
                CellData(col_id=3, cell_value=str(item.sales_amount)),
                CellData(col_id=4, cell_value=rate_str),
            ]
            rows.append(RowData(row_id=row_idx, row_data=cells))

        return SheetData(
            sheet_name="属性分布",
            sheet_active=False,
            sheet_info=columns,
            sheet_data=rows,
        )

    def _build_top_goods_sheet(self) -> SheetData:
        """构建 Top10 商品 Sheet

        列结构: 排名 | 商品图片 | 商品标题 | 品类 | 销量(30天) | 销售额(USD) | 价格(USD)
        数据来源: top_goods.result.top10_slim
        """
        columns = [
            ColumnInfo(col_name="商品图片", col_id=1, col_type=EXPORT_COLUMN_TYPE_IMAGE),
            ColumnInfo(col_name="商品标题", col_id=2, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="品类", col_id=3, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销量(30天)", col_id=4, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销售额(USD)", col_id=5, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="价格(USD)", col_id=6, col_type=EXPORT_COLUMN_TYPE_STRING),
        ]

        rows: List[RowData] = []
        top_goods = (
            self.data.top_goods.result.top10_slim if self.data.top_goods.result else []
        )

        for row_idx, item in enumerate(top_goods, start=1):
            # 价格：使用 min_price 字段（已是字符串格式的美元价格）
            price_str = item.min_price or "-"

            cells = [
                CellData(col_id=1, cell_value=item.pic_url or ""),
                CellData(col_id=2, cell_value=item.title or ""),
                CellData(col_id=3, cell_value=item.category_detail or item.category_name or ""),
                CellData(col_id=4, cell_value=str(item.sales_volume)),
                CellData(col_id=5, cell_value=str(item.sales_amount)),
                CellData(col_id=6, cell_value=price_str),
            ]
            rows.append(RowData(row_id=row_idx, row_data=cells))

        return SheetData(
            sheet_name="Top10商品",
            sheet_active=False,
            sheet_info=columns,
            sheet_data=rows,
        )


__all__ = ["AbroadExcelExporter"]
