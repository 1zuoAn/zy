# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2026/1/19 15:00
# @File     : excel_exporter.py
"""
抖衣数据洞察 Excel 导出器

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
from app.service.chains.workflow.deepresearch.douyi.schema import (
    DouyiAggregatedData,
    DouyiMainParseParam,
)


class DouyiExcelExporter:
    """抖衣数据洞察 Excel 导出器"""

    def __init__(self, aggregated_data: DouyiAggregatedData, param: DouyiMainParseParam, is_thinking: bool):
        self.data = aggregated_data
        self.param = param
        self.is_thinking = is_thinking

    def build_excel_data_list(self) -> List[ExcelData]:
        """构建多个 ExcelData，每个只包含一个 Sheet"""
        excel_list: List[ExcelData] = []

        # 根据数据可用性添加独立的 ExcelData
        # 销售趋势（合并销量和销售额）
        if self.data.sale_volume_data and self.data.sale_volume_data.result:
            excel_list.append(self._build_sales_trend_excel())
        if self.data.price_data and self.data.price_data.result:
            excel_list.append(self._build_price_excel())
        if self.data.property_data and self.data.property_data.result:
            excel_list.append(self._build_property_excel())
        if self.is_thinking and self.data.top10_products and self.data.top10_products.result:
            excel_list.append(self._build_top_goods_excel())

        return excel_list

    def _generate_filename(self, data_type: str) -> str:
        """生成 Excel 文件名（新命名风格）

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

        列结构: 日期 | 销量
        """
        columns = [
            ColumnInfo(col_name="日期", col_id=1, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销量", col_id=2, col_type=EXPORT_COLUMN_TYPE_STRING),
        ]

        rows: List[RowData] = []

        # 获取销量趋势数据
        volume_data = (
            self.data.sale_volume_data.result.trend_dtos
            if self.data.sale_volume_data and self.data.sale_volume_data.result
            else []
        )
        # 获取销售额趋势数据，构建日期到销售额的映射
        amount_map: dict[str, float] = {}
        if self.data.sale_amount_data and self.data.sale_amount_data.result:
            for item in self.data.sale_amount_data.result.trend_dtos:
                if item.granularity_date:
                    # 销售额从分转换为元
                    amount_yuan = item.daily_sum_day_sale / 100 if item.daily_sum_day_sale else 0
                    amount_map[item.granularity_date] = amount_yuan

        for row_idx, item in enumerate(volume_data, start=1):
            date_str = item.granularity_date or ""
            amount_yuan = amount_map.get(date_str, 0)

            cells = [
                CellData(col_id=1, cell_value=date_str),
                CellData(col_id=2, cell_value=str(item.daily_sum_day_sales_volume)),
            ]
            rows.append(RowData(row_id=row_idx, row_data=cells))

        return SheetData(
            sheet_name="销售趋势",
            sheet_active=False,
            sheet_info=columns,
            sheet_data=rows,
        )

    def _build_price_sheet(self) -> SheetData:
        """构建价格带分布 Sheet"""
        columns = [
            ColumnInfo(col_name="价格区间", col_id=1, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销量", col_id=2, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="占比", col_id=3, col_type=EXPORT_COLUMN_TYPE_STRING),
        ]

        rows: List[RowData] = []
        price_data = self.data.price_data.result.price_slim if self.data.price_data.result else []

        for row_idx, item in enumerate(price_data, start=1):
            # 价格从分转换为元
            left_yuan = item.left_price / 100 if item.left_price else 0
            right_yuan = item.right_price / 100 if item.right_price else None

            # 构建价格区间字符串
            if right_yuan is not None:
                price_range = f"{left_yuan:.0f}-{right_yuan:.0f}元"
            else:
                price_range = f"{left_yuan:.0f}元以上"

            # 格式化占比
            rate_str = f"{item.rate * 100:.2f}%" if item.rate is not None else "-"

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
        """构建属性分布 Sheet（属性分支专用）"""
        columns = [
            ColumnInfo(col_name="属性名", col_id=1, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="属性值", col_id=2, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销量", col_id=3, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="占比", col_id=4, col_type=EXPORT_COLUMN_TYPE_STRING),
        ]

        rows: List[RowData] = []
        property_data = self.data.property_data.result.property_slim if self.data.property_data.result else []

        for row_idx, item in enumerate(property_data, start=1):
            # 格式化占比
            rate_str = f"{item.rate * 100:.2f}%" if item.rate is not None else "-"

            cells = [
                CellData(col_id=1, cell_value=item.property_name or ""),
                CellData(col_id=2, cell_value=item.property_value or ""),
                CellData(col_id=3, cell_value=str(item.sales_volume)),
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
        """构建 Top10 商品 Sheet"""
        columns = [
            ColumnInfo(col_name="商品图片", col_id=1, col_type=EXPORT_COLUMN_TYPE_IMAGE),
            ColumnInfo(col_name="商品标题", col_id=2, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="品类", col_id=3, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销量", col_id=4, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销售额(元)", col_id=5, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="价格(元)", col_id=6, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="店铺", col_id=7, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="品牌", col_id=8, col_type=EXPORT_COLUMN_TYPE_STRING),
        ]

        rows: List[RowData] = []
        top_goods = self.data.top10_products.result.top10_slim if self.data.top10_products.result else []

        for row_idx, item in enumerate(top_goods, start=1):
            # 销售额和价格从分转换为元
            sales_amount_yuan = item.sales_amount / 100 if item.sales_amount else 0
            price_yuan = item.goods_price / 100 if item.goods_price else 0

            cells = [
                CellData(col_id=1, cell_value=item.pic_url or ""),
                CellData(col_id=2, cell_value=item.title or ""),
                CellData(col_id=3, cell_value=item.category_detail or ""),
                CellData(col_id=4, cell_value=str(item.sales_volume)),
                CellData(col_id=5, cell_value=f"{sales_amount_yuan:.2f}"),
                CellData(col_id=6, cell_value=f"{price_yuan:.2f}"),
                CellData(col_id=7, cell_value=item.shop_name or ""),
                CellData(col_id=8, cell_value=item.brand or ""),
            ]
            rows.append(RowData(row_id=row_idx, row_data=cells))

        return SheetData(
            sheet_name="Top10商品",
            sheet_active=False,
            sheet_info=columns,
            sheet_data=rows,
        )


__all__ = ["DouyiExcelExporter"]
