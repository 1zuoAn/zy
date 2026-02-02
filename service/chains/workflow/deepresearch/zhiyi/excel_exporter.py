# -*- coding: utf-8 -*-
# @Author   : xujiajun@zhiyitech.cn
# @Time     : 2026/1/26
# @File     : excel_exporter.py
"""
知衣数据洞察 Excel 导出器

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
from app.service.chains.workflow.deepresearch.zhiyi.schema import (
    ZhiyiAggregatedData,
    ZhiyiThinkingApiParseParam,
)


class ZhiyiExcelExporter:
    """知衣数据洞察 Excel 导出器"""

    def __init__(self, aggregated_data: ZhiyiAggregatedData, param: ZhiyiThinkingApiParseParam, is_thinking: bool):
        self.data = aggregated_data
        self.param = param
        self.is_thinking = is_thinking

    def build_excel_data_list(self) -> List[ExcelData]:
        """构建多个 ExcelData，每个只包含一个 Sheet"""
        excel_list: List[ExcelData] = []

        is_shop = self.data.is_shop
        table_type = self.data.table_type
        dimension_type = self.data.hydc_dimension_type

        # 1. 销售趋势（所有分支都有）
        if self.data.overview_volume and self.data.overview_volume.result:
            excel_list.append(self._build_sales_trend_excel())

        # 2. 品类趋势（仅 table_type=1）
        if table_type == "1" and self.data.category_data and self.data.category_data.result:
            excel_list.append(self._build_category_excel())

        # 3. 价格带分布（table_type=1 或 table_type=3）
        if self.data.price_data and self.data.price_data.result:
            if table_type == "1" or table_type == "3":
                excel_list.append(self._build_price_excel())

        # 4. 颜色分布
        if self.data.color_data and self.data.color_data.result:
            if table_type == "1" or (table_type == "2" and dimension_type == 1):
                excel_list.append(self._build_color_excel())

        # 5. 属性分布（table_type=2）
        if self.data.property_data and self.data.property_data.result:
            if table_type == "2":
                if is_shop or (not is_shop and dimension_type == 0):
                    excel_list.append(self._build_property_excel())

        # 6. 品牌分布（仅大盘品牌分析）
        if self.data.brand_data and self.data.brand_data.result:
            if not is_shop and table_type == "2" and dimension_type == 2:
                excel_list.append(self._build_brand_excel())

        # 7. Top10商品（所有分支都有）
        if self.is_thinking and self.data.top_goods and self.data.top_goods.result:
            excel_list.append(self._build_top_goods_excel())

        return excel_list

    def _generate_filename(self, data_type: str) -> str:
        """生成 Excel 文件名

        格式:
        - 店铺: {店铺名}{品类}{月份}{数据类型}.xlsx
        - 大盘: {品类}{月份}{数据类型}.xlsx
        """
        shop_prefix = f"{self.data.shop_name}" if self.data.is_shop and self.data.shop_name else ""
        category_name = self.param.category_id_name or self.param.root_category_id_name or "品类"
        date_range = f"{self.param.start_date}~{self.param.end_date}"
        return f"{shop_prefix}{category_name}{date_range}{data_type}.xlsx"

    # ========== 独立 Excel 构建方法（每个 Excel 包含一个 Sheet） ==========

    def _build_sales_trend_excel(self) -> ExcelData:
        """构建销售趋势 Excel（合并销量和销售额）"""
        sheet = self._build_sales_trend_sheet()
        sheet.sheet_active = True
        return ExcelData(
            xlsx_name=self._generate_filename("销售趋势"),
            xlsx_data=[sheet],
        )

    def _build_category_excel(self) -> ExcelData:
        """构建品类趋势 Excel"""
        sheet = self._build_category_sheet()
        sheet.sheet_active = True
        return ExcelData(
            xlsx_name=self._generate_filename("品类趋势"),
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

    def _build_color_excel(self) -> ExcelData:
        """构建颜色分布 Excel"""
        sheet = self._build_color_sheet()
        sheet.sheet_active = True
        return ExcelData(
            xlsx_name=self._generate_filename("颜色分布"),
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

    def _build_brand_excel(self) -> ExcelData:
        """构建品牌分布 Excel"""
        sheet = self._build_brand_sheet()
        sheet.sheet_active = True
        return ExcelData(
            xlsx_name=self._generate_filename("品牌分布"),
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

    # ========== Sheet 构建方法 ==========

    def _build_sales_trend_sheet(self) -> SheetData:
        """构建销售趋势 Sheet（合并销量和销售额）

        列结构: 日期 | 销量 | 销售额(元)
        数据来源: overview_volume.result.trend_dtos 和 overview_amount.result.trend_dtos
        通过 insert_date 合并两个列表的数据
        """
        columns = [
            ColumnInfo(col_name="日期", col_id=1, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销量", col_id=2, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销售额(元)", col_id=3, col_type=EXPORT_COLUMN_TYPE_STRING),
        ]

        rows: List[RowData] = []

        # 获取销量趋势和销售额趋势数据
        volume_result = self.data.overview_volume.result if self.data.overview_volume else None
        amount_result = self.data.overview_amount.result if self.data.overview_amount else None

        volume_trend = volume_result.trend_dtos if volume_result else []
        amount_trend = amount_result.trend_dtos if amount_result else []

        # 构建日期到销售额的映射
        amount_map: dict[str, int] = {}
        for item in amount_trend:
            if item.insert_date:
                amount_map[item.insert_date] = item.daily_sum_day_sale

        # 遍历销量趋势，合并销售额数据
        for row_idx, item in enumerate(volume_trend, start=1):
            date_str = item.insert_date or ""
            sale_volume = item.daily_sum_day_sales_volume
            sale_amount_fen = amount_map.get(date_str, 0)

            # 价格从分转换为元
            sale_amount_yuan = sale_amount_fen / 100 if sale_amount_fen else 0

            cells = [
                CellData(col_id=1, cell_value=date_str),
                CellData(col_id=2, cell_value=str(sale_volume)),
                CellData(col_id=3, cell_value=f"{sale_amount_yuan:.2f}"),
            ]
            rows.append(RowData(row_id=row_idx, row_data=cells))

        return SheetData(
            sheet_name="销售趋势",
            sheet_active=False,
            sheet_info=columns,
            sheet_data=rows,
        )

    def _build_category_sheet(self) -> SheetData:
        """构建品类趋势 Sheet

        列结构: 排名 | 品类路径 | 销量 | 销售额(元) | 销量占比 | 销售额占比
        数据来源: category_data.result.top10_slim
        """
        columns = [
            ColumnInfo(col_name="排名", col_id=1, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="品类路径", col_id=2, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销量", col_id=3, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销售额(元)", col_id=4, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销量占比", col_id=5, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销售额占比", col_id=6, col_type=EXPORT_COLUMN_TYPE_STRING),
        ]

        rows: List[RowData] = []
        category_result = self.data.category_data.result if self.data.category_data else None
        category_items = category_result.top10_slim if category_result else []

        for row_idx, item in enumerate(category_items, start=1):
            # 价格从分转换为元
            sale_amount_yuan = item.sum_day_sales_amount / 100 if item.sum_day_sales_amount else 0

            # 占比转换为百分比
            volume_rate = f"{item.sum_day_sales_volume_rate * 100:.2f}%" if item.sum_day_sales_volume_rate else "-"
            amount_rate = f"{item.sum_day_sales_amount_rate * 100:.2f}%" if item.sum_day_sales_amount_rate else "-"

            cells = [
                CellData(col_id=1, cell_value=str(item.rank)),
                CellData(col_id=2, cell_value=item.category_path or ""),
                CellData(col_id=3, cell_value=str(item.sum_day_sales_volume)),
                CellData(col_id=4, cell_value=f"{sale_amount_yuan:.2f}"),
                CellData(col_id=5, cell_value=volume_rate),
                CellData(col_id=6, cell_value=amount_rate),
            ]
            rows.append(RowData(row_id=row_idx, row_data=cells))

        return SheetData(
            sheet_name="品类趋势",
            sheet_active=False,
            sheet_info=columns,
            sheet_data=rows,
        )

    def _build_price_sheet(self) -> SheetData:
        """构建价格带分布 Sheet

        列结构: 价格区间(元) | 销量 | 占比
        数据来源: price_data.result.price_slim
        """
        columns = [
            ColumnInfo(col_name="价格区间(元)", col_id=1, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销量", col_id=2, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="占比", col_id=3, col_type=EXPORT_COLUMN_TYPE_STRING),
        ]

        rows: List[RowData] = []
        price_result = self.data.price_data.result if self.data.price_data else None
        price_items = price_result.price_slim if price_result else []

        for row_idx, item in enumerate(price_items, start=1):
            # 价格从分转换为元
            left_yuan = item.left_price / 100 if item.left_price else 0
            right_yuan = item.right_price / 100 if item.right_price else None

            # 构建价格区间字符串 (最大范围适配xxx以上)
            if right_yuan is not None and right_yuan > 0 and right_yuan < 900000:
                price_range = f"{left_yuan:.0f}-{right_yuan:.0f}元"
            else:
                price_range = f"{left_yuan:.0f}元以上"

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

    def _build_color_sheet(self) -> SheetData:
        """构建颜色分布 Sheet

        列结构: 颜色 | 销量 | 销售额(元) | 占比
        数据来源: color_data.result.color_slim
        """
        columns = [
            ColumnInfo(col_name="颜色", col_id=1, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销量", col_id=2, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销售额(元)", col_id=3, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="占比", col_id=4, col_type=EXPORT_COLUMN_TYPE_STRING),
        ]

        rows: List[RowData] = []
        color_result = self.data.color_data.result if self.data.color_data else None
        color_items = color_result.color_slim if color_result else []

        for row_idx, item in enumerate(color_items, start=1):
            # 价格从分转换为元
            sale_amount_yuan = item.sales_amount / 100 if item.sales_amount else 0

            # 格式化占比
            rate_str = f"{item.rate * 100:.2f}%" if item.rate else "-"

            cells = [
                CellData(col_id=1, cell_value=item.property_value or ""),
                CellData(col_id=2, cell_value=str(item.sales_volume)),
                CellData(col_id=3, cell_value=f"{sale_amount_yuan:.2f}"),
                CellData(col_id=4, cell_value=rate_str),
            ]
            rows.append(RowData(row_id=row_idx, row_data=cells))

        return SheetData(
            sheet_name="颜色分布",
            sheet_active=False,
            sheet_info=columns,
            sheet_data=rows,
        )

    def _build_property_sheet(self) -> SheetData:
        """构建属性分布 Sheet

        列结构: {属性名} | 销量 | 销售额(元) | 占比
        数据来源: property_data.result.color_slim（复用ColorCleanResponse结构）
        """
        # 使用动态属性名作为第一列列名
        property_col_name = self.data.property_name or "属性值"

        columns = [
            ColumnInfo(col_name=property_col_name, col_id=1, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销量", col_id=2, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销售额(元)", col_id=3, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="占比", col_id=4, col_type=EXPORT_COLUMN_TYPE_STRING),
        ]

        rows: List[RowData] = []
        property_result = self.data.property_data.result if self.data.property_data else None
        property_items = property_result.color_slim if property_result else []

        for row_idx, item in enumerate(property_items, start=1):
            # 价格从分转换为元
            sale_amount_yuan = item.sales_amount / 100 if item.sales_amount else 0

            # 格式化占比
            rate_str = f"{item.rate * 100:.2f}%" if item.rate else "-"

            cells = [
                CellData(col_id=1, cell_value=item.property_value or ""),
                CellData(col_id=2, cell_value=str(item.sales_volume)),
                CellData(col_id=3, cell_value=f"{sale_amount_yuan:.2f}"),
                CellData(col_id=4, cell_value=rate_str),
            ]
            rows.append(RowData(row_id=row_idx, row_data=cells))

        return SheetData(
            sheet_name="属性分布",
            sheet_active=False,
            sheet_info=columns,
            sheet_data=rows,
        )

    def _build_brand_sheet(self) -> SheetData:
        """构建品牌分布 Sheet

        列结构: 品牌 | 销量 | 销售额(元) | 占比
        数据来源: brand_data.result.brand_slim
        """
        columns = [
            ColumnInfo(col_name="品牌", col_id=1, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销量", col_id=2, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销售额(元)", col_id=3, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="占比", col_id=4, col_type=EXPORT_COLUMN_TYPE_STRING),
        ]

        rows: List[RowData] = []
        brand_result = self.data.brand_data.result if self.data.brand_data else None
        brand_items = brand_result.brand_slim if brand_result else []

        for row_idx, item in enumerate(brand_items, start=1):
            # 价格从分转换为元
            sale_amount_yuan = item.sales_amount / 100 if item.sales_amount else 0

            # 格式化占比
            rate_str = f"{item.rate * 100:.2f}%" if item.rate else "-"

            cells = [
                CellData(col_id=1, cell_value=item.brand_name or ""),
                CellData(col_id=2, cell_value=str(item.sales_volume)),
                CellData(col_id=3, cell_value=f"{sale_amount_yuan:.2f}"),
                CellData(col_id=4, cell_value=rate_str),
            ]
            rows.append(RowData(row_id=row_idx, row_data=cells))

        return SheetData(
            sheet_name="品牌分布",
            sheet_active=False,
            sheet_info=columns,
            sheet_data=rows,
        )

    def _build_top_goods_sheet(self) -> SheetData:
        """构建 Top10 商品 Sheet

        列结构: 排名 | 商品图片 | 商品标题 | 品类 | 销量 | 销售额(元) | 价格(元)
        数据来源: top_goods.result.top10_slim
        """
        columns = [
            ColumnInfo(col_name="排名", col_id=1, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="商品图片", col_id=2, col_type=EXPORT_COLUMN_TYPE_IMAGE),
            ColumnInfo(col_name="商品标题", col_id=3, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="品类", col_id=4, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销量", col_id=5, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="销售额(元)", col_id=6, col_type=EXPORT_COLUMN_TYPE_STRING),
            ColumnInfo(col_name="价格(元)", col_id=7, col_type=EXPORT_COLUMN_TYPE_STRING),
        ]

        rows: List[RowData] = []
        top_goods_result = self.data.top_goods.result if self.data.top_goods else None
        top_goods_items = top_goods_result.top10_slim if top_goods_result else []

        for row_idx, item in enumerate(top_goods_items, start=1):
            # 价格从分转换为元
            sale_amount_yuan = item.sales_amount / 100 if item.sales_amount else 0
            min_price_yuan = item.min_price / 100 if item.min_price else 0
            max_price_yuan = item.max_s_price / 100 if item.max_s_price else None

            # 构建价格字符串
            if max_price_yuan and max_price_yuan > min_price_yuan:
                price_str = f"{min_price_yuan:.2f}-{max_price_yuan:.2f}"
            else:
                price_str = f"{min_price_yuan:.2f}"

            cells = [
                CellData(col_id=1, cell_value=str(row_idx)),
                CellData(col_id=2, cell_value=item.pic_url or ""),
                CellData(col_id=3, cell_value=item.title or ""),
                CellData(col_id=4, cell_value=item.category_name or ""),
                CellData(col_id=5, cell_value=str(item.sales_volume)),
                CellData(col_id=6, cell_value=f"{sale_amount_yuan:.2f}"),
                CellData(col_id=7, cell_value=price_str),
            ]
            rows.append(RowData(row_id=row_idx, row_data=cells))

        return SheetData(
            sheet_name="Top10商品",
            sheet_active=False,
            sheet_info=columns,
            sheet_data=rows,
        )
