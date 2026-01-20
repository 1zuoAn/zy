"""
海外选品工作流辅助工具函数

提供参数规范化、JSON解析等通用功能，减少子图中的重复代码。
"""
import json
import re
from typing import Any


def normalize_category_list(category_list: Any) -> list[list[str]]:
    """
    规范化类目列表为二维数组格式

    Args:
        category_list: 类目列表，可能是以下格式之一：
            - None 或空列表: []
            - 二维数组: [["1", "341", "352"]]
            - 一维数组（逗号分隔）: ["1,341,352"]
            - 一维数组（普通）: ["1", "341", "352"]

    Returns:
        二维数组格式的类目列表，如 [["1", "341", "352"]]
        如果输入为空，返回空列表 []

    Examples:
        >>> normalize_category_list([["1", "341"]])
        [["1", "341"]]
        >>> normalize_category_list(["1,341,352"])
        [["1", "341", "352"]]
        >>> normalize_category_list(["1", "341"])
        [["1", "341"]]
        >>> normalize_category_list([])
        []
    """
    if not category_list:
        return []

    # 已经是二维数组
    if isinstance(category_list[0], list):
        return category_list

    # 一维数组，单个元素包含逗号分隔的ID
    if len(category_list) == 1 and ',' in category_list[0]:
        return [category_list[0].split(',')]

    # 一维数组，多个元素
    return [category_list]


def normalize_label_value(label_value: Any) -> list:
    """
    规范化标签值为列表格式

    Args:
        label_value: 标签值，可能是以下格式之一：
            - None 或空字符串: None, ""
            - 空列表: []
            - 二维列表: [["label1", "label2"]]
            - 一维列表: ["label1", "label2"]
            - 字符串（逗号分隔）: "label1,label2"

    Returns:
        规范化后的标签列表
        - 空值返回 [] (避免[""]导致API报错"参数传入不符合规则")
        - 二维列表保持不变
        - 一维列表包装为 [list]
        - 字符串分割后包装为 [[split_result]]

    Examples:
        >>> normalize_label_value(None)
        []
        >>> normalize_label_value([["label1"]])
        [["label1"]]
        >>> normalize_label_value(["label1", "label2"])
        [["label1", "label2"]]
        >>> normalize_label_value("label1,label2")
        [["label1", "label2"]]
    """
    # 空值处理 - 返回空列表,避免[""]导致API报错
    if not label_value or label_value == "":
        return []

    # 列表类型
    if isinstance(label_value, list):
        if not label_value:
            return []
        # 二维列表
        if isinstance(label_value[0], list):
            return label_value
        # 一维列表
        return [label_value]

    # 字符串类型（逗号分隔）
    return [str(label_value).split(",")]


def extract_json_from_llm_output(content: str, key: str | None = None) -> Any:
    """
    从LLM输出中提取JSON对象

    Args:
        content: LLM返回的文本内容，可能包含JSON对象
        key: 可选，要提取的JSON对象中的特定键

    Returns:
        - 如果指定了key，返回JSON对象中该键的值
        - 如果未指定key，返回整个解析后的JSON对象
        - 如果解析失败或key不存在，返回None

    Examples:
        >>> extract_json_from_llm_output('{"content_list": ["item1"]}', "content_list")
        ["item1"]
        >>> extract_json_from_llm_output('Some text {"data": 123} more text')
        {"data": 123}
    """
    try:
        # 使用正则表达式提取JSON对象
        json_match = re.search(r"\{[\s\S]*}", content)
        if not json_match:
            return None

        # 解析JSON
        parsed = json.loads(json_match.group())

        # 返回指定键的值或整个对象
        if key is not None:
            return parsed.get(key)
        return parsed

    except (json.JSONDecodeError, AttributeError):
        return None


def extract_result_count(result: Any) -> int:
    """
    从API响应中提取结果数量

    Args:
        result: API响应对象，应包含 result_count 或 result_list 属性

    Returns:
        结果数量，优先使用 result_count，否则使用 result_list 的长度

    Examples:
        >>> class MockResult:
        ...     result_count = 10
        ...     result_list = []
        >>> extract_result_count(MockResult())
        10
    """
    if hasattr(result, 'result_count') and result.result_count:
        return result.result_count
    if hasattr(result, 'result_list') and result.result_list:
        return len(result.result_list)
    return 0


