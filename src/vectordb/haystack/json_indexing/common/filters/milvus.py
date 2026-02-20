"""Milvus-specific filter translation."""

from typing import Any


def _format_value(value: Any) -> str:
    """Format a value for Milvus filter expressions.

    Args:
        value: The value to format.

    Returns:
        Formatted value as string.
    """
    return f'"{value}"' if isinstance(value, str) else str(value)


def build_milvus_filter(
    filters: dict[str, Any] | None,
    json_field_name: str = "metadata",
) -> str | None:
    """Convert dict filters to Milvus expression string.

    Supports: equality, comparison, contains operators.

    Args:
        filters: Dictionary of filter conditions.
        json_field_name: Name of the JSON field in Milvus.

    Returns:
        Milvus filter expression string or None if no filters.
    """
    if not filters:
        return None

    expressions = []
    operator_map = {
        "$eq": "==",
        "$ne": "!=",
        "$gt": ">",
        "$gte": ">=",
        "$lt": "<",
        "$lte": "<=",
    }

    for key, value in filters.items():
        path_expr = f'{json_field_name}["{key}"]'

        if isinstance(value, dict):
            for op, op_value in value.items():
                if op == "$contains":
                    expressions.append(
                        f"json_contains({path_expr}, {_format_value(op_value)})"
                    )
                elif op in operator_map:
                    milvus_op = operator_map[op]
                    expressions.append(
                        f"{path_expr} {milvus_op} {_format_value(op_value)}"
                    )
        else:
            expressions.append(f"{path_expr} == {_format_value(value)}")

    return " and ".join(expressions) if expressions else None
