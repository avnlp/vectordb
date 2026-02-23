"""Weaviate-specific filter translation."""

from typing import Any

from weaviate.classes.query import Filter


def build_weaviate_filter(filters: dict[str, Any] | None) -> Filter | None:
    """Convert dict filters to Weaviate Filter object.

    Args:
        filters: Dictionary of filter conditions.

    Returns:
        Weaviate Filter object or None if no filters.
    """
    if not filters:
        return None

    # Start with first filter
    filter_obj = None

    for key, value in filters.items():
        if isinstance(value, dict):
            for op, op_value in value.items():
                if op == "$eq":
                    f = Filter.by_property(key).equal(op_value)
                elif op == "$ne":
                    f = Filter.by_property(key).not_equal(op_value)
                elif op == "$gt":
                    f = Filter.by_property(key).greater_than(op_value)
                elif op == "$gte":
                    f = Filter.by_property(key).greater_or_equal(op_value)
                elif op == "$lt":
                    f = Filter.by_property(key).less_than(op_value)
                elif op == "$lte":
                    f = Filter.by_property(key).less_or_equal(op_value)
                else:
                    continue

                filter_obj = f if filter_obj is None else filter_obj & f
        else:
            # Simple equality
            f = Filter.by_property(key).equal(value)
            filter_obj = f if filter_obj is None else filter_obj & f

    return filter_obj
