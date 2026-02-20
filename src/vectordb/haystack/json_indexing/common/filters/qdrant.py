"""Qdrant-specific filter translation."""

from typing import Any

from qdrant_client.http.models import FieldCondition, Filter, MatchValue, Range


def build_qdrant_filter(filters: dict[str, Any] | None) -> Filter | None:
    """Convert dict filters to Qdrant Filter object.

    Args:
        filters: Dictionary of filter conditions.

    Returns:
        Qdrant Filter object or None if no filters.
    """
    if not filters:
        return None

    conditions = []

    for key, value in filters.items():
        if isinstance(value, dict):
            for op, op_value in value.items():
                if op == "$eq":
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=op_value))
                    )
                elif op == "$ne":
                    # Qdrant doesn't have native $ne, skip for now
                    pass
                elif op == "$gt":
                    conditions.append(FieldCondition(key=key, range=Range(gt=op_value)))
                elif op == "$gte":
                    conditions.append(
                        FieldCondition(key=key, range=Range(gte=op_value))
                    )
                elif op == "$lt":
                    conditions.append(FieldCondition(key=key, range=Range(lt=op_value)))
                elif op == "$lte":
                    conditions.append(
                        FieldCondition(key=key, range=Range(lte=op_value))
                    )
        else:
            # Simple equality
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

    if not conditions:
        return None

    if len(conditions) == 1:
        return Filter(must=conditions)

    return Filter(must=conditions)
