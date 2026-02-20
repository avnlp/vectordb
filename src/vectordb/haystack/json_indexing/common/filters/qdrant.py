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

    must_conditions = []
    must_not_conditions = []

    for key, value in filters.items():
        if isinstance(value, dict):
            for op, op_value in value.items():
                if op == "$eq":
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=op_value))
                    )
                elif op == "$ne":
                    must_not_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=op_value))
                    )
                elif op == "$gt":
                    must_conditions.append(
                        FieldCondition(key=key, range=Range(gt=op_value))
                    )
                elif op == "$gte":
                    must_conditions.append(
                        FieldCondition(key=key, range=Range(gte=op_value))
                    )
                elif op == "$lt":
                    must_conditions.append(
                        FieldCondition(key=key, range=Range(lt=op_value))
                    )
                elif op == "$lte":
                    must_conditions.append(
                        FieldCondition(key=key, range=Range(lte=op_value))
                    )
        else:
            # Simple equality
            must_conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )

    if not must_conditions and not must_not_conditions:
        return None

    return Filter(
        must=must_conditions or None,
        must_not=must_not_conditions or None,
    )
