"""Chroma-specific filter translation."""

from typing import Any


def build_chroma_filter(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert dict filters to Chroma where format.

    Chroma uses MongoDB-style filter syntax similar to Pinecone.

    Args:
        filters: Dictionary of filter conditions.

    Returns:
        Chroma-compatible filter dictionary or None.
    """
    if not filters:
        return None

    conditions = []

    for key, value in filters.items():
        if isinstance(value, dict):
            for op, op_value in value.items():
                conditions.append({key: {op: op_value}})
        else:
            conditions.append({key: {"$eq": value}})

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions} if conditions else None
