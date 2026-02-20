"""Pinecone-specific filter translation."""

from typing import Any


def build_pinecone_filter(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert dict filters to Pinecone filter format.

    Pinecone uses MongoDB-style filter syntax: {field: {$op: value}}

    Args:
        filters: Dictionary of filter conditions.

    Returns:
        Pinecone filter dict or None if no filters.
    """
    if not filters:
        return None

    pinecone_filter = {}

    for key, value in filters.items():
        if isinstance(value, dict):
            # Already in operator format
            pinecone_filter[key] = value
        else:
            # Simple equality
            pinecone_filter[key] = {"$eq": value}

    return pinecone_filter if pinecone_filter else None
