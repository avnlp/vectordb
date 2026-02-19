"""Scope isolation utilities for multi-tenancy and namespace support.

This module provides utilities for implementing data isolation in multi-tenant
vector database deployments. Scoping ensures that queries and documents from one
tenant cannot access data belonging to another tenant, which is essential for
shared infrastructure scenarios.

Key Functions:
    - inject_scope_to_metadata: Add tenant/namespace field to document metadata
    - inject_scope_to_filter: Add scope condition to Haystack/MongoDB-style filters
    - build_scope_filter_expr: Build Milvus-style filter expression strings

Filter Formats:
    - Haystack/MongoDB-style: Uses $and/$eq operators for composition
    - Milvus-style: Uses SQL-like string expressions with 'and' operator

Usage Patterns:
    1. During indexing: Inject scope into document metadata before upsert
    2. During retrieval: Inject scope into query filters to restrict results

Usage:
    >>> from vectordb.utils.scope import (
    ...     inject_scope_to_metadata,
    ...     inject_scope_to_filter,
    ... )
    >>> metadata = inject_scope_to_metadata({"category": "news"}, "tenant_a")
    >>> # Result: {"category": "news", "tenant_id": "tenant_a"}
    >>> filters = inject_scope_to_filter({"category": "news"}, "tenant_a")
    >>> # Result: {"$and": [{"tenant_id": {"$eq": "tenant_a"}}, {"category": "news"}]}
"""

from typing import Any, Dict, Optional


def inject_scope_to_metadata(
    metadata: Optional[Dict[str, Any]],
    scope: Optional[str],
    scope_field: str = "tenant_id",
) -> Dict[str, Any]:
    """Inject scope value into metadata dict.

    Args:
        metadata: Existing metadata dict (or None).
        scope: Scope/tenant/namespace value to inject.
        scope_field: Field name for scope in metadata.

    Returns:
        Metadata dict with scope field added (if scope provided).
    """
    result = dict(metadata) if metadata else {}
    if scope:
        result[scope_field] = scope
    return result


def inject_scope_to_filter(
    filters: Optional[Dict[str, Any]],
    scope: Optional[str],
    scope_field: str = "tenant_id",
) -> Optional[Dict[str, Any]]:
    """Inject scope condition into canonical filter dict.

    Uses MongoDB-style filter format with $and operator.

    Args:
        filters: Existing filter dict (or None).
        scope: Scope/tenant/namespace value to require.
        scope_field: Field name for scope in filters.

    Returns:
        Filter dict with scope condition added, or None if both inputs are None/empty.

    Example:
        >>> inject_scope_to_filter({"category": "news"}, "tenant_a")
        {"$and": [{"tenant_id": {"$eq": "tenant_a"}}, {"category": "news"}]}
    """
    if not scope:
        return filters

    scope_condition = {scope_field: {"$eq": scope}}

    if not filters:
        return scope_condition

    # Combine with existing filters using $and
    return {"$and": [scope_condition, filters]}


def build_scope_filter_expr(
    scope: Optional[str],
    scope_field: str = "namespace",
    existing_expr: str = "",
) -> str:
    """Build Milvus-style scope filter expression.

    Args:
        scope: Scope/namespace value.
        scope_field: Field name for partition key.
        existing_expr: Existing filter expression to combine with.

    Returns:
        Combined filter expression string.

    Example:
        >>> build_scope_filter_expr("tenant_a", "namespace", 'category == "news"')
        '(category == "news") and namespace == "tenant_a"'
    """
    if not scope:
        return existing_expr

    scope_expr = f'{scope_field} == "{scope}"'

    if not existing_expr:
        return scope_expr

    return f"({existing_expr}) and {scope_expr}"
