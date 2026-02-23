"""Document ID utilities for consistent ID handling across vectordb wrappers.

This module provides centralized document ID management to ensure consistent
behavior across all vector database integrations. Document IDs are critical for
upsert operations, deduplication, and linking retrieval results back to source
documents.

Key Functions:
    - get_doc_id: Extract ID from Haystack Document with fallback to metadata or UUID
    - set_doc_id: Set ID on both doc.id and doc.meta["doc_id"] for consistency
    - coerce_id: Convert any value (int, UUID, string) to string representation

ID Resolution Priority:
    1. doc.id attribute (primary source)
    2. doc.meta[fallback_key] (legacy/alternate storage)
    3. Auto-generated UUID4 (fallback when no ID exists)

Design Notes:
    Setting IDs in both locations (doc.id and doc.meta["doc_id"]) ensures that IDs
    survive serialization and round-trip through different storage backends that
    may only preserve metadata fields.

Usage:
    >>> from vectordb.utils.ids import get_doc_id, set_doc_id, coerce_id
    >>> doc_id = get_doc_id(haystack_doc)
    >>> set_doc_id(haystack_doc, "custom-id-123")
    >>> str_id = coerce_id(12345)  # Returns "12345"
"""

from typing import Any
from uuid import uuid4


def get_doc_id(doc: Any, fallback_meta_key: str = "doc_id") -> str:
    """Extract document ID from Haystack Document.

    Priority:
        1. doc.id (if set)
        2. doc.meta[fallback_meta_key] (if exists)
        3. Generate UUID

    Args:
        doc: Haystack Document.
        fallback_meta_key: Meta key to check if doc.id is not set.

    Returns:
        String document ID.
    """
    # Prefer doc.id
    if hasattr(doc, "id") and doc.id:
        return str(doc.id)

    # Fallback to meta
    if hasattr(doc, "meta") and doc.meta and fallback_meta_key in doc.meta:
        return str(doc.meta[fallback_meta_key])

    return str(uuid4())


def coerce_id(value: Any) -> str:
    """Coerce any value to string ID.

    Args:
        value: ID value (int, str, UUID, etc.)

    Returns:
        String representation.
    """
    return str(value) if value is not None else str(uuid4())


def set_doc_id(doc: Any, doc_id: str) -> None:
    """Set document ID on Haystack Document.

    Sets both doc.id and doc.meta["doc_id"] for consistency.

    Args:
        doc: Haystack Document to modify.
        doc_id: ID value to set.
    """
    doc.id = doc_id
    if not hasattr(doc, "meta") or doc.meta is None:
        doc.meta = {}
    doc.meta["doc_id"] = doc_id
