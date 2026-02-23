"""Metadata handling utilities for JSON indexing pipelines."""

from typing import Any


def flatten_metadata(doc_meta: dict[str, Any]) -> dict[str, Any]:
    """Flatten document metadata for vector databases.

    Converts only primitive types (str, int, float, bool); converts
    other types to strings and skips None values.

    Args:
        doc_meta: Dictionary of metadata from a Document.

    Returns:
        Flattened dictionary with only primitive and stringified values.
    """
    flat_meta = {}
    for key, value in doc_meta.items():
        if isinstance(value, (str, int, float, bool)):
            flat_meta[key] = value
        elif value is not None:
            flat_meta[key] = str(value)
    return flat_meta
