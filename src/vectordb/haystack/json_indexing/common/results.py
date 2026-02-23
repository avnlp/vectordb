"""Result normalization utilities for JSON indexing pipelines."""

from typing import Any


def normalize_search_results(
    raw_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize search results to standard format.

    Args:
        raw_results: Raw results from vector database search.

    Returns:
        Normalized results with consistent structure.
    """
    normalized = []
    for result in raw_results:
        normalized.append(
            {
                "id": result.get("id"),
                "score": result.get("score"),
                "content": result.get("content"),
                "metadata": result.get("metadata", {}),
            }
        )
    return normalized
