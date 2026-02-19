"""Semantic diversification for Haystack retrieval results.

This module provides diversification utilities for reducing redundancy in
Haystack pipeline results. Diversification filters out documents that are
too semantically similar to already-selected documents.

Algorithm:
    The diversification algorithm iterates through documents in their original
    order (typically ranked by relevance) and includes each document only if
    it is sufficiently dissimilar to all previously-selected documents.

    Similarity is measured using cosine similarity between document embeddings.
    A document is rejected if it exceeds the similarity threshold with more
    than max_similar_docs already-selected documents.

Configuration:
    Diversification is controlled by a configuration dictionary:
        semantic_diversification:
          enabled: true  # Enable/disable diversification
          diversity_threshold: 0.7  # Cosine similarity threshold (0-1)
          max_similar_docs: 2  # Max similar docs before rejection

Use Cases:
    - Post-retrieval filtering to reduce near-duplicate results
    - Preprocessing for summarization to avoid repetitive content
    - Improving result variety for exploratory queries

Usage:
    >>> from vectordb.haystack.utils import DiversificationHelper
    >>> config = {"semantic_diversification": {"enabled": True, "threshold": 0.7}}
    >>> diversified = DiversificationHelper.apply(documents, config)
"""

import math
from typing import Any

from haystack import Document


class DiversificationHelper:
    """Helper class for semantic diversification of search results.

    Filters documents based on embedding similarity to reduce redundancy.
    Documents are processed in order, with each candidate checked against
    all previously-selected documents.
    """

    @classmethod
    def apply(
        cls,
        documents: list[Document],
        config: dict[str, Any],
    ) -> list[Document]:
        """Apply semantic diversification to search results.

        Args:
            documents: List of retrieved documents with embeddings.
            config: Configuration dict with 'semantic_diversification' key.

        Returns:
            Diversified subset of documents.
        """
        diversification_config = config.get("semantic_diversification", {})
        if not diversification_config.get("enabled", False):
            return documents

        threshold = diversification_config.get("diversity_threshold", 0.7)
        max_similar = diversification_config.get("max_similar_docs", 2)

        if not documents or not documents[0].embedding:
            return documents

        diversified: list[Document] = []
        for doc in documents:
            if cls._should_include(doc, diversified, threshold, max_similar):
                diversified.append(doc)

        return diversified

    @classmethod
    def _should_include(
        cls,
        doc: Document,
        selected: list[Document],
        threshold: float,
        max_similar: int,
    ) -> bool:
        """Check if document should be included in diversified results.

        Args:
            doc: Document to check.
            selected: Already-selected documents.
            threshold: Similarity threshold (0-1).
            max_similar: Maximum similar docs allowed.

        Returns:
            True if document should be included.
        """
        if not selected or not doc.embedding:
            return True

        similar_count = 0
        for selected_doc in selected:
            if not selected_doc.embedding:
                continue
            similarity = cls._cosine_similarity(doc.embedding, selected_doc.embedding)
            if similarity >= threshold:
                similar_count += 1

        return similar_count < max_similar

    @classmethod
    def _cosine_similarity(cls, vec_a: list[float], vec_b: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec_a: First vector.
            vec_b: Second vector.

        Returns:
            Cosine similarity score (0-1).
        """
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        magnitude_a = math.sqrt(sum(a**2 for a in vec_a))
        magnitude_b = math.sqrt(sum(b**2 for b in vec_b))

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)
