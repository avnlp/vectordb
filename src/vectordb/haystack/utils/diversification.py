"""Sequential similarity count filter for Haystack retrieval results.

This module provides a lightweight deduplication utility that processes
documents in their current order and removes candidates that are too similar
to too many already-selected documents.

Algorithm:
    For each document (in original relevance order):
        1. Count how many selected docs have cosine_sim >= threshold.
        2. Keep the document only if count < max_similar_docs.

Note:
    This utility is not an MMR reranker and does not optimize a global
    relevance-diversity objective. Haystack MMR behavior is implemented in
    the diversity_filtering pipelines via SentenceTransformersDiversityRanker.
"""

import math
from typing import Any

from haystack import Document


class DiversificationHelper:
    """Helper class for sequential similarity count filtering."""

    @classmethod
    def apply(
        cls,
        documents: list[Document],
        config: dict[str, Any],
    ) -> list[Document]:
        """Apply sequential similarity count filtering to search results.

        Args:
            documents: List of retrieved documents with embeddings.
            config: Configuration dict with 'semantic_diversification' key.

        Returns:
            Filtered subset of documents.
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
        """Check whether a document passes the similarity count rule.

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
