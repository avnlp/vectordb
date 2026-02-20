"""Result merging and fusion strategies for hybrid search.

Implements Reciprocal Rank Fusion (RRF), weighted fusion, and other ranking strategies
to combine results from dense and sparse retrievers.

Hybrid Search Background:
    Dense retrieval (embeddings) excels at semantic matching but struggles with
    exact keyword matches. Sparse retrieval (BM25) excels at keyword matching
    but misses semantic relationships. Hybrid search combines both approaches.

Fusion Strategies:
    - RRF (Reciprocal Rank Fusion): Combines rankings without requiring scores.
      Uses formula: score = sum(1 / (k + rank)) for each list containing the doc.
      Default k=60 per Cormack et al. research.
    - Weighted Fusion: Combines normalized scores with configurable weights.
      Requires score normalization since different retrievers use different scales.

Design Considerations:
    - Documents are deduplicated using stable IDs (SHA1 of normalized content)
    - RRF is preferred when retrievers return different score scales
    - Weighted fusion is preferred when you want to emphasize one retriever

Usage:
    >>> from vectordb.haystack.components import ResultMerger
    >>> # RRF fusion
    >>> fused = ResultMerger.rrf_fusion(dense_docs, sparse_docs, k=60, top_k=10)
    >>> # Weighted fusion
    >>> fused = ResultMerger.weighted_fusion(
    ...     dense_docs, sparse_docs, dense_weight=0.7, sparse_weight=0.3
    ... )
"""

import hashlib
import logging
from typing import Any

from haystack import Document


logger = logging.getLogger(__name__)


class ResultMerger:
    """Merge and fuse retrieval results from multiple sources.

    Supports:
    - Reciprocal Rank Fusion (RRF): Best for combining dense + sparse
    - Weighted Sum: Weighted combination of relevance scores
    - Simple Concat: Deduplicate and rank by occurrence

    All methods are static since ResultMerger maintains no state.
    This design allows easy integration into Haystack pipelines as
    a utility class rather than a component with lifecycle management.

    Key Design Decisions:
        - Uses SHA1 for stable document IDs (Python's hash() is randomized)
        - RRF k=60 is the default based on Cormack et al. research
        - Score normalization uses min-max scaling to [0, 1]

    Note:
        This class is stateless and thread-safe. All methods can be
        called concurrently without side effects.
    """

    @staticmethod
    def stable_doc_id(doc: Document) -> str:
        """Generate a stable document ID.

        Uses SHA1 hash of normalized content for stability across processes.
        Python's hash() is randomized per process, so we use SHA1.

        Args:
            doc: Haystack Document.

        Returns:
            Stable identifier string.
        """
        if doc.meta and doc.meta.get("doc_id"):
            return str(doc.meta["doc_id"])
        if doc.id:
            return doc.id

        content = (doc.content or "").strip().lower()
        return hashlib.sha1(content.encode(), usedforsecurity=False).hexdigest()

    @staticmethod
    def rrf_fusion(
        dense_docs: list[Document],
        sparse_docs: list[Document],
        k: int = 60,
        top_k: int | None = None,
    ) -> list[Document]:
        """Reciprocal Rank Fusion.

        Args:
            dense_docs: Documents from dense retriever (ordered by relevance).
            sparse_docs: Documents from sparse retriever (ordered by relevance).
            k: RRF parameter (constant added to rank).
            top_k: Return top K documents (default: max of input lengths).

        Returns:
            Fused and reranked documents.
        """
        if not dense_docs and not sparse_docs:
            logger.warning("Both dense_docs and sparse_docs are empty")
            return []

        rrf_scores: dict[str, float] = {}

        for rank, doc in enumerate(dense_docs, 1):
            doc_id = ResultMerger.stable_doc_id(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

        for rank, doc in enumerate(sparse_docs, 1):
            doc_id = ResultMerger.stable_doc_id(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

        doc_map = {}
        for doc in dense_docs + sparse_docs:
            doc_id = ResultMerger.stable_doc_id(doc)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        sorted_docs = [
            doc_map[doc_id]
            for doc_id in sorted(
                rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
            )
            if doc_id in doc_map
        ]

        if top_k is None:
            top_k = max(len(dense_docs), len(sparse_docs))

        result = sorted_docs[:top_k]
        logger.info(
            "RRF fusion: %d dense + %d sparse → %d results",
            len(dense_docs),
            len(sparse_docs),
            len(result),
        )
        return result

    @staticmethod
    def rrf_fusion_many(
        ranked_lists: list[list[Document]],
        k: int = 60,
        top_k: int | None = None,
    ) -> list[Document]:
        """Reciprocal Rank Fusion for N ranked lists.

        RRF score = sum(1 / (k + rank)) across all lists where document appears.

        Args:
            ranked_lists: List of document lists, each ordered by relevance.
            k: RRF constant (default 60 per original paper).
            top_k: Number of results to return (default: max list length).

        Returns:
            Fused and reranked documents.
        """
        if not ranked_lists:
            logger.warning("ranked_lists is empty")
            return []

        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for result_list in ranked_lists:
            for rank, doc in enumerate(result_list, start=1):
                doc_id = ResultMerger.stable_doc_id(doc)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

        # Sort by RRF score
        sorted_ids = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )
        fused_docs = [doc_map[doc_id] for doc_id in sorted_ids]

        if top_k is None:
            top_k = max(len(lst) for lst in ranked_lists) if ranked_lists else 10

        result = fused_docs[:top_k]
        logger.info(
            "N-way RRF fusion: %d lists → %d results",
            len(ranked_lists),
            len(result),
        )
        return result

    @staticmethod
    def weighted_fusion(
        dense_docs: list[Document],
        sparse_docs: list[Document],
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        top_k: int | None = None,
    ) -> list[Document]:
        """Weighted sum fusion.

        Args:
            dense_docs: Documents from dense retriever (with scores).
            sparse_docs: Documents from sparse retriever (with scores).
            dense_weight: Weight for dense scores (default 0.7).
            sparse_weight: Weight for sparse scores (default 0.3).
            top_k: Return top K documents.

        Returns:
            Fused and reranked documents.

        Raises:
            ValueError: If weights don't sum to 1.0.
        """
        total_weight = dense_weight + sparse_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning("Weights sum to %.2f, normalizing", total_weight)
            dense_weight /= total_weight
            sparse_weight /= total_weight

        if not dense_docs and not sparse_docs:
            logger.warning("Both dense_docs and sparse_docs are empty")
            return []

        def normalize_scores(
            docs: list[Document],
        ) -> dict[str, tuple[Document, float]]:
            """Normalize document scores to [0, 1] range using min-max scaling.

            Different retrievers use different score scales
            (e.g., cosine similarity [-1, 1] vs BM25 [0, infinity]).
            """
            if not docs:
                return {}

            scores = [getattr(doc, "score", 0.0) or 0.0 for doc in docs]
            min_score = min(scores) if scores else 0.0
            max_score = max(scores) if scores else 1.0

            result_map: dict[str, tuple[Document, float]] = {}
            for doc in docs:
                score = getattr(doc, "score", 0.0) or 0.0
                normalized = (
                    (score - min_score) / (max_score - min_score)
                    if max_score > min_score
                    else 0.5
                )
                doc_id = doc.id or doc.meta.get("doc_id", hash(doc.content))
                result_map[doc_id] = (doc, normalized)

            return result_map

        dense_map = normalize_scores(dense_docs)
        sparse_map = normalize_scores(sparse_docs)

        combined_scores: dict[str, float] = {}
        for doc_id, (_, score) in dense_map.items():
            combined_scores[doc_id] = (
                combined_scores.get(doc_id, 0.0) + dense_weight * score
            )

        for doc_id, (_, score) in sparse_map.items():
            combined_scores[doc_id] = (
                combined_scores.get(doc_id, 0.0) + sparse_weight * score
            )

        doc_map = {**dense_map, **sparse_map}
        sorted_docs = [
            doc_map[doc_id][0]
            for doc_id in sorted(
                combined_scores.keys(),
                key=lambda x: combined_scores[x],
                reverse=True,
            )
            if doc_id in doc_map
        ]

        if top_k is None:
            top_k = max(len(dense_docs), len(sparse_docs))

        result = sorted_docs[:top_k]
        logger.info(
            "Weighted fusion (%.1f/%.1f): %d dense + %d sparse → %d results",
            dense_weight,
            sparse_weight,
            len(dense_docs),
            len(sparse_docs),
            len(result),
        )
        return result

    @staticmethod
    def deduplicate_by_content(
        docs: list[Document],
        similarity_threshold: float = 0.95,
    ) -> list[Document]:
        """Deduplicate documents by content similarity.

        Args:
            docs: List of documents to deduplicate.
            similarity_threshold: Content similarity threshold for dedup.

        Returns:
            Deduplicated document list.
        """
        if not docs:
            return []

        unique_docs: list[Document] = []
        seen_contents: set[int] = set()

        for doc in docs:
            content_hash = hash(doc.content.lower().strip())

            if content_hash not in seen_contents:
                unique_docs.append(doc)
                seen_contents.add(content_hash)

        logger.info(
            "Deduplication: %d → %d documents",
            len(docs),
            len(unique_docs),
        )
        return unique_docs

    @staticmethod
    def validate_fusion_config(
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate fusion configuration.

        Args:
            config: Fusion configuration dictionary.

        Returns:
            Validated configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        if not isinstance(config, dict):
            raise ValueError(f"Config must be dict, got {type(config)}")

        fusion_type = config.get("type", "rrf").lower()
        if fusion_type not in ["rrf", "weighted"]:
            raise ValueError(
                f"Unsupported fusion type: {fusion_type}. Must be 'rrf' or 'weighted'"
            )

        top_k = config.get("top_k", 10)
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")

        return config
