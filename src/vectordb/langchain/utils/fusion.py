"""Result fusion utilities for combining multi-source retrieval results.

This module provides algorithms for merging results from multiple retrieval
sources (e.g., dense and sparse retrievers, multiple vector stores, or different
query variations). Fusion is essential for hybrid search pipelines.

Fusion Strategies:
    Reciprocal Rank Fusion (RRF):
        RRF combines rankings from multiple sources using the formula:
            score(d) = Σ 1/(k + rank_i(d))
        where k is a constant (typically 60) and rank_i is the rank from source i.
        RRF is robust to score miscalibration between sources since it uses ranks
        rather than raw scores.

    Weighted Merge:
        Combines normalized rank-based scores with configurable weights per source.
        Useful when you know one source is more reliable than another.

When to Use Each Strategy:
    - RRF: Default choice for hybrid search. Works well when sources have different
      score distributions. More stable and parameter-free beyond k.
    - Weighted Merge: When you have prior knowledge about source reliability.
      For example, weighting dense retrieval higher for semantic queries.

Deduplication:
    ResultMerger provides deduplication to remove documents that appear in
    multiple source lists. By default, uses page_content for uniqueness.
    For more robust deduplication, specify a metadata key (e.g., document ID)
    using the dedup_key parameter to handle semantically identical documents
    with minor content differences.

Usage:
    >>> from vectordb.langchain.utils import ResultMerger
    >>> # Fuse dense and sparse retrieval results
    >>> fused = ResultMerger.reciprocal_rank_fusion([dense_results, sparse_results])
    >>> # Weighted fusion with dense emphasis
    >>> fused = ResultMerger.weighted_merge([dense, sparse], weights=[0.7, 0.3])
    >>> # Use metadata ID for robust deduplication
    >>> fused = ResultMerger.reciprocal_rank_fusion(
    ...     [dense_results, sparse_results], dedup_key="id"
    ... )
    >>> # Merge and deduplicate with custom metadata key
    >>> fused = ResultMerger.merge_and_deduplicate(
    ...     [results1, results2], method="rrf", dedup_key="doc_id"
    ... )
"""

from langchain_core.documents import Document


class ResultMerger:
    """Helper for merging and fusing multiple retrieval result sets.

    Provides static methods for various fusion strategies including RRF,
    weighted merge, and deduplication. All methods work with LangChain
    Document objects.
    """

    @staticmethod
    def reciprocal_rank_fusion(
        results_list: list[list[Document]],
        k: int = 60,
        weights: list[float] | None = None,
        dedup_key: str | None = None,
    ) -> list[Document]:
        """Merge results using Reciprocal Rank Fusion (RRF).

        Args:
            results_list: List of result sets from multiple searches.
            k: RRF parameter (default 60).
            weights: Optional weights for each result set (default equal weights).
            dedup_key: Optional metadata key for deduplication. If provided, uses
                doc.metadata[dedup_key] for uniqueness. Otherwise falls back to
                page_content.

        Returns:
            Merged list of documents sorted by RRF score.
        """
        if not results_list:
            return []

        if weights is None:
            weights = [1.0 / len(results_list)] * len(results_list)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Calculate RRF scores
        rrf_scores = {}
        doc_map = {}

        for result_set, weight in zip(results_list, weights):
            for rank, doc in enumerate(result_set, 1):
                # Use metadata key for uniqueness if provided, otherwise page_content
                if dedup_key:
                    key = doc.metadata.get(dedup_key)
                    if key is None:
                        key = doc.page_content
                else:
                    key = doc.page_content

                doc_map[key] = doc

                rrf_score = (weight * 1.0) / (k + rank)
                rrf_scores[key] = rrf_scores.get(key, 0) + rrf_score

        sorted_keys = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )

        return [doc_map[key] for key in sorted_keys]

    @staticmethod
    def weighted_merge(
        results_list: list[list[Document]],
        weights: list[float] | None = None,
        dedup_key: str | None = None,
    ) -> list[Document]:
        """Merge results with weighted scoring.

        Args:
            results_list: List of result sets from multiple searches.
            weights: Weights for each result set (default equal weights).
            dedup_key: Optional metadata key for deduplication. If provided, uses
                doc.metadata[dedup_key] for uniqueness. Otherwise falls back to
                page_content.

        Returns:
            Merged list of documents sorted by weighted score.
        """
        if not results_list:
            return []

        if weights is None:
            weights = [1.0 / len(results_list)] * len(results_list)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Calculate weighted scores
        weighted_scores = {}
        doc_map = {}

        for result_set, weight in zip(results_list, weights):
            # Normalize ranks: first document gets max score
            for rank, doc in enumerate(result_set):
                # Use metadata key for uniqueness if provided, otherwise page_content
                if dedup_key:
                    key = doc.metadata.get(dedup_key)
                    if key is None:
                        key = doc.page_content
                else:
                    key = doc.page_content

                doc_map[key] = doc

                # Score decreases with rank
                score = weight * max(0, 1.0 - (rank / max(len(result_set), 1)))
                weighted_scores[key] = weighted_scores.get(key, 0) + score

        sorted_keys = sorted(
            weighted_scores.keys(),
            key=lambda x: weighted_scores[x],
            reverse=True,
        )

        return [doc_map[key] for key in sorted_keys]

    @staticmethod
    def deduplication(
        documents: list[Document],
        key: str = "page_content",
    ) -> list[Document]:
        """Remove duplicate documents.

        Args:
            documents: List of documents to deduplicate.
            key: Key to use for deduplication ("page_content" or metadata key).

        Returns:
            Deduplicated list of documents.
        """
        seen = set()
        unique_docs = []

        for doc in documents:
            if key == "page_content":
                unique_key = doc.page_content
            else:
                unique_key = doc.metadata.get(key)

            if unique_key not in seen:
                seen.add(unique_key)
                unique_docs.append(doc)

        return unique_docs

    @staticmethod
    def merge_and_deduplicate(
        results_list: list[list[Document]],
        method: str = "rrf",
        weights: list[float] | None = None,
        dedup_key: str | None = None,
    ) -> list[Document]:
        """Merge results, deduplicate, and sort.

        Args:
            results_list: List of result sets from multiple searches.
            method: Merging method ("rrf" or "weighted").
            weights: Optional weights for each result set.
            dedup_key: Optional metadata key for deduplication. If provided, uses
                doc.metadata[dedup_key] for uniqueness. Otherwise falls back to
                page_content.

        Returns:
            Merged and deduplicated list of documents.
        """
        if method == "rrf":
            merged = ResultMerger.reciprocal_rank_fusion(
                results_list, weights=weights, dedup_key=dedup_key
            )
        elif method == "weighted":
            merged = ResultMerger.weighted_merge(
                results_list, weights=weights, dedup_key=dedup_key
            )
        else:
            msg = f"Unknown merge method: {method}"
            raise ValueError(msg)

        # Additional deduplication pass if a different key is needed
        if dedup_key:
            return ResultMerger.deduplication(merged, key=dedup_key)
        return merged
