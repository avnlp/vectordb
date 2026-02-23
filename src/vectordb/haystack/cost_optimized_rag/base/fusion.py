"""Result fusion strategies for hybrid search cost optimization.

Combines results from sparse (cheap) and dense (expensive) retrieval methods
without requiring score normalization. Fusion enables using sparse retrieval
as a filter while preserving dense ranking quality for final results.

Fusion Strategy Comparison:

    Reciprocal Rank Fusion (RRF):
        - Combines rankings using: score = Σ 1/(k + rank)
        - No score normalization required (robust to different score scales)
        - Parameter k=60 balances high-ranked vs deep-ranked docs
        - Best when sparse and dense scores are incomparable
        - Computation: O(n) where n = total unique docs
        - Cost: Negligible (microseconds for typical top-100 lists)

    Weighted Fusion:
        - Linear combination: score = w₁×dense + w₂×sparse
        - Requires score normalization (min-max or z-score)
        - Configurable weights for quality/cost trade-off
        - Best when you have calibrated score distributions
        - Risk: Poor performance if one method has systematic bias

Cost Impact:
    - Fusion overhead is negligible compared to embedding costs
    - Enables 2-3x reduction in embedding computations via sparse pre-filtering
    - RRF is preferred for cost-optimal pipelines (no calibration needed)

Algorithm Notes:
    RRF is parameter-free except for k. Lower k (e.g., 20) emphasizes top
    rankings; higher k (e.g., 100) gives more weight to deep retrievals.
    Default k=60 works well for combining top-50 lists.
"""

from typing import Any


class ResultFuser:
    """Fuse results from multiple search methods for hybrid retrieval.

    Combines sparse and dense retrieval results to leverage the cost
    efficiency of sparse methods with the quality of dense embeddings.

    Design Rationale:
        Sparse retrieval (BM25/keyword) is cheap but misses semantic nuance.
        Dense retrieval (embeddings) is expensive but catches paraphrases.
        Fusion lets you use sparse to filter candidates, dense to rerank,
        achieving 70-90% of dense quality at 20-30% of the cost.

    Usage Pattern:
        1. Retrieve 100 candidates with BM25 (cheap)
        2. Retrieve 50 candidates with dense search (expensive)
        3. Fuse to get best 20 from combined candidate pool
        4. Optionally rerank with cross-encoder for final ordering

    Performance:
        - RRF: O(n log n) for sorting, negligible vs embedding time
        - Weighted: O(n) for score normalization and combination
        - Both suitable for real-time (<10ms for typical inputs)
    """

    @staticmethod
    def reciprocal_rank_fusion(
        dense_results: list[tuple[str, float]],
        sparse_results: list[tuple[str, float]],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """Fuse dense and sparse results using Reciprocal Rank Fusion (RRF).

        RRF score = Σ 1/(k + rank)
        Combines rankings without score normalization.

        Cost Optimization Logic:
            RRF enables using sparse results to expand the candidate pool
            without additional embedding costs. Documents retrieved by
            either method get credit, prioritizing consensus at top ranks.

        Args:
            dense_results: List of (doc_id, score) from dense search.
            sparse_results: List of (doc_id, score) from sparse search.
            k: RRF parameter (default 60).
                Lower values emphasize top-ranked documents.
                Higher values give more credit to deep retrievals.

        Returns:
            List of (doc_id, rrf_score) sorted by RRF score descending.
        """
        rrf_scores: dict[str, float] = {}

        # Dense results
        for rank, (doc_id, _) in enumerate(dense_results, 1):
            score = 1.0 / (k + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + score

        # Sparse results
        for rank, (doc_id, _) in enumerate(sparse_results, 1):
            score = 1.0 / (k + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + score

        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def weighted_fusion(
        dense_results: list[tuple[str, float]],
        sparse_results: list[tuple[str, float]],
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> list[tuple[str, float]]:
        """Fuse results using weighted combination.

        Linearly combines normalized scores from dense and sparse search.

        Cost Trade-off Configuration:
            Weights control reliance on expensive vs cheap methods:
            - dense_weight=0.9: Prioritize quality (higher embedding costs)
            - dense_weight=0.5: Balance (moderate costs, good quality)
            - dense_weight=0.1: Prioritize cost (sparse-heavy, lower quality)

        Args:
            dense_results: List of (doc_id, score) from dense search.
            sparse_results: List of (doc_id, score) from sparse search.
            dense_weight: Weight for dense scores (0-1).
                Higher values improve quality but increase costs
                (more reliance on expensive embedding retrieval).
            sparse_weight: Weight for sparse scores (0-1).
                Higher values reduce costs but may miss semantic matches.

        Returns:
            List of (doc_id, weighted_score) sorted descending.

        Raises:
            ValueError: If weights don't sum to ~1.0.
        """
        total_weight = dense_weight + sparse_weight
        if abs(total_weight - 1.0) > 0.01:
            msg = f"Weights must sum to ~1.0, got {total_weight}"
            raise ValueError(msg)

        # Normalize dense scores
        dense_scores: dict[str, float] = {}
        if dense_results:
            max_dense = max(s for _, s in dense_results)
            for doc_id, score in dense_results:
                dense_scores[doc_id] = score / max_dense if max_dense > 0 else 0

        # Normalize sparse scores
        sparse_scores: dict[str, float] = {}
        if sparse_results:
            max_sparse = max(s for _, s in sparse_results)
            for doc_id, score in sparse_results:
                sparse_scores[doc_id] = score / max_sparse if max_sparse > 0 else 0

        # Combine
        combined: dict[str, float] = {}
        all_docs = set(dense_scores.keys()) | set(sparse_scores.keys())

        for doc_id in all_docs:
            d_score = dense_scores.get(doc_id, 0)
            s_score = sparse_scores.get(doc_id, 0)
            combined[doc_id] = d_score * dense_weight + s_score * sparse_weight

        return sorted(combined.items(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def merge_search_results(
        results: list[list[dict[str, Any]]],
        method: str = "rrf",
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Merge multiple search result lists.

        Args:
            results: List of result lists from different searches.
            method: "rrf" for RRF, "weighted" for weighted fusion.
            **kwargs: Additional arguments for fusion method.

        Returns:
            Merged result list.

        Raises:
            ValueError: If method not supported.
        """
        if not results or not any(results):
            return []

        # Extract (doc_id, score) tuples
        scored_results = [
            [(r.get("id", str(i)), r.get("score", 1.0)) for i, r in enumerate(res)]
            for res in results
        ]

        if method == "rrf":
            if len(scored_results) < 2:
                return results[0]
            fused = ResultFuser.reciprocal_rank_fusion(
                scored_results[0],
                scored_results[1],
                k=kwargs.get("k", 60),
            )
        elif method == "weighted":
            if len(scored_results) < 2:
                return results[0]
            fused = ResultFuser.weighted_fusion(
                scored_results[0],
                scored_results[1],
                dense_weight=kwargs.get("dense_weight", 0.7),
                sparse_weight=kwargs.get("sparse_weight", 0.3),
            )
        else:
            msg = f"Unknown fusion method: {method}"
            raise ValueError(msg)

        # Reconstruct result dicts
        id_to_result = {}
        for result_list in results:
            for r in result_list:
                doc_id = r.get("id")
                if doc_id and doc_id not in id_to_result:
                    id_to_result[doc_id] = r

        merged = []
        for doc_id, score in fused:
            if doc_id in id_to_result:
                res = dict(id_to_result[doc_id])
                res["fused_score"] = score
                merged.append(res)

        return merged
