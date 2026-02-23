"""Evaluation metrics and detailed reporting for compression pipelines.

Evaluates both retrieval quality and compression effectiveness. Measures how well
the compression pipeline preserves relevant documents while reducing token usage.

Ranking Quality Metrics:
    - NDCG@K: Normalized Discounted Cumulative Gain for ranking quality
    - MRR: Mean Reciprocal Rank (position of first relevant document)
    - Recall@K: Fraction of relevant documents retrieved in top K

Compression Effectiveness Metrics:
    - Tokens Saved: Total reduction in token count after compression
    - Compression Ratio: Compressed tokens / Original tokens (lower is better)
    - Mean Relevance Score: Average relevance score of compressed documents

Evaluation Workflow:
    1. Run compression pipeline on evaluation queries
    2. Compare retrieved documents against ground truth (ideal_results)
    3. Calculate ranking metrics (NDCG, MRR, Recall@K)
    4. Calculate compression metrics (token savings, compression ratio)
    5. Generate detailed report with all metrics

Target Benchmarks:
    - NDCG@10 > 0.7: Good ranking quality
    - Compression Ratio < 0.5: Excellent compression (50% token reduction)
    - Recall@10 > 0.8: Most relevant documents retrieved
"""

from dataclasses import dataclass

from haystack import Document

from vectordb.haystack.contextual_compression.compression_utils import RankerResult


@dataclass
class CompressionEvaluationMetrics:
    """Metrics for compression pipeline evaluation.

    Attributes:
        ndcg: Normalized Discounted Cumulative Gain score.
        mrr: Mean Reciprocal Rank.
        recall_at_k: Recall@K values for different K values.
        tokens_saved: Total tokens saved by compression.
        compression_ratio: Ratio of compressed to original tokens.
        mean_score: Average relevance score of compressed documents.
    """

    ndcg: float
    mrr: float
    recall_at_k: dict[int, float]
    tokens_saved: int
    compression_ratio: float
    mean_score: float


class CompressionEvaluator:
    """Evaluate compression pipeline results using standard metrics."""

    @staticmethod
    def calculate_ndcg(
        ranked_results: list[RankerResult],
        ideal_results: list[Document],
        k: int = 10,
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain.

        Measures ranking quality by comparing to ideal ranking.

        Args:
            ranked_results: List of RankerResult objects from compression.
            ideal_results: List of ideal/relevant documents.
            k: Evaluate metric@K (default: 10).

        Returns:
            NDCG score between 0 and 1.
        """
        if not ranked_results or not ideal_results:
            return 0.0

        # NDCG (Normalized Discounted Cumulative Gain) Algorithm:
        # Measures ranking quality by weighting relevant docs at top positions higher
        # DCG = sum(relevance_i / log2(i + 2)) for i from 0 to k-1
        # NDCG = DCG / Ideal_DCG (normalized to 0-1 range)
        # Create ideal ranking (all relevant docs first)
        ideal_relevances = [1.0] * min(len(ideal_results), k)
        ideal_relevances += [0.0] * (k - len(ideal_relevances))

        # Create actual ranking based on retrieved documents
        ideal_set = {doc.id for doc in ideal_results if doc.id}
        actual_relevances = [
            1.0 if result.document.id in ideal_set else 0.0
            for result in ranked_results[:k]
        ]
        actual_relevances += [0.0] * (k - len(actual_relevances))

        # Calculate DCG
        def calculate_dcg(relevances: list[float]) -> float:
            return sum(rel / (i + 2) for i, rel in enumerate(relevances))

        actual_dcg = calculate_dcg(actual_relevances)
        ideal_dcg = calculate_dcg(ideal_relevances)

        if ideal_dcg == 0:
            return 0.0
        return actual_dcg / ideal_dcg

    @staticmethod
    def calculate_mrr(
        ranked_results: list[RankerResult],
        relevant_docs: list[Document],
    ) -> float:
        """Calculate Mean Reciprocal Rank.

        Measures rank position of first relevant document.

        Args:
            ranked_results: List of RankerResult objects from compression.
            relevant_docs: List of relevant documents to match against.

        Returns:
            MRR score (reciprocal of rank of first relevant result).
        """
        if not ranked_results or not relevant_docs:
            return 0.0

        # MRR (Mean Reciprocal Rank) Algorithm:
        # Finds position of first relevant document in ranked results
        # MRR = 1 / rank_position (or 0 if no relevant docs found)
        # Example: First relevant doc at position 3 -> MRR = 1/3 = 0.333
        relevant_set = {doc.id for doc in relevant_docs if doc.id}

        for i, result in enumerate(ranked_results):
            if result.document.id in relevant_set:
                return 1.0 / (i + 1)

        return 0.0

    @staticmethod
    def calculate_recall_at_k(
        ranked_results: list[RankerResult],
        relevant_docs: list[Document],
        k: int = 10,
    ) -> float:
        """Calculate Recall@K.

        Measures fraction of relevant docs found in top K results.

        Args:
            ranked_results: List of RankerResult objects from compression.
            relevant_docs: List of relevant documents.
            k: Calculate recall for top K results.

        Returns:
            Recall@K score between 0 and 1.
        """
        if not relevant_docs:
            return 0.0

        if not ranked_results:
            return 0.0

        # Recall@K Algorithm:
        # Measures coverage: what fraction of relevant docs were retrieved in top K
        # Recall@K = |relevant ∩ retrieved| / |relevant|
        # Example: 5 relevant docs, 4 found in top-10 -> Recall@10 = 4/5 = 0.8
        relevant_set = {doc.id for doc in relevant_docs if doc.id}
        retrieved_set = {
            result.document.id for result in ranked_results[:k] if result.document.id
        }

        if not relevant_set:
            return 0.0

        intersection = len(retrieved_set & relevant_set)
        return intersection / len(relevant_set)

    @staticmethod
    def evaluate_compression(
        ranked_results: list[RankerResult],
        ideal_results: list[Document],
        original_token_count: int,
        compressed_token_count: int,
        k: int = 10,
    ) -> CompressionEvaluationMetrics:
        """Comprehensive evaluation of compression pipeline results.

        Args:
            ranked_results: List of RankerResult objects from compression.
            ideal_results: List of ideal/relevant documents.
            original_token_count: Total tokens in original documents.
            compressed_token_count: Total tokens in compressed results.
            k: Evaluate metrics@K.

        Returns:
            CompressionEvaluationMetrics with all metrics.
        """
        ndcg = CompressionEvaluator.calculate_ndcg(ranked_results, ideal_results, k)
        mrr = CompressionEvaluator.calculate_mrr(ranked_results, ideal_results)

        recall_at_k = {}
        for k_val in [1, 5, 10, 20]:
            recall_at_k[k_val] = CompressionEvaluator.calculate_recall_at_k(
                ranked_results, ideal_results, k_val
            )

        tokens_saved = max(0, original_token_count - compressed_token_count)
        compression_ratio = (
            compressed_token_count / original_token_count
            if original_token_count > 0
            else 1.0
        )

        mean_score = (
            sum(r.score for r in ranked_results) / len(ranked_results)
            if ranked_results
            else 0.0
        )

        return CompressionEvaluationMetrics(
            ndcg=ndcg,
            mrr=mrr,
            recall_at_k=recall_at_k,
            tokens_saved=tokens_saved,
            compression_ratio=compression_ratio,
            mean_score=mean_score,
        )


def print_detailed_report(
    metrics: CompressionEvaluationMetrics,
    pipeline_name: str = "Compression Pipeline",
) -> None:
    """Print detailed evaluation report.

    Args:
        metrics: CompressionEvaluationMetrics with evaluation results.
        pipeline_name: Name of the pipeline being evaluated.
    """
    print("\n" + "=" * 60)
    print(f"{pipeline_name} - Detailed Evaluation Report")
    print("=" * 60)

    print("\nRanking Quality Metrics:")
    print(f"  NDCG@10:              {metrics.ndcg:.4f}")
    print(f"  MRR:                  {metrics.mrr:.4f}")
    print(f"  Mean Relevance Score: {metrics.mean_score:.4f}")

    print("\nRecall Metrics:")
    for k, recall in sorted(metrics.recall_at_k.items()):
        print(f"  Recall@{k:2d}:             {recall:.4f}")

    print("\nCompression Metrics:")
    print(f"  Compression Ratio:    {metrics.compression_ratio:.4f}")
    print("    (Lower is better. <0.5 = excellent, <0.7 = good)")
    print(f"  Tokens Saved:         {metrics.tokens_saved}")

    print("\n" + "=" * 60 + "\n")
