"""Reranker factory for Haystack pipelines using cross-encoder and diversity rankers.

This module provides factory methods for creating Haystack reranker components.
Reranking improves retrieval precision by using more sophisticated scoring models
on a candidate set retrieved by faster bi-encoder search.

Reranker Types:
    Similarity Ranker (create):
        Uses cross-encoder models to score query-document pairs. Cross-encoders
        jointly encode query and document, enabling fine-grained relevance scoring
        at the cost of higher latency.

    Diversity Ranker (create_diversity_ranker):
        Uses MMR (Maximal Marginal Relevance) to balance relevance with diversity.
        Reduces redundancy in results by penalizing documents similar to already-
        selected ones.

Recommended Models:
    - BAAI/bge-reranker-v2-m3: High accuracy, multilingual, moderate speed
    - cross-encoder/ms-marco-MiniLM-L-6-v2: Fast, English-focused
    - BAAI/bge-reranker-base: Good balance of speed and accuracy

Configuration:
    Both factory methods read from configuration dictionaries with the following
    structure:
        reranker:
          model: "BAAI/bge-reranker-v2-m3"  # Required
          top_k: 5  # Optional, default 5
        mmr:
          model: "sentence-transformers/all-MiniLM-L6-v2"  # Required
          top_k: 10  # Optional, default 10

Usage:
    >>> from vectordb.haystack.utils import RerankerFactory
    >>> reranker = RerankerFactory.create(config)
    >>> diversity_ranker = RerankerFactory.create_diversity_ranker(config)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from haystack.components.rankers import SentenceTransformersSimilarityRanker


if TYPE_CHECKING:
    from haystack.components.rankers import SentenceTransformersDiversityRanker


class RerankerFactory:
    """Factory for creating Haystack reranker components.

    Provides class methods to instantiate and warm up reranker components
    from configuration dictionaries. All model names must be specified in
    full in the config; no alias resolution is performed.
    """

    @classmethod
    def create(cls, config: dict[str, Any]) -> SentenceTransformersSimilarityRanker:
        """Create a reranker from configuration.

        Uses SentenceTransformersSimilarityRanker (cross-encoder based).

        Args:
            config: Configuration with 'reranker' section containing
                   'model' (required) and 'top_k' (optional, default 5).

        Returns:
            Warmed-up SentenceTransformersSimilarityRanker.

        Raises:
            KeyError: If 'reranker.model' is not specified.
        """
        reranker_config = config.get("reranker", {})
        model = reranker_config["model"]
        top_k = reranker_config.get("top_k", 5)

        reranker = SentenceTransformersSimilarityRanker(model=model, top_k=top_k)
        reranker.warm_up()
        return reranker

    @classmethod
    def create_diversity_ranker(
        cls, config: dict[str, Any]
    ) -> "SentenceTransformersDiversityRanker":
        """Create a diversity ranker (MMR) from configuration.

        Uses SentenceTransformersDiversityRanker for Maximal Marginal Relevance.

        Args:
            config: Configuration with 'mmr' section containing 'model' (required),
                   'top_k' (optional), 'lambda_threshold' (optional).

        Returns:
            Warmed-up SentenceTransformersDiversityRanker.

        Raises:
            KeyError: If 'mmr.model' is not specified.
        """
        from haystack.components.rankers import SentenceTransformersDiversityRanker

        mmr_config = config.get("mmr", {})
        model = mmr_config["model"]
        top_k = mmr_config.get("top_k", 10)

        ranker = SentenceTransformersDiversityRanker(
            model=model,
            top_k=top_k,
            strategy="maximum_margin_relevance",
        )
        ranker.warm_up()
        return ranker
