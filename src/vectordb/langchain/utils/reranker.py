"""Cross-encoder reranking utilities for LangChain retrieval pipelines.

This module provides helper functions for reranking retrieved documents using
cross-encoder models. Cross-encoders compute relevance scores by jointly encoding
the query and document, enabling more accurate relevance judgments than bi-encoder
similarity alone.

Cross-Encoder vs Bi-Encoder:
    Bi-encoders (used in vector search) embed query and documents independently,
    enabling fast approximate nearest neighbor search. Cross-encoders compute
    attention across query-document pairs, capturing fine-grained interactions
    but at higher computational cost. Typical pipelines retrieve candidates with
    bi-encoders, then rerank top-k with cross-encoders.

Supported Models:
    Uses HuggingFace cross-encoder models via langchain-community. Recommended:
    - cross-encoder/ms-marco-MiniLM-L-6-v2: Fast, good accuracy (default)
    - cross-encoder/ms-marco-MiniLM-L-12-v2: More accurate, slower
    - BAAI/bge-reranker-v2-m3: Multilingual, high accuracy

Usage:
    >>> from vectordb.langchain.utils import RerankerHelper
    >>> reranker = RerankerHelper.create_reranker({"reranker": {"model": "..."}})
    >>> reranked = RerankerHelper.rerank(reranker, query, documents, top_k=5)
"""

from typing import Any

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document


class RerankerHelper:
    """Helper for reranking documents using cross-encoder models.

    Provides class methods for creating rerankers from configuration and
    applying reranking to document lists. Supports both score-preserving
    and score-discarding variants.
    """

    @classmethod
    def create_reranker(cls, config: dict[str, Any]) -> HuggingFaceCrossEncoder:
        """Create cross-encoder reranker from config.

        Args:
            config: Configuration dictionary with reranker section.

        Returns:
            HuggingFaceCrossEncoder instance.
        """
        reranker_config = config.get("reranker", {})
        model_name = reranker_config.get(
            "model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        return HuggingFaceCrossEncoder(model_name=model_name)

    @classmethod
    def rerank(
        cls,
        reranker: HuggingFaceCrossEncoder,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        """Rerank documents using cross-encoder.

        Args:
            reranker: HuggingFaceCrossEncoder instance.
            query: Query text.
            documents: Documents to rerank.
            top_k: Number of top documents to return.

        Returns:
            Reranked documents.
        """
        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]
        scores = reranker.rank(pairs)

        sorted_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        if top_k is not None:
            sorted_docs = sorted_docs[:top_k]

        return [doc for doc, _ in sorted_docs]

    @classmethod
    def rerank_with_scores(
        cls,
        reranker: HuggingFaceCrossEncoder,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """Rerank documents and return with scores.

        Args:
            reranker: HuggingFaceCrossEncoder instance.
            query: Query text.
            documents: Documents to rerank.
            top_k: Number of top documents to return.

        Returns:
            List of (Document, score) tuples.
        """
        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]
        scores = reranker.rank(pairs)

        sorted_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        if top_k is not None:
            sorted_docs = sorted_docs[:top_k]

        return sorted_docs
