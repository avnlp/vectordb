"""Maximal Marginal Relevance (MMR) reranking for LangChain pipelines.

This module implements MMR for diversity-aware document reranking. MMR balances
relevance to the query with diversity among selected documents, reducing redundancy
in retrieval results.

MMR Algorithm:
    MMR iteratively selects documents by scoring each candidate with:

        MMR(d) = λ × sim(d, query) - (1-λ) × max_sim(d, selected)

    Where:
    - λ (lambda_param): Trade-off between relevance and diversity (0.0-1.0)
    - sim(d, query): Cosine similarity between document and query embeddings
    - max_sim(d, selected): Maximum similarity to any already-selected document

Lambda Parameter Guidelines:
    - λ = 1.0: Pure relevance ranking (no diversity penalty)
    - λ = 0.7-0.8: Emphasize relevance, mild diversity (recommended for precision)
    - λ = 0.5: Balanced relevance and diversity (good default)
    - λ = 0.3-0.4: Emphasize diversity (recommended for exploratory search)
    - λ = 0.0: Pure diversity (minimum redundancy, ignores relevance)

Use Cases:
    - Search results with many near-duplicate documents
    - Exploratory queries where coverage matters more than precision
    - Providing diverse context to LLMs to reduce repetition
    - Multi-document summarization preprocessing

Usage:
    >>> from vectordb.langchain.utils import MMRHelper
    >>> reranked = MMRHelper.mmr_rerank(
    ...     documents, embeddings, query_embedding, lambda_param=0.5, k=10
    ... )
"""

import numpy as np
from langchain_core.documents import Document


class MMRHelper:
    """Helper for Maximal Marginal Relevance reranking.

    Implements greedy MMR selection using cosine similarity for both
    relevance and diversity scoring. All methods are static to allow
    direct class method calls without instantiation.
    """

    @staticmethod
    def cosine_similarity(embedding1: list[float], embedding2: list[float]) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Cosine similarity score.
        """
        a = np.array(embedding1, dtype=np.float32)
        b = np.array(embedding2, dtype=np.float32)

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    @staticmethod
    def mmr_rerank(
        documents: list[Document],
        embeddings: list[list[float]],
        query_embedding: list[float],
        lambda_param: float = 0.5,
        k: int = 10,
    ) -> list[tuple[Document, float]]:
        """Apply MMR (Maximal Marginal Relevance) reranking.

        Args:
            documents: List of documents to rerank.
            embeddings: Embeddings for each document.
            query_embedding: Embedding of the query.
            lambda_param: Trade-off parameter between relevance and diversity
                (0-1). Higher values prioritize relevance, lower values
                prioritize diversity.
            k: Number of documents to return.

        Returns:
            List of (Document, MMR_score) tuples sorted by MMR score.
        """
        if not documents or k == 0:
            return []

        k = min(k, len(documents))

        # Calculate relevance scores (cosine similarity to query)
        relevance_scores = [
            MMRHelper.cosine_similarity(query_embedding, emb) for emb in embeddings
        ]

        selected_indices = []
        selected_scores = []
        remaining_indices = set(range(len(documents)))

        # Select first document (most relevant to query)
        first_idx = np.argmax(relevance_scores)
        selected_indices.append(first_idx)
        selected_scores.append(relevance_scores[first_idx])
        remaining_indices.remove(first_idx)

        # Iteratively select remaining documents
        while len(selected_indices) < k and remaining_indices:
            mmr_scores = {}

            for idx in remaining_indices:
                # Calculate relevance to query
                relevance = relevance_scores[idx]

                # Calculate redundancy: max similarity to any selected doc
                redundancy = 0.0
                if selected_indices:
                    redundancy = max(
                        MMRHelper.cosine_similarity(
                            embeddings[idx], embeddings[selected_idx]
                        )
                        for selected_idx in selected_indices
                    )

                # Calculate MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
                mmr_scores[idx] = mmr_score

            # Select document with highest MMR score
            best_idx = max(mmr_scores, key=mmr_scores.get)
            selected_indices.append(best_idx)
            selected_scores.append(mmr_scores[best_idx])
            remaining_indices.remove(best_idx)

        return [
            (documents[idx], score)
            for idx, score in zip(selected_indices, selected_scores)
        ]

    @staticmethod
    def mmr_rerank_simple(
        documents: list[Document],
        embeddings: list[list[float]],
        query_embedding: list[float],
        k: int = 10,
        lambda_param: float = 0.5,
    ) -> list[Document]:
        """Simple MMR reranking returning only documents.

        Args:
            documents: List of documents to rerank.
            embeddings: Embeddings for each document.
            query_embedding: Embedding of the query.
            k: Number of documents to return.
            lambda_param: Trade-off parameter between relevance and diversity
                (0-1). Higher values prioritize relevance, lower values
                prioritize diversity.

        Returns:
            List of reranked documents.
        """
        mmr_results = MMRHelper.mmr_rerank(
            documents,
            embeddings,
            query_embedding,
            lambda_param=lambda_param,
            k=k,
        )
        return [doc for doc, _ in mmr_results]
