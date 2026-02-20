"""Semantic diversification utilities for reducing redundancy in retrieval results.

This module provides algorithms for selecting diverse subsets of documents
from retrieval results. Diversification reduces redundancy when retrieved
documents contain similar or overlapping information.

Diversification Strategies:
    Greedy Selection (diversify):
        Iteratively selects documents that have the lowest maximum similarity
        to any already-selected document (MMR-style). Uses a similarity
        threshold to reject documents that are too similar to any selected
        document.

    Clustering-Based (clustering_based_diversity):
        Groups documents into semantic clusters using K-Means, then selects
        representative documents from each cluster. Ensures broad topic coverage.

When to Use Diversification:
    - Retrieved documents contain near-duplicates or paraphrases
    - Query matches multiple document versions (e.g., FAQ variants)
    - Building summarization inputs where redundancy hurts quality
    - Exploratory search where topic coverage matters

Comparison with MMR:
    - Greedy selection uses MMR-style max-similarity for diversity scoring
    - MMR balances relevance and diversity during retrieval
    - Diversification focuses purely on diversity among already-retrieved docs
    - Use MMR when you want relevance-aware diversity during retrieval
    - Use diversification for post-processing already-ranked results

Usage:
    >>> from vectordb.langchain.utils import DiversificationHelper
    >>> diverse_docs = DiversificationHelper.diversify(
    ...     documents, embeddings, max_documents=5, similarity_threshold=0.7
    ... )
"""

import numpy as np
from langchain_core.documents import Document


class DiversificationHelper:
    """Helper for semantic diversification of document sets.

    Provides static methods for selecting diverse document subsets using
    either greedy selection or clustering-based approaches. All methods
    work with LangChain Document objects and their embeddings.
    """

    @staticmethod
    def cosine_similarity(embedding1: list[float], embedding2: list[float]) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Cosine similarity score between 0 and 1.
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
    def diversify(
        documents: list[Document],
        embeddings: list[list[float]],
        max_documents: int = 5,
        similarity_threshold: float = 0.7,
    ) -> list[Document]:
        """Select diverse documents using MMR-style max-similarity scoring.

        Iteratively selects the document with the lowest maximum similarity
        to any already-selected document, rejecting candidates above the
        similarity threshold.

        Args:
            documents: List of documents to diversify.
            embeddings: Embeddings for each document.
            max_documents: Maximum number of documents to return.
            similarity_threshold: Remove documents more similar than this to
                selected docs.

        Returns:
            List of diverse documents.

        Raises:
            ValueError: If the number of embeddings does not match the number
                of documents.
        """
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")

        if not documents or max_documents <= 0:
            return []

        if len(documents) <= max_documents:
            return documents

        selected_indices = []
        selected_embeddings = []

        # Always include first document
        selected_indices.append(0)
        selected_embeddings.append(embeddings[0])

        # Iteratively add diverse documents
        remaining_indices = set(range(1, len(documents)))

        while len(selected_indices) < max_documents and remaining_indices:
            best_idx = None
            min_similarity = 2.0

            for idx in remaining_indices:
                # Calculate max similarity to any selected document (MMR-style)
                max_sim = max(
                    DiversificationHelper.cosine_similarity(
                        embeddings[idx], selected_emb
                    )
                    for selected_emb in selected_embeddings
                )

                # Select document with minimum max similarity
                if max_sim < min_similarity:
                    min_similarity = max_sim
                    best_idx = idx

            # Only add if below threshold
            if best_idx is not None:
                if min_similarity < similarity_threshold:
                    selected_indices.append(best_idx)
                    selected_embeddings.append(embeddings[best_idx])
                remaining_indices.remove(best_idx)
            else:
                break

        return [documents[idx] for idx in selected_indices]

    @staticmethod
    def clustering_based_diversity(
        documents: list[Document],
        embeddings: list[list[float]],
        num_clusters: int = 3,
        samples_per_cluster: int = 2,
    ) -> list[Document]:
        """Select diverse documents using clustering.

        Groups documents into clusters and selects representatives from each.

        Args:
            documents: List of documents to diversify.
            embeddings: Embeddings for each document.
            num_clusters: Number of clusters to create.
            samples_per_cluster: Number of samples to select from each cluster.

        Returns:
            List of diverse documents selected from clusters.

        Raises:
            ValueError: If the number of embeddings does not match the number
                of documents.
        """
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")

        if not documents or num_clusters <= 0 or samples_per_cluster <= 0:
            return []

        n_docs = len(documents)
        actual_clusters = min(num_clusters, n_docs)

        try:
            from sklearn.cluster import KMeans
        except ImportError:
            # Fallback to simple threshold-based diversity if sklearn not available
            return DiversificationHelper.diversify(
                documents,
                embeddings,
                max_documents=actual_clusters * samples_per_cluster,
            )

        # Perform clustering
        embeddings_array = np.array(embeddings, dtype=np.float32)
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)

        # Select samples from each cluster
        selected_indices = []
        for cluster_id in range(actual_clusters):
            cluster_indices = [
                i for i, label in enumerate(cluster_labels) if label == cluster_id
            ]

            if cluster_indices:
                # Calculate distance to cluster center
                cluster_center = kmeans.cluster_centers_[cluster_id]
                distances = [
                    np.linalg.norm(embeddings_array[idx] - cluster_center)
                    for idx in cluster_indices
                ]

                # Select closest samples
                sorted_indices = sorted(
                    zip(cluster_indices, distances), key=lambda x: x[1]
                )
                for idx, _ in sorted_indices[:samples_per_cluster]:
                    if idx not in selected_indices:
                        selected_indices.append(idx)

        return [documents[idx] for idx in selected_indices]
