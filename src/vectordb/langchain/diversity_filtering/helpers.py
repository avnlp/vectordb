"""Diversity helpers for LangChain diversity filtering pipelines.

This module implements two diversification strategies:

1. Maximal Marginal Relevance (MMR):
   Query-aware reranking that balances relevance and diversity.
2. Clustering-based selection:
   K-Means clustering over candidate embeddings with representative sampling.
"""

import numpy as np
from langchain_core.documents import Document


class DiversityFilteringHelper:
    """Helper methods for diversity filtering search pipelines."""

    @staticmethod
    def cosine_similarity(embedding1: list[float], embedding2: list[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        a = np.array(embedding1, dtype=np.float32)
        b = np.array(embedding2, dtype=np.float32)

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    @staticmethod
    def mmr_diversify(
        documents: list[Document],
        embeddings: list[list[float]],
        query_embedding: list[float],
        max_documents: int = 5,
        lambda_param: float = 0.5,
    ) -> list[Document]:
        """Select diverse documents using Maximal Marginal Relevance (MMR).

        MMR iteratively selects documents by balancing query relevance against
        redundancy with already-selected documents:

            MMR(d) = lambda * sim(d, query) - (1 - lambda) * max_sim(d, selected)
        """
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")

        if not 0.0 <= lambda_param <= 1.0:
            raise ValueError("lambda_param must be between 0.0 and 1.0")

        if not documents or max_documents <= 0:
            return []

        query_dim = len(query_embedding)
        for embedding in embeddings:
            if len(embedding) != query_dim:
                raise ValueError(
                    "All document embeddings must match query embedding dimensions"
                )

        k = min(max_documents, len(documents))
        relevance_scores = [
            DiversityFilteringHelper.cosine_similarity(query_embedding, embedding)
            for embedding in embeddings
        ]

        selected_indices: list[int] = []
        remaining_indices = set(range(len(documents)))

        # First pick the document most relevant to the query.
        first_idx = int(np.argmax(relevance_scores))
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # Then greedily optimize the MMR objective.
        while len(selected_indices) < k and remaining_indices:
            best_idx = -1
            best_score = float("-inf")

            for idx in sorted(remaining_indices):
                relevance = relevance_scores[idx]
                redundancy = max(
                    DiversityFilteringHelper.cosine_similarity(
                        embeddings[idx], embeddings[selected_idx]
                    )
                    for selected_idx in selected_indices
                )
                mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return [documents[idx] for idx in selected_indices]

    @staticmethod
    def _greedy_fallback_diversify(
        documents: list[Document],
        embeddings: list[list[float]],
        max_documents: int,
    ) -> list[Document]:
        """Fallback greedy diversification for environments without sklearn."""
        if not documents or max_documents <= 0:
            return []

        if len(documents) <= max_documents:
            return documents

        selected_indices = [0]
        selected_embeddings = [embeddings[0]]
        remaining_indices = set(range(1, len(documents)))

        while len(selected_indices) < max_documents and remaining_indices:
            best_idx = -1
            min_max_similarity = float("inf")

            for idx in remaining_indices:
                max_similarity = max(
                    DiversityFilteringHelper.cosine_similarity(
                        embeddings[idx], selected_embedding
                    )
                    for selected_embedding in selected_embeddings
                )
                if max_similarity < min_max_similarity:
                    min_max_similarity = max_similarity
                    best_idx = idx

            selected_indices.append(best_idx)
            selected_embeddings.append(embeddings[best_idx])
            remaining_indices.remove(best_idx)

        return [documents[idx] for idx in selected_indices]

    @staticmethod
    def clustering_diversify(
        documents: list[Document],
        embeddings: list[list[float]],
        num_clusters: int = 3,
        samples_per_cluster: int = 2,
    ) -> list[Document]:
        """Select diverse documents using K-Means clustering."""
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")

        if not documents or num_clusters <= 0 or samples_per_cluster <= 0:
            return []

        n_docs = len(documents)
        actual_clusters = min(num_clusters, n_docs)

        try:
            from sklearn.cluster import KMeans
        except ImportError:
            return DiversityFilteringHelper._greedy_fallback_diversify(
                documents=documents,
                embeddings=embeddings,
                max_documents=actual_clusters * samples_per_cluster,
            )

        embeddings_array = np.array(embeddings, dtype=np.float32)
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)

        selected_indices: list[int] = []
        for cluster_id in range(actual_clusters):
            cluster_indices = [
                idx for idx, label in enumerate(cluster_labels) if label == cluster_id
            ]

            if not cluster_indices:
                continue

            cluster_center = kmeans.cluster_centers_[cluster_id]
            distances = [
                np.linalg.norm(embeddings_array[idx] - cluster_center)
                for idx in cluster_indices
            ]
            sorted_cluster = sorted(
                zip(cluster_indices, distances), key=lambda item: item[1]
            )

            for idx, _ in sorted_cluster[:samples_per_cluster]:
                if idx not in selected_indices:
                    selected_indices.append(idx)

        return [documents[idx] for idx in selected_indices]
