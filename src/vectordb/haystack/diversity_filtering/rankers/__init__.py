"""Clustering-based diversity ranker for Haystack.

This component clusters candidate documents based on their semantic embeddings
and selects the most relevant document from each cluster. This ensures diverse
results by picking representatives from different topic clusters.

Algorithm:
1. Embed all candidate documents using sentence-transformers
2. Cluster embeddings into k clusters (where k = top_k)
3. For each cluster, select the document most similar to the query
4. Return the selected documents sorted by relevance to query
"""

import logging
from typing import Any, List, Optional

from haystack import Document, component
from haystack.utils import ComponentDevice


logger = logging.getLogger(__name__)


@component
class ClusteringDiversityRanker:
    """Clusters documents and selects diverse representatives.

    This ranker uses K-Means clustering to group similar documents together,
    then selects the most relevant document from each cluster. This provides
    diversity by ensuring results come from different topic clusters.

    Attributes:
        model: Sentence transformer model name for embedding.
        top_k: Number of diverse documents to return.
        device: Device to run the model on.
        similarity: Similarity metric for relevance scoring.

    Example:
        >>> from haystack import Document
        >>> from vectordb.haystack.diversity_filtering.rankers import (
        ...     ClusteringDiversityRanker,
        ... )
        >>> ranker = ClusteringDiversityRanker(
        ...     model="sentence-transformers/all-MiniLM-L6-v2", top_k=5
        ... )
        >>> ranker.warm_up()
        >>> docs = [
        ...     Document(content="Python programming"),
        ...     Document(content="Java programming"),
        ... ]
        >>> result = ranker.run(query="programming languages", documents=docs)
    """

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 10,
        device: Optional[ComponentDevice] = None,
        similarity: str = "cosine",
    ):
        """Initialize clustering diversity ranker.

        Args:
            model: Sentence transformer model for embeddings.
            top_k: Number of diverse documents to return.
            device: Device to run model on (cuda, cpu, mps). None = auto-detect.
            similarity: Similarity metric ("cosine" or "dot_product").
        """
        self.model_name = model
        self.top_k = top_k
        self.device = device
        self.similarity = similarity
        self._embedding_model = None

    def warm_up(self) -> None:
        """Initialize the sentence-transformers model."""
        from sentence_transformers import SentenceTransformer

        device = None
        if self.device:
            device = self.device.to_huggingface()

        self._embedding_model = SentenceTransformer(self.model_name, device=device)
        logger.info(
            "Warmed up ClusteringDiversityRanker with model %s", self.model_name
        )

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> dict[str, List[Document]]:
        """Rank documents by clustering for diversity.

        Args:
            query: The search query.
            documents: List of documents to rank.
            top_k: Override default top_k.

        Returns:
            Dictionary with "documents" key containing diverse documents.
        """
        if not documents:
            return {"documents": []}

        k = top_k or self.top_k
        k = min(k, len(documents))

        if k == 0:
            return {"documents": []}

        import numpy as np
        from sklearn.cluster import KMeans

        doc_contents = [doc.content for doc in documents]

        query_embedding = self._embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.similarity == "cosine",
        )[0]

        doc_embeddings = self._embedding_model.encode(
            doc_contents,
            convert_to_numpy=True,
            normalize_embeddings=self.similarity == "cosine",
            show_progress_bar=False,
        )

        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(doc_embeddings)

        selected_indices = []
        for cluster_id in range(k):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                continue

            cluster_embeddings = doc_embeddings[cluster_indices]

            if self.similarity == "cosine":
                similarities = np.dot(cluster_embeddings, query_embedding)
            else:
                diffs = cluster_embeddings - query_embedding
                similarities = -np.sum(diffs**2, axis=1)

            best_in_cluster = cluster_indices[np.argmax(similarities)]
            selected_indices.append(best_in_cluster)

        selected_indices = sorted(
            selected_indices, key=lambda i: documents[i].score or 0, reverse=True
        )

        selected_docs = [documents[i] for i in selected_indices[:k]]

        logger.debug(
            "Selected %d diverse documents from %d clusters",
            len(selected_docs),
            len(set(cluster_labels[selected_indices])),
        )

        return {"documents": selected_docs}
