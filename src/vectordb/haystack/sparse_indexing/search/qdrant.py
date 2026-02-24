"""Qdrant sparse search pipeline for keyword/BM25-style retrieval."""

from pathlib import Path
from typing import Any

from haystack import Document
from haystack.dataclasses import SparseEmbedding

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory
from vectordb.utils.logging import LoggerFactory


logger = LoggerFactory(logger_name=__name__).get_logger()


def _to_qdrant_sparse(sparse: SparseEmbedding) -> Any:
    """Convert Haystack SparseEmbedding to Qdrant SparseVector format."""
    from qdrant_client.http.models import SparseVector

    return SparseVector(
        indices=list(sparse.indices),
        values=list(sparse.values),
    )


class QdrantSparseSearchPipeline:
    """Qdrant sparse search pipeline using SPLADE embeddings."""

    def __init__(self, config_or_path: dict[str, Any] | str | Path) -> None:
        """Initialize pipeline from configuration."""
        self.config = ConfigLoader.load(config_or_path)
        db_config = self.config["qdrant"]

        self.db = QdrantVectorDB(
            location=db_config.get("location"),
            collection_name=db_config.get("collection_name"),
            api_key=db_config.get("api_key"),
        )

        self.embedder = EmbedderFactory.create_sparse_text_embedder(self.config)
        self.top_k = self.config.get("query", {}).get("top_k", 10)

        logger.info("Initialized QdrantSparseSearchPipeline")

    def search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """Search using sparse vectors.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            Dict with 'query' and 'documents' keys.
        """
        top_k = top_k or self.top_k

        # 1. Embed query with sparse embedding
        sparse_embedding = self.embedder.run(query)["embedding"]

        if sparse_embedding is None:
            logger.warning("Could not generate sparse embedding for query")
            return {"query": query, "documents": []}

        # 2. Convert to Qdrant format
        sparse_vector = _to_qdrant_sparse(sparse_embedding)

        # 3. Query Qdrant with sparse vector
        results = self.db.search(
            query_vector={"sparse": sparse_vector},
            top_k=top_k,
        )

        # 4. Convert results to Haystack Documents
        documents = []
        for result in results:
            payload = result.payload or {}
            doc = Document(
                content=payload.get("content", ""),
                id=str(result.id),
                meta={
                    "score": result.score or 0.0,
                    "doc_id": payload.get("doc_id"),
                    **{
                        k: v
                        for k, v in payload.items()
                        if k not in ["content", "doc_id"]
                    },
                },
            )
            documents.append(doc)

        logger.info(f"Found {len(documents)} documents for query: {query[:50]}...")
        return {"query": query, "documents": documents}
