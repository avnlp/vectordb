"""Milvus sparse search pipeline for keyword/BM25-style retrieval."""

from pathlib import Path
from typing import Any

from haystack import Document
from haystack.dataclasses import SparseEmbedding

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory
from vectordb.utils.logging import LoggerFactory


logger = LoggerFactory(logger_name=__name__).get_logger()


def _to_milvus_sparse(sparse: SparseEmbedding) -> dict[int, float]:
    """Convert Haystack SparseEmbedding to Milvus sparse format."""
    return dict(zip(sparse.indices, sparse.values))


class MilvusSparseSearchPipeline:
    """Milvus sparse search pipeline using SPLADE embeddings."""

    def __init__(self, config_or_path: dict[str, Any] | str | Path) -> None:
        """Initialize pipeline from configuration."""
        self.config = ConfigLoader.load(config_or_path)
        db_config = self.config["milvus"]

        self.db = MilvusVectorDB(
            connection_args=db_config.get("connection_args", {}),
            collection_name=db_config.get("collection_name"),
        )

        self.embedder = EmbedderFactory.create_sparse_text_embedder(self.config)
        self.top_k = self.config.get("query", {}).get("top_k", 10)

        logger.info("Initialized MilvusSparseSearchPipeline")

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

        # 2. Convert to Milvus format
        sparse_vector = _to_milvus_sparse(sparse_embedding)

        # 3. Query Milvus with sparse vector
        results = self.db.search(
            query_texts=[query],  # Pass original query for BM25 fallback
            query_sparse=[sparse_vector],
            top_k=top_k,
        )

        # 4. Convert results to Haystack Documents
        documents = []
        if results and len(results) > 0:
            for result in results[0]:  # First query result
                doc = Document(
                    content=result.page_content,
                    id=result.metadata.get("doc_id"),
                    meta={
                        "score": result.metadata.get("score", 0.0),
                        **result.metadata,
                    },
                )
                documents.append(doc)

        logger.info(f"Found {len(documents)} documents for query: {query[:50]}...")
        return {"query": query, "documents": documents}
