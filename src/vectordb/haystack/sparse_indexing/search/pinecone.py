"""Pinecone sparse search pipeline for keyword/BM25-style retrieval."""

from pathlib import Path
from typing import Any

from haystack import Document
from haystack.dataclasses import SparseEmbedding

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.haystack.utils import ConfigLoader, EmbedderFactory
from vectordb.utils.logging import LoggerFactory


logger = LoggerFactory(logger_name=__name__).get_logger()


def _to_pinecone_sparse(sparse: SparseEmbedding) -> dict[str, list]:
    """Convert Haystack SparseEmbedding to Pinecone sparse_values format."""
    return {
        "indices": list(sparse.indices),
        "values": list(sparse.values),
    }


class PineconeSparseSearchPipeline:
    """Pinecone sparse-only search pipeline using SPLADE embeddings."""

    def __init__(self, config_or_path: dict[str, Any] | str | Path) -> None:
        """Initialize pipeline from configuration."""
        self.config = ConfigLoader.load(config_or_path)
        db_config = self.config["pinecone"]

        self.db = PineconeVectorDB(
            api_key=db_config.get("api_key"),
            index_name=db_config.get("index_name"),
        )

        self.embedder = EmbedderFactory.create_sparse_text_embedder(self.config)
        self.top_k = self.config.get("query", {}).get("top_k", 10)
        self.namespace = db_config.get("namespace", "")

        logger.info("Initialized PineconeSparseSearchPipeline")

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

        # 2. Convert to Pinecone format
        sparse_values = _to_pinecone_sparse(sparse_embedding)

        # 3. Query Pinecone with sparse vector only
        response = self.db.query(
            vector=[0.0],  # Dummy dense vector for sparse-only search
            sparse_vector=sparse_values,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True,
        )

        # 4. Convert results to Haystack Documents
        documents = []
        if response and response.get("matches"):
            for match in response["matches"]:
                metadata = match.get("metadata", {})
                content = metadata.pop("content", "")

                doc = Document(
                    content=content,
                    id=match.get("id"),
                    meta={
                        "score": match.get("score", 0.0),
                        **metadata,
                    },
                )
                documents.append(doc)

        logger.info(f"Found {len(documents)} documents for query: {query[:50]}...")
        return {"query": query, "documents": documents}
