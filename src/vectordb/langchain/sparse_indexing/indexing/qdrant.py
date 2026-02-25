"""Qdrant sparse indexing pipeline (LangChain)."""

import logging
from typing import Any

from langchain_core.documents import Document

from vectordb.databases.qdrant import QdrantVectorDB

from .base import BaseSparseIndexingPipeline


logger = logging.getLogger(__name__)


class QdrantSparseIndexingPipeline(BaseSparseIndexingPipeline):
    """Qdrant indexing pipeline for sparse search (LangChain).

    Inherits common document loading and embedding logic from
    BaseSparseIndexingPipeline. Only implements Qdrant-specific
    database initialization and indexing.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
        """
        super().__init__(config_or_path, db_config_key="qdrant")

        logger.info("Initialized Qdrant sparse indexing pipeline (LangChain)")

    def _initialize_db(self) -> None:
        """Initialize Qdrant database client."""
        qdrant_config = self.config["qdrant"]
        self.db = QdrantVectorDB(
            url=qdrant_config.get("url", "http://localhost:6333"),
            api_key=qdrant_config.get("api_key"),
        )

        self.collection_name = qdrant_config.get("collection_name", "sparse_search")

    def _index_documents(
        self,
        documents: list[Document],
        sparse_embeddings: list[dict[str, float]],
    ) -> int:
        """Index documents with sparse embeddings to Qdrant.

        Args:
            documents: List of LangChain documents to index.
            sparse_embeddings: List of sparse embeddings (one per document).

        Returns:
            Number of documents successfully indexed.
        """
        # Convert to Haystack documents for Qdrant
        from haystack.dataclasses import Document as HaystackDocument

        haystack_docs = [
            HaystackDocument(
                content=doc.page_content,
                meta={
                    **doc.metadata,
                    "sparse_embedding": sparse_emb,
                },
            )
            for doc, sparse_emb in zip(documents, sparse_embeddings)
        ]

        self.db.index_documents(
            documents=haystack_docs,
        )

        logger.info(
            "Indexed %d documents with sparse embeddings to Qdrant", len(documents)
        )
        return len(documents)
