"""Pinecone sparse indexing pipeline (LangChain)."""

import logging
from typing import Any

from langchain_core.documents import Document

from vectordb.databases.pinecone import PineconeVectorDB

from .base import BaseSparseIndexingPipeline


logger = logging.getLogger(__name__)


class PineconeSparseIndexingPipeline(BaseSparseIndexingPipeline):
    """Pinecone indexing pipeline for sparse search (LangChain).

    Inherits common document loading and embedding logic from
    BaseSparseIndexingPipeline. Only implements Pinecone-specific
    database initialization and indexing.

    Loads documents, generates sparse embeddings, creates index, and indexes.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Args:
            config_or_path: Config dict or path to YAML file.

        Raises:
            ValueError: If required config missing.
        """
        super().__init__(config_or_path, db_config_key="pinecone")

        logger.info("Initialized Pinecone sparse indexing pipeline (LangChain)")

    def _initialize_db(self) -> None:
        """Initialize Pinecone database client."""
        pinecone_config = self.config["pinecone"]
        self.db = PineconeVectorDB(
            api_key=pinecone_config["api_key"],
            index_name=pinecone_config.get("index_name"),
        )

        self.index_name = pinecone_config.get("index_name")
        self.namespace = pinecone_config.get("namespace", "")
        self.dimension = pinecone_config.get("dimension", 384)

    def _index_documents(
        self,
        documents: list[Document],
        sparse_embeddings: list[dict[str, float]],
    ) -> int:
        """Index documents with sparse embeddings to Pinecone.

        Args:
            documents: List of LangChain documents to index.
            sparse_embeddings: List of sparse embeddings (one per document).

        Returns:
            Number of documents successfully indexed.
        """
        recreate = self.config.get("pinecone", {}).get("recreate", False)
        self.db.create_index(
            index_name=self.index_name,
            dimension=self.dimension,
            metric=self.config.get("pinecone", {}).get("metric", "cosine"),
            recreate=recreate,
        )

        # Upsert with sparse embeddings only
        upsert_data = []
        for i, (doc, sparse_emb) in enumerate(zip(documents, sparse_embeddings)):
            upsert_data.append(
                {
                    "id": f"{self.index_name}_{i}",
                    "values": [0.0] * self.dimension,  # Placeholder dense vector
                    "sparse_values": sparse_emb,
                    "metadata": {
                        "text": doc.page_content,
                        **(doc.metadata or {}),
                    },
                }
            )

        num_indexed = self.db.upsert(
            data=upsert_data,
            namespace=self.namespace,
        )

        logger.info(
            "Indexed %d documents with sparse embeddings to Pinecone", num_indexed
        )
        return num_indexed
