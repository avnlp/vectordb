"""Weaviate sparse indexing pipeline (LangChain)."""

import logging
from typing import Any

from langchain_core.documents import Document

from vectordb.databases.weaviate import WeaviateVectorDB

from .base import BaseSparseIndexingPipeline


logger = logging.getLogger(__name__)


class WeaviateSparseIndexingPipeline(BaseSparseIndexingPipeline):
    """Weaviate indexing pipeline for sparse search (LangChain).

    Inherits common document loading and embedding logic from
    BaseSparseIndexingPipeline. Only implements Weaviate-specific
    database initialization and indexing.

    Note: Weaviate computes BM25 natively, so no sparse embeddings
    are generated at indexing time.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
        """
        super().__init__(config_or_path, db_config_key="weaviate")

        logger.info("Initialized Weaviate sparse indexing pipeline (LangChain)")

    def _initialize_db(self) -> None:
        """Initialize Weaviate database client."""
        weaviate_config = self.config["weaviate"]
        self.db = WeaviateVectorDB(
            url=weaviate_config["url"],
            api_key=weaviate_config.get("api_key"),
        )

        self.collection_name = weaviate_config.get("collection_name", "SparseSearch")

    def _index_documents(
        self,
        documents: list[Document],
        sparse_embeddings: list[dict[str, float]],
    ) -> int:
        """Index documents to Weaviate (BM25 computed natively).

        Weaviate computes BM25 natively from stored text at query time,
        so sparse embeddings are not used at indexing time.

        Args:
            documents: List of LangChain documents to index.
            sparse_embeddings: List of sparse embeddings (not used by Weaviate).

        Returns:
            Number of documents successfully indexed.
        """
        upsert_data = []
        for doc in documents:
            properties = doc.metadata or {}
            properties["text"] = doc.page_content
            upsert_data.append(properties)

        # The collection must be selected before upserting.
        self.db._select_collection(self.collection_name)
        self.db.upsert(data=upsert_data)

        logger.info(
            "Indexed %d documents with sparse embeddings to Weaviate", len(documents)
        )
        return len(documents)
