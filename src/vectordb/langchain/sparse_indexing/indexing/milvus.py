"""Milvus sparse indexing pipeline (LangChain)."""

import logging
from typing import Any

from langchain_core.documents import Document

from vectordb.databases.milvus import MilvusVectorDB

from .base import BaseSparseIndexingPipeline


logger = logging.getLogger(__name__)


class MilvusSparseIndexingPipeline(BaseSparseIndexingPipeline):
    """Milvus indexing pipeline for sparse search (LangChain).

    Inherits common document loading and embedding logic from
    BaseSparseIndexingPipeline. Only implements Milvus-specific
    database initialization and indexing.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize indexing pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
        """
        super().__init__(config_or_path, db_config_key="milvus")

        logger.info("Initialized Milvus sparse indexing pipeline (LangChain)")

    def _initialize_db(self) -> None:
        """Initialize Milvus database client."""
        milvus_config = self.config["milvus"]
        self.db = MilvusVectorDB(
            host=milvus_config.get("host", "localhost"),
            port=milvus_config.get("port", 19530),
        )

        self.collection_name = milvus_config.get("collection_name", "sparse_search")
        self.dimension = milvus_config.get("dimension", 384)

    def _index_documents(
        self,
        documents: list[Document],
        sparse_embeddings: list[dict[str, float]],
    ) -> int:
        """Index documents with sparse embeddings to Milvus.

        Args:
            documents: List of LangChain documents to index.
            sparse_embeddings: List of sparse embeddings (one per document).

        Returns:
            Number of documents successfully indexed.
        """
        # Prepare data for Milvus with sparse embeddings in metadata
        # Milvus insert_documents expects Haystack-style documents
        # For LangChain documents, we store sparse embeddings in metadata
        upsert_data = []
        for _i, (doc, sparse_emb) in enumerate(zip(documents, sparse_embeddings)):
            upsert_data.append(
                {
                    "content": doc.page_content,
                    "meta": {
                        **doc.metadata,
                        "sparse_embedding": sparse_emb,
                    },
                }
            )

        # Convert to Haystack documents for Milvus
        from haystack.dataclasses import Document as HaystackDocument

        haystack_docs = [
            HaystackDocument(
                content=item["content"],
                meta=item["meta"],
            )
            for item in upsert_data
        ]

        self.db.insert_documents(
            documents=haystack_docs,
            collection_name=self.collection_name,
        )

        logger.info(
            "Indexed %d documents with sparse embeddings to Milvus", len(documents)
        )
        return len(documents)
