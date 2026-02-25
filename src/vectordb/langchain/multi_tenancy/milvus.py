"""Milvus multi-tenancy implementation using partitions.

This module implements the MultiTenancyPipeline interface for Milvus,
using Milvus's partition mechanism for tenant isolation.
"""

import logging
from typing import Any

from langchain_core.documents import Document

from vectordb.databases.milvus import MilvusVectorDB

from .base import MultiTenancyPipeline


logger = logging.getLogger(__name__)


class MilvusMultiTenancyPipeline(MultiTenancyPipeline):
    """Milvus multi-tenancy pipeline using partitions for isolation.

    Each tenant's data is stored in a separate Milvus partition,
    ensuring complete isolation and preventing cross-tenant data leakage.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "multi_tenancy",
        dimension: int = 384,
    ) -> None:
        """Initialize Milvus multi-tenancy pipeline.

        Args:
            host: Milvus host address.
            port: Milvus port number.
            collection_name: Name of the Milvus collection.
            dimension: Embedding dimension (default: 384).
        """
        self.db = MilvusVectorDB(
            host=host,
            port=port,
        )
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension

        logger.info(
            "Initialized Milvus multi-tenancy pipeline at %s:%d",
            host,
            port,
        )

    def index_for_tenant(
        self, tenant_id: str, documents: list[Document], embeddings: list[list[float]]
    ) -> int:
        """Index documents for a specific tenant using partition isolation.

        Args:
            tenant_id: Unique identifier for the tenant.
            documents: List of LangChain Document objects to index.
            embeddings: List of embedding vectors corresponding to documents.

        Returns:
            Number of documents successfully indexed.

        Raises:
            ValueError: If tenant_id is invalid or documents/embeddings mismatch.
        """
        if not tenant_id:
            raise ValueError("tenant_id cannot be empty")

        if len(documents) != len(embeddings):
            raise ValueError(
                f"Documents count ({len(documents)}) "
                f"does not match embeddings count ({len(embeddings)})"
            )

        if not documents:
            logger.warning("No documents to index for tenant: %s", tenant_id)
            return 0

        num_indexed = self.db.upsert(
            documents=documents,
            embeddings=embeddings,
            collection_name=self.collection_name,
            partition_name=tenant_id,
        )

        logger.info(
            "Indexed %d documents for tenant %s in partition %s",
            num_indexed,
            tenant_id,
            tenant_id,
        )

        return num_indexed

    def search_for_tenant(
        self,
        tenant_id: str,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[Document]:
        """Search within a tenant's isolated partition only.

        Args:
            tenant_id: Unique identifier for the tenant.
            query: Search query text (used for logging if query_embedding provided).
            top_k: Number of results to return.
            filters: Optional metadata filters.
            query_embedding: Pre-computed query embedding vector. Required for Milvus
                search. If not provided, a ValueError is raised.

        Returns:
            List of Document objects from tenant's partition.

        Raises:
            ValueError: If tenant_id is invalid or query_embedding not provided.
        """
        if not tenant_id:
            raise ValueError("tenant_id cannot be empty")

        if query_embedding is None:
            raise ValueError(
                "query_embedding is required for Milvus search. "
                "Generate embeddings before calling search_for_tenant."
            )

        logger.info(
            "Searching for tenant %s in partition %s with query: %s",
            tenant_id,
            tenant_id,
            query[:50],
        )

        # Search within tenant's partition using the scope parameter
        haystack_docs = self.db.search(
            query_embedding=query_embedding,
            top_k=top_k,
            collection_name=self.collection_name,
            filters=filters,
            scope=tenant_id,
        )

        # Convert Haystack Documents to LangChain Documents
        from vectordb.langchain.utils.document_converter import (
            HaystackToLangchainConverter,
        )

        documents = HaystackToLangchainConverter.convert(haystack_docs)

        logger.info(
            "Retrieved %d documents for tenant %s",
            len(documents),
            tenant_id,
        )

        return documents

    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete all data for a tenant by deleting the partition.

        Args:
            tenant_id: Unique identifier for the tenant.

        Returns:
            True if deletion successful, False if tenant not found.

        Raises:
            Exception: If deletion operation fails.
        """
        if not tenant_id:
            raise ValueError("tenant_id cannot be empty")

        try:
            self.db.delete_partition(
                collection_name=self.collection_name,
                partition_name=tenant_id,
            )
            logger.info(
                "Deleted partition %s for tenant %s",
                tenant_id,
                tenant_id,
            )
            return True
        except Exception as e:
            logger.error(
                "Failed to delete tenant %s: %s",
                tenant_id,
                str(e),
            )
            raise

    def list_tenants(self) -> list[str]:
        """List all active tenants (partitions) in Milvus collection.

        Returns:
            List of tenant IDs (partition names).
        """
        try:
            partitions = self.db.get_partitions(self.collection_name)
            logger.info(
                "Found %d tenants in collection %s",
                len(partitions),
                self.collection_name,
            )
            return partitions
        except Exception as e:
            logger.error("Failed to list tenants: %s", str(e))
            return []
