"""Pinecone multi-tenancy implementation using namespaces.

This module implements the MultiTenancyPipeline interface for Pinecone,
using Pinecone's namespace mechanism for tenant isolation.
"""

import logging
from typing import Any

from langchain_core.documents import Document

from vectordb.databases.pinecone import PineconeVectorDB

from .base import MultiTenancyPipeline


logger = logging.getLogger(__name__)


class PineconeMultiTenancyPipeline(MultiTenancyPipeline):
    """Pinecone multi-tenancy pipeline using namespaces for isolation.

    Each tenant's data is stored in a separate Pinecone namespace,
    ensuring complete isolation and preventing cross-tenant data leakage.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        dimension: int = 384,
    ) -> None:
        """Initialize Pinecone multi-tenancy pipeline.

        Args:
            api_key: Pinecone API key.
            index_name: Name of the Pinecone index.
            dimension: Embedding dimension (default: 384).
        """
        self.db = PineconeVectorDB(
            api_key=api_key,
            index_name=index_name,
        )
        self.index_name = index_name
        self.dimension = dimension

        logger.info(
            "Initialized Pinecone multi-tenancy pipeline for index: %s",
            index_name,
        )

    def index_for_tenant(
        self, tenant_id: str, documents: list[Document], embeddings: list[list[float]]
    ) -> int:
        """Index documents for a specific tenant using namespace isolation.

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
            namespace=tenant_id,
        )

        logger.info(
            "Indexed %d documents for tenant %s in namespace %s",
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
        """Search within a tenant's isolated namespace only.

        Args:
            tenant_id: Unique identifier for the tenant.
            query: Search query text (used for logging if query_embedding provided).
            top_k: Number of results to return.
            filters: Optional metadata filters.
            query_embedding: Pre-computed query embedding vector. Required for Pinecone
                search. If not provided, a ValueError is raised.

        Returns:
            List of Document objects from tenant's namespace.

        Raises:
            ValueError: If tenant_id is invalid or query_embedding not provided.
        """
        if not tenant_id:
            raise ValueError("tenant_id cannot be empty")

        if query_embedding is None:
            raise ValueError(
                "query_embedding is required for Pinecone search. "
                "Generate embeddings before calling search_for_tenant."
            )

        logger.info(
            "Searching for tenant %s in namespace %s with query: %s",
            tenant_id,
            tenant_id,
            query[:50],
        )

        # Search within tenant's namespace
        haystack_docs = self.db.query(
            vector=query_embedding,
            top_k=top_k,
            filter=filters,
            namespace=tenant_id,
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
        """Delete all data for a tenant by deleting the namespace.

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
            self.db.delete_namespace(namespace=tenant_id)
            logger.info("Deleted all data for tenant %s", tenant_id)
            return True
        except Exception as e:
            logger.error("Failed to delete tenant %s: %s", tenant_id, str(e))
            raise

    def list_tenants(self) -> list[str]:
        """List all active tenants (namespaces) in the Pinecone index.

        Returns:
            List of tenant IDs (namespace names).
        """
        try:
            namespaces = self.db.get_index_stats().get("namespaces", {})
            tenant_list = list(namespaces.keys())
            logger.info("Found %d tenants in index", len(tenant_list))
            return tenant_list
        except Exception as e:
            logger.error("Failed to list tenants: %s", str(e))
            return []
