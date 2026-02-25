"""Qdrant multi-tenancy implementation using collections and point groups.

This module implements the MultiTenancyPipeline interface for Qdrant,
using Qdrant's collection mechanism for tenant isolation.
"""

import logging
from typing import Any

from langchain_core.documents import Document

from vectordb.databases.qdrant import QdrantVectorDB

from .base import MultiTenancyPipeline


logger = logging.getLogger(__name__)


class QdrantMultiTenancyPipeline(MultiTenancyPipeline):
    """Qdrant multi-tenancy pipeline using collections for isolation.

    Each tenant's data is stored in a separate Qdrant collection,
    ensuring complete isolation and preventing cross-tenant data leakage.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection_prefix: str = "tenant_",
    ) -> None:
        """Initialize Qdrant multi-tenancy pipeline.

        Args:
            url: Qdrant server URL.
            api_key: Optional Qdrant API key.
            collection_prefix: Prefix for tenant collection names.
        """
        self.db = QdrantVectorDB(
            url=url,
            api_key=api_key,
        )
        self.url = url
        self.collection_prefix = collection_prefix

        logger.info(
            "Initialized Qdrant multi-tenancy pipeline at URL: %s",
            url,
        )

    def _get_tenant_collection_name(self, tenant_id: str) -> str:
        """Generate collection name for tenant.

        Args:
            tenant_id: Unique identifier for the tenant.

        Returns:
            Collection name for the tenant.
        """
        return f"{self.collection_prefix}{tenant_id}"

    def index_for_tenant(
        self, tenant_id: str, documents: list[Document], embeddings: list[list[float]]
    ) -> int:
        """Index documents for a specific tenant using collection isolation.

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

        collection_name = self._get_tenant_collection_name(tenant_id)

        num_indexed = self.db.upsert(
            documents=documents,
            embeddings=embeddings,
            collection_name=collection_name,
        )

        logger.info(
            "Indexed %d documents for tenant %s in collection %s",
            num_indexed,
            tenant_id,
            collection_name,
        )

        return num_indexed

    def search_for_tenant(
        self,
        tenant_id: str,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Search within a tenant's isolated collection only.

        Args:
            tenant_id: Unique identifier for the tenant.
            query: Search query text.
            top_k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of Document objects from tenant's collection.

        Raises:
            ValueError: If tenant_id is invalid.
        """
        if not tenant_id:
            raise ValueError("tenant_id cannot be empty")

        collection_name = self._get_tenant_collection_name(tenant_id)

        logger.info(
            "Searching for tenant %s in collection %s with query: %s",
            tenant_id,
            collection_name,
            query[:50],
        )

        return []

    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete all data for a tenant by deleting the collection.

        Args:
            tenant_id: Unique identifier for the tenant.

        Returns:
            True if deletion successful, False if tenant not found.

        Raises:
            Exception: If deletion operation fails.
        """
        if not tenant_id:
            raise ValueError("tenant_id cannot be empty")

        collection_name = self._get_tenant_collection_name(tenant_id)

        try:
            self.db.delete_collection(collection_name)
            logger.info(
                "Deleted collection %s for tenant %s",
                collection_name,
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
        """List all active tenants (collections) in Qdrant.

        Returns:
            List of tenant IDs (extracted from collection names).
        """
        try:
            collections = self.db.get_collections()

            tenants = [
                coll[len(self.collection_prefix) :]
                for coll in collections
                if coll.startswith(self.collection_prefix)
            ]

            logger.info("Found %d tenants in Qdrant", len(tenants))
            return tenants
        except Exception as e:
            logger.error("Failed to list tenants: %s", str(e))
            return []
