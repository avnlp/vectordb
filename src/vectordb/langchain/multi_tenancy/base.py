"""Abstract base class for multi-tenancy support in LangChain pipelines.

This module defines the MultiTenancyPipeline abstract base class which
establishes the contract for tenant isolation across different vector databases.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document


logger = logging.getLogger(__name__)


class MultiTenancyPipeline(ABC):
    """Abstract base class for multi-tenant vector database pipelines.

    Defines the interface for tenant isolation through namespace/partition/
    collection mechanisms specific to each vector database.
    """

    @abstractmethod
    def index_for_tenant(
        self, tenant_id: str, documents: list[Document], embeddings: list[list[float]]
    ) -> int:
        """Index documents for a specific tenant in isolation.

        Ensures cross-tenant data leakage is prevented by isolating data
        in tenant-specific namespaces/partitions/collections.

        Args:
            tenant_id: Unique identifier for the tenant.
            documents: List of LangChain Document objects to index.
            embeddings: List of embedding vectors corresponding to documents.

        Returns:
            Number of documents successfully indexed.

        Raises:
            ValueError: If tenant_id is invalid or documents/embeddings mismatch.
        """
        raise NotImplementedError(
            "Subclasses must implement index_for_tenant() for database-specific tenant indexing"
        )

    @abstractmethod
    def search_for_tenant(
        self,
        tenant_id: str,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Search within a specific tenant's data only.

        Ensures search results are scoped to the tenant's isolated namespace/
        partition/collection, preventing cross-tenant data leakage.

        Args:
            tenant_id: Unique identifier for the tenant.
            query: Search query text.
            top_k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of Document objects from tenant's data.

        Raises:
            ValueError: If tenant_id is invalid or not found.
        """
        raise NotImplementedError(
            "Subclasses must implement search_for_tenant() for database-specific tenant search"
        )

    @abstractmethod
    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete all data for a specific tenant.

        Removes the tenant's entire isolated namespace/partition/collection,
        ensuring complete data removal.

        Args:
            tenant_id: Unique identifier for the tenant.

        Returns:
            True if deletion successful, False if tenant not found.

        Raises:
            Exception: If deletion operation fails.
        """
        raise NotImplementedError(
            "Subclasses must implement delete_tenant() for database-specific tenant deletion"
        )

    @abstractmethod
    def list_tenants(self) -> list[str]:
        """List all active tenants in the system.

        Returns:
            List of tenant IDs currently present in the database.
        """
        raise NotImplementedError(
            "Subclasses must implement list_tenants() for database-specific tenant listing"
        )
