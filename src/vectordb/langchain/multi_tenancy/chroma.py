"""Chroma multi-tenancy implementation using collections.

This module implements the MultiTenancyPipeline interface for Chroma,
using Chroma's collection mechanism for tenant isolation.
"""

import logging
from typing import Any

from langchain_core.documents import Document

from vectordb.databases.chroma import ChromaVectorDB

from .base import MultiTenancyPipeline


logger = logging.getLogger(__name__)


class ChromaMultiTenancyPipeline(MultiTenancyPipeline):
    """Chroma multi-tenancy pipeline using collections for isolation.

    Each tenant's data is stored in a separate Chroma collection,
    ensuring complete isolation and preventing cross-tenant data leakage.
    """

    def __init__(
        self,
        path: str = "./chroma_data",
        collection_prefix: str = "tenant_",
    ) -> None:
        """Initialize Chroma multi-tenancy pipeline.

        Args:
            path: Path to Chroma data directory.
            collection_prefix: Prefix for tenant collection names.
        """
        self.db = ChromaVectorDB(path=path)
        self.path = path
        self.collection_prefix = collection_prefix

        logger.info(
            "Initialized Chroma multi-tenancy pipeline at path: %s",
            path,
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
        query_embedding: list[float] | None = None,
    ) -> list[Document]:
        """Search within a tenant's isolated collection only.

        Args:
            tenant_id: Unique identifier for the tenant.
            query: Search query text.
            top_k: Number of results to return.
            filters: Optional metadata filters.
            query_embedding: Optional pre-computed query embedding. If provided,
                used for similarity search instead of query text.

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

        # Get the tenant-specific collection and query it
        collection = self.db._get_collection(collection_name)

        # Query the tenant's collection using embedding if provided, otherwise use text
        if query_embedding is not None:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filters,
                include=["metadatas", "documents", "distances"],
            )
        else:
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filters,
                include=["metadatas", "documents", "distances"],
            )

        # Convert Chroma query results to LangChain Documents
        from vectordb.utils.chroma_document_converter import ChromaDocumentConverter

        documents = (
            ChromaDocumentConverter.convert_query_results_to_langchain_documents(
                results
            )
        )

        logger.info(
            "Retrieved %d documents for tenant %s",
            len(documents),
            tenant_id,
        )

        return documents

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
        """List all active tenants (collections) in Chroma.

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

            logger.info("Found %d tenants in Chroma", len(tenants))
            return tenants
        except Exception as e:
            logger.error("Failed to list tenants: %s", str(e))
            return []
