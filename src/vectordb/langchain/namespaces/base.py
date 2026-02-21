"""Abstract base class for namespace pipelines in LangChain.

This module defines the NamespacePipeline abstract base class which
establishes the contract for namespace isolation across different vector databases.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from langchain_core.documents import Document

from .types import (
    CrossNamespaceResult,
    NamespaceOperationResult,
    NamespaceQueryResult,
    NamespaceStats,
)


logger = logging.getLogger(__name__)


class NamespacePipeline(ABC):
    """Abstract base class for namespace vector database pipelines.

    Defines the interface for namespace isolation through namespace/partition/
    collection mechanisms specific to each vector database.
    """

    @abstractmethod
    def create_namespace(self, namespace: str) -> NamespaceOperationResult:
        """Create a new namespace.

        Args:
            namespace: Unique namespace identifier.

        Returns:
            Operation result with success status.
        """
        raise NotImplementedError("Subclasses must implement create_namespace()")

    @abstractmethod
    def delete_namespace(self, namespace: str) -> NamespaceOperationResult:
        """Delete an existing namespace and all its data.

        Args:
            namespace: Namespace identifier to delete.

        Returns:
            Operation result with success status.
        """
        raise NotImplementedError("Subclasses must implement delete_namespace()")

    @abstractmethod
    def list_namespaces(self) -> list[str]:
        """List all namespaces.

        Returns:
            List of namespace identifiers.
        """
        raise NotImplementedError("Subclasses must implement list_namespaces()")

    @abstractmethod
    def namespace_exists(self, namespace: str) -> bool:
        """Check if a namespace exists.

        Args:
            namespace: Namespace identifier to check.

        Returns:
            True if namespace exists, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement namespace_exists()")

    @abstractmethod
    def get_namespace_stats(self, namespace: str) -> NamespaceStats:
        """Get statistics for a namespace.

        Args:
            namespace: Namespace identifier.

        Returns:
            Statistics for the namespace.
        """
        raise NotImplementedError("Subclasses must implement get_namespace_stats()")

    @abstractmethod
    def index_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
        namespace: str,
    ) -> NamespaceOperationResult:
        """Index documents with pre-computed embeddings into a namespace.

        Args:
            documents: List of LangChain Document objects to index.
            embeddings: List of embedding vectors corresponding to documents.
            namespace: Target namespace for indexing.

        Returns:
            Operation result with count of indexed documents.
        """
        raise NotImplementedError("Subclasses must implement index_documents()")

    @abstractmethod
    def query_namespace(
        self, query: str, namespace: str, top_k: int = 10
    ) -> list[NamespaceQueryResult]:
        """Query a specific namespace.

        Args:
            query: Search query text.
            namespace: Namespace to search within.
            top_k: Number of results to return.

        Returns:
            List of query results from the namespace.
        """
        raise NotImplementedError("Subclasses must implement query_namespace()")

    @abstractmethod
    def query_cross_namespace(
        self,
        query: str,
        namespaces: list[str] | None = None,
        top_k: int = 10,
    ) -> CrossNamespaceResult:
        """Query across multiple namespaces with timing comparison.

        Args:
            query: Search query text.
            namespaces: List of namespaces to query. If None, queries all.
            top_k: Number of results per namespace.

        Returns:
            Cross-namespace result with results and timing comparison.
        """
        raise NotImplementedError("Subclasses must implement query_cross_namespace()")
