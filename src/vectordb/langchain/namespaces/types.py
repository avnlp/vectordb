"""Core types, dataclasses, and utilities for namespace/partition pipelines.

Provides shared enums, dataclasses, exceptions, and utilities for namespace
management across all vector databases.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from langchain_core.documents import Document


__all__ = [
    "IsolationStrategy",
    "TenantStatus",
    "NamespaceConfig",
    "NamespaceStats",
    "NamespaceTimingMetrics",
    "NamespaceQueryResult",
    "CrossNamespaceComparison",
    "CrossNamespaceResult",
    "NamespaceOperationResult",
    "NamespaceError",
    "NamespaceNotFoundError",
    "NamespaceExistsError",
    "NamespaceOperationNotSupportedError",
    "NamespaceConnectionError",
    "NamespaceNameGenerator",
    "QuerySampler",
]


class IsolationStrategy(Enum):
    """Isolation strategy for namespace implementation."""

    NAMESPACE = "namespace"
    TENANT = "tenant"
    PARTITION_KEY = "partition_key"
    PAYLOAD_FILTER = "payload_filter"
    COLLECTION = "collection"


class TenantStatus(Enum):
    """Status of a tenant/namespace (Weaviate-specific, generalizable)."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    OFFLOADED = "offloaded"
    UNKNOWN = "unknown"


@dataclass
class NamespaceConfig:
    """Configuration for a single namespace/partition/tenant.

    Attributes:
        name: Unique namespace identifier.
        description: Human-readable description.
        split: Dataset split this namespace corresponds to.
        metadata: Additional metadata key-value pairs.
    """

    name: str
    description: str = ""
    split: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NamespaceStats:
    """Statistics for a namespace.

    Attributes:
        namespace: Namespace identifier.
        document_count: Number of documents in namespace.
        vector_count: Number of vectors in namespace.
        status: Tenant status (active, inactive, offloaded, unknown).
        created_at: Timestamp when namespace was created.
        last_updated: Timestamp of last update.
        size_bytes: Size of namespace in bytes.
    """

    namespace: str
    document_count: int
    vector_count: int = 0
    status: TenantStatus = TenantStatus.ACTIVE
    created_at: datetime | None = None
    last_updated: datetime | None = None
    size_bytes: int = 0


@dataclass
class NamespaceTimingMetrics:
    """Timing metrics for namespace operations.

    Attributes:
        namespace_lookup_ms: Time to resolve namespace.
        vector_search_ms: Time for ANN search.
        total_ms: End-to-end operation time.
        documents_searched: Documents in namespace searched.
        documents_returned: Number of results returned.
    """

    namespace_lookup_ms: float
    vector_search_ms: float
    total_ms: float
    documents_searched: int
    documents_returned: int


@dataclass
class NamespaceQueryResult:
    """Result from querying a namespace.

    Attributes:
        document: Retrieved LangChain Document.
        relevance_score: Vector similarity score.
        rank: Rank in results (1-indexed).
        namespace: Namespace the result came from.
        timing: Optional timing metrics for the query.
    """

    document: Document
    relevance_score: float
    rank: int
    namespace: str
    timing: NamespaceTimingMetrics | None = None


@dataclass
class CrossNamespaceComparison:
    """Timing comparison across namespaces.

    Attributes:
        namespace: Namespace identifier.
        timing: Timing metrics for this namespace.
        result_count: Number of results from this namespace.
        top_score: Highest relevance score from this namespace.
    """

    namespace: str
    timing: NamespaceTimingMetrics
    result_count: int
    top_score: float


@dataclass
class CrossNamespaceResult:
    """Result from cross-namespace comparison query.

    Attributes:
        query: The query string.
        namespace_results: Dict mapping namespace to list of results.
        timing_comparison: List of timing comparisons per namespace.
        total_time_ms: Total time for all namespace queries.
    """

    query: str
    namespace_results: dict[str, list[NamespaceQueryResult]]
    timing_comparison: list[CrossNamespaceComparison]
    total_time_ms: float


@dataclass
class NamespaceOperationResult:
    """Result from namespace CRUD operations.

    Attributes:
        success: Whether the operation succeeded.
        namespace: Namespace the operation was performed on.
        operation: Operation type (create, delete, list, stats).
        message: Human-readable message.
        data: Optional additional data from the operation.
    """

    success: bool
    namespace: str
    operation: str
    message: str = ""
    data: Any = None


class NamespaceError(Exception):
    """Base exception for namespace operations."""


class NamespaceNotFoundError(NamespaceError):
    """Raised when namespace does not exist."""


class NamespaceExistsError(NamespaceError):
    """Raised when namespace already exists (for explicit create)."""


class NamespaceOperationNotSupportedError(NamespaceError):
    """Raised when operation not supported by database."""


class NamespaceConnectionError(NamespaceError):
    """Raised when connection to database fails."""


class NamespaceNameGenerator:
    """Generate consistent namespace names from config."""

    @staticmethod
    def from_split(dataset: str, split: str, prefix: str = "") -> str:
        """Generate namespace name from dataset and split.

        Args:
            dataset: Dataset name (e.g., "arc", "triviaqa").
            split: Split name (e.g., "train", "test").
            prefix: Optional prefix for the namespace name.

        Returns:
            Formatted namespace name.

        Examples:
            >>> NamespaceNameGenerator.from_split("arc", "train")
            'arc_train'
            >>> NamespaceNameGenerator.from_split("arc", "train", "qa")
            'qa_arc_train'
        """
        base = f"{dataset}_{split}"
        return f"{prefix}_{base}" if prefix else base

    @staticmethod
    def from_ticker(ticker: str, prefix: str = "earnings") -> str:
        """Generate namespace name from ticker symbol.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL").
            prefix: Prefix for the namespace name.

        Returns:
            Formatted namespace name.

        Examples:
            >>> NamespaceNameGenerator.from_ticker("AAPL")
            'earnings_AAPL'
        """
        return f"{prefix}_{ticker}"

    @staticmethod
    def from_config(config: dict[str, Any]) -> list[str]:
        """Extract namespace names from YAML config.

        Args:
            config: Configuration dictionary with namespaces section.

        Returns:
            List of namespace names from config.
        """
        namespaces_config = config.get("namespaces", {})
        definitions = namespaces_config.get("definitions", [])
        return [ns["name"] for ns in definitions if "name" in ns]


class QuerySampler:
    """Sample test queries from dataset."""

    @staticmethod
    def sample_from_documents(
        documents: list[Document],
        sample_size: int = 5,
        query_field: str = "question",
        seed: int | None = None,
    ) -> list[str]:
        """Sample query strings from document metadata.

        Args:
            documents: List of LangChain Documents to sample from.
            sample_size: Number of queries to sample.
            query_field: Metadata field containing query text.
            seed: Random seed for reproducibility.

        Returns:
            List of sampled query strings.
        """
        if seed is not None:
            random.seed(seed)

        queries = []
        for doc in documents:
            if query_field in doc.metadata:
                queries.append(str(doc.metadata[query_field]))
            elif doc.page_content:
                queries.append(doc.page_content[:100])

        if not queries:
            return []

        sample_size = min(sample_size, len(queries))
        return random.sample(queries, sample_size)
