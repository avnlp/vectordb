"""Core types, dataclasses, and utilities for multi-tenancy pipelines.

Provides shared enums, dataclasses, exceptions, and utilities for multi-tenant
RAG pipelines across all vector databases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from haystack import Document


__all__ = [
    "TenantIsolationStrategy",
    "TenantStatus",
    "TenantIsolationConfig",
    "MultitenancyTimingMetrics",
    "TenantIndexResult",
    "TenantRetrievalResult",
    "TenantRAGResult",
    "TenantQueryResult",
    "TenantOperationResult",
    "MultitenancyError",
    "TenantNotFoundError",
    "TenantConnectionError",
    "TenantExistsError",
    "TenantOperationNotSupportedError",
]


class TenantIsolationStrategy(Enum):
    """Isolation strategy for multi-tenancy implementation.

    Each strategy corresponds to a database-specific approach:

    - PARTITION_KEY: Milvus partition key field (is_partition_key=True)
    - NATIVE_MULTITENANCY: Weaviate native multi-tenancy (per-tenant shards)
    - NAMESPACE: Pinecone namespaces (one per tenant)
    - TIERED: Qdrant tiered multitenancy (is_tenant=True payload index)
    - DATABASE_SCOPING: Chroma tenant + database scoping
    """

    PARTITION_KEY = "partition_key"
    NATIVE_MULTITENANCY = "native_multitenancy"
    NAMESPACE = "namespace"
    TIERED = "tiered"
    DATABASE_SCOPING = "database_scoping"


class TenantStatus(Enum):
    """Status of a tenant in the multi-tenant system.

    Used primarily for Weaviate tenant state management and generalizable
    to other databases.

    - ACTIVE: Tenant is active and accepting operations
    - INACTIVE: Tenant exists but is not active
    - OFFLOADED: Tenant data is offloaded to cold storage (Weaviate)
    - UNKNOWN: Tenant status cannot be determined
    """

    ACTIVE = "active"
    INACTIVE = "inactive"
    OFFLOADED = "offloaded"
    UNKNOWN = "unknown"


@dataclass
class TenantIsolationConfig:
    """Configuration for tenant isolation per database.

    Attributes:
        strategy: Isolation strategy to use for tenant separation.
        field_name: Name of the field used for tenant identification.
        auto_create_tenant: Whether to auto-create tenants on first write.
        partition_key_isolation: Enable per-tenant HNSW index (Milvus 2.6+).
        num_partitions: Number of partitions for partition key (Milvus).
    """

    strategy: Literal[
        "partition_key",
        "namespace",
        "native_multitenancy",
        "tiered",
        "database_scoping",
    ]
    field_name: str = "tenant_id"
    auto_create_tenant: bool = True
    partition_key_isolation: bool = False
    num_partitions: int = 64


@dataclass
class MultitenancyTimingMetrics:
    """Timing metrics for multi-tenant operations.

    Captures granular timing information for performance analysis
    and cross-tenant benchmarking.

    Attributes:
        tenant_resolution_ms: Time to resolve/validate tenant context.
        index_operation_ms: Time for indexing operation (0 if not indexing).
        retrieval_ms: Time for vector retrieval operation.
        total_ms: End-to-end operation time.
        tenant_id: Tenant identifier for this operation.
        num_documents: Number of documents processed.
    """

    tenant_resolution_ms: float
    index_operation_ms: float
    retrieval_ms: float
    total_ms: float
    tenant_id: str
    num_documents: int


@dataclass
class TenantIndexResult:
    """Result from indexing documents for a tenant.

    Attributes:
        tenant_id: Tenant identifier.
        documents_indexed: Number of documents successfully indexed.
        collection_name: Name of the collection/index used.
        timing: Timing metrics for the indexing operation.
        success: Whether the operation completed successfully.
        message: Human-readable status message.
    """

    tenant_id: str
    documents_indexed: int
    collection_name: str
    timing: MultitenancyTimingMetrics
    success: bool = True
    message: str = ""


@dataclass
class TenantRetrievalResult:
    """Result from retrieval within tenant scope.

    Attributes:
        tenant_id: Tenant identifier.
        query: The query string used for retrieval.
        documents: List of retrieved Haystack Documents.
        scores: Relevance scores for each document.
        timing: Timing metrics for the retrieval operation.
    """

    tenant_id: str
    query: str
    documents: list[Document]
    scores: list[float]
    timing: MultitenancyTimingMetrics


@dataclass
class TenantRAGResult:
    """Result from RAG pipeline within tenant scope.

    Attributes:
        tenant_id: Tenant identifier.
        query: The query string used for RAG.
        retrieved_documents: List of retrieved context documents.
        generated_response: LLM-generated response.
        timing: Timing metrics for the RAG operation.
        retrieval_scores: Relevance scores for retrieved documents.
    """

    tenant_id: str
    query: str
    retrieved_documents: list[Document]
    generated_response: str
    timing: MultitenancyTimingMetrics
    retrieval_scores: list[float] = field(default_factory=list)


@dataclass
class TenantQueryResult:
    """Single query result within tenant scope.

    Attributes:
        document: Retrieved Haystack Document.
        relevance_score: Vector similarity score.
        rank: Rank in results (1-indexed).
        tenant_id: Tenant the result belongs to.
    """

    document: Document
    relevance_score: float
    rank: int
    tenant_id: str


@dataclass
class TenantOperationResult:
    """Result from tenant CRUD operations.

    Attributes:
        success: Whether the operation succeeded.
        tenant_id: Tenant identifier the operation was performed on.
        operation: Operation type (create, delete, list, stats).
        message: Human-readable message.
        data: Optional additional data from the operation.
    """

    success: bool
    tenant_id: str
    operation: str
    message: str = ""
    data: Any = None


@dataclass
class TenantStats:
    """Statistics for a tenant.

    Attributes:
        tenant_id: Tenant identifier.
        document_count: Number of documents for this tenant.
        vector_count: Number of vectors for this tenant.
        status: Tenant status (active, inactive, offloaded, unknown).
        created_at: Timestamp when tenant was created.
        last_updated: Timestamp of last update.
        size_bytes: Size of tenant data in bytes.
    """

    tenant_id: str
    document_count: int
    vector_count: int = 0
    status: TenantStatus = TenantStatus.ACTIVE
    created_at: datetime | None = None
    last_updated: datetime | None = None
    size_bytes: int = 0


class MultitenancyError(Exception):
    """Base exception for multi-tenancy operations."""


class TenantNotFoundError(MultitenancyError):
    """Raised when tenant does not exist."""


class TenantExistsError(MultitenancyError):
    """Raised when tenant already exists (for explicit create)."""


class TenantOperationNotSupportedError(MultitenancyError):
    """Raised when operation not supported by database."""


class TenantConnectionError(MultitenancyError):
    """Raised when connection to database fails."""
