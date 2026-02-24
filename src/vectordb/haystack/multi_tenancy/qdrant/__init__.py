"""Qdrant multi-tenancy implementation with payload-based tenant isolation.

This module provides Qdrant-specific multi-tenancy pipelines for indexing and
searching documents with tenant-level data isolation. Qdrant supports multi-tenancy
through payload-based filtering where each tenant's data is tagged with a tenant_id
payload field and filtered using payload conditions.

Qdrant Multi-Tenancy Strategy:
    Payload-based model where each tenant's data is stored in the same collection
    but tagged with a tenant_id payload. Collection names follow the base pattern
    with automatic tenant filtering via payload condition:
    must={"key": "tenant_id", "match": {"value": "{tenant_id}"}}

Key Components:
    - QdrantMultitenancyIndexingPipeline: Tenant-scoped document indexing with
      payload tags
    - QdrantMultitenancySearchPipeline: Tenant-scoped retrieval with payload
      filtering

Isolation Model:
    Qdrant payload filtering provides logical isolation via must conditions:
    - Data is tagged with tenant_id field in payload during indexing
    - Queries automatically include tenant_id payload filter condition
    - Efficient filtering with payload indices for performance
    - Supports both single-collection and multi-collection strategies

Usage:
    >>> from vectordb.haystack.multi_tenancy.qdrant import (
    ...     QdrantMultitenancyIndexingPipeline,
    ...     QdrantMultitenancySearchPipeline,
    ... )
    >>> # Indexing
    >>> indexer = QdrantMultitenancyIndexingPipeline(
    ...     "config.yaml", tenant_context=TenantContext(tenant_id="tenant_abc")
    ... )
    >>> result = indexer.run()
    >>> # Search
    >>> search = QdrantMultitenancySearchPipeline(
    ...     "config.yaml", tenant_context=TenantContext(tenant_id="tenant_abc")
    ... )
    >>> results = search.query("machine learning", top_k=10)

Configuration (YAML):
    qdrant:
      url: "http://localhost:6333"
    collection:
      name: "documents"
    tenant:
      id: "default_tenant"
    embedding:
      model: "sentence-transformers/all-MiniLM-L6-v2"

Integration Points:
    - vectordb.haystack.multi_tenancy.common: Shared tenant utilities
    - vectordb.databases.qdrant: QdrantVectorDB wrapper
    - vectordb.dataloaders: Dataset loading for tenant data
"""

from vectordb.haystack.multi_tenancy.qdrant.indexing import (
    QdrantMultitenancyIndexingPipeline,
)
from vectordb.haystack.multi_tenancy.qdrant.search import (
    QdrantMultitenancySearchPipeline,
)


__all__ = [
    "QdrantMultitenancyIndexingPipeline",
    "QdrantMultitenancySearchPipeline",
]
