"""Pinecone multi-tenancy implementation with namespace tenant isolation.

This module provides Pinecone-specific multi-tenancy pipelines for indexing and
searching documents with tenant-level data isolation. Pinecone supports multi-tenancy
through namespace separation where each tenant gets a dedicated namespace within
the same index.

Pinecone Multi-Tenancy Strategy:
    Namespace-per-tenant model where each tenant's data is stored in a separate
    Pinecone namespace. Namespace names follow the pattern: {tenant_id} within
    the shared index, providing natural tenant boundaries.

Key Components:
    - PineconeMultitenancyIndexingPipeline: Tenant-scoped document indexing with
      namespaces
    - PineconeMultitenancySearchPipeline: Tenant-scoped retrieval with namespace
      filtering

Isolation Model:
    Pinecone namespaces provide physical isolation at query time:
    - Each tenant has a dedicated namespace within the index
    - Queries automatically target the tenant's namespace
    - No cross-tenant data visibility by default
    - Efficient per-tenant data management and deletion

Usage:
    >>> from vectordb.haystack.multi_tenancy.pinecone import (
    ...     PineconeMultitenancyIndexingPipeline,
    ...     PineconeMultitenancySearchPipeline,
    ... )
    >>> # Indexing
    >>> indexer = PineconeMultitenancyIndexingPipeline(
    ...     "config.yaml", tenant_context=TenantContext(tenant_id="tenant_abc")
    ... )
    >>> result = indexer.run()
    >>> # Search
    >>> search = PineconeMultitenancySearchPipeline(
    ...     "config.yaml", tenant_context=TenantContext(tenant_id="tenant_abc")
    ... )
    >>> results = search.query("machine learning", top_k=10)

Configuration (YAML):
    pinecone:
      api_key: "${PINECONE_API_KEY}"
      environment: "us-west1-gcp"
    index:
      name: "documents"
    tenant:
      id: "default_tenant"
    embedding:
      model: "sentence-transformers/all-MiniLM-L6-v2"

Integration Points:
    - vectordb.haystack.multi_tenancy.common: Shared tenant utilities
    - vectordb.databases.pinecone: PineconeVectorDB wrapper
    - vectordb.dataloaders: Dataset loading for tenant data
"""

from vectordb.haystack.multi_tenancy.pinecone.indexing import (
    PineconeMultitenancyIndexingPipeline,
)
from vectordb.haystack.multi_tenancy.pinecone.search import (
    PineconeMultitenancySearchPipeline,
)


__all__ = [
    "PineconeMultitenancyIndexingPipeline",
    "PineconeMultitenancySearchPipeline",
]
