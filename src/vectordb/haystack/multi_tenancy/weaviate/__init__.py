"""Weaviate multi-tenancy implementation with native tenant isolation.

This module provides Weaviate-specific multi-tenancy pipelines for indexing and
searching documents with tenant-level data isolation. Weaviate supports multi-tenancy
natively through tenant-aware collections where each tenant gets a dedicated shard
within a multi-tenant enabled class.

Weaviate Multi-Tenancy Strategy:
    Native multi-tenancy model where each tenant's data is stored in a dedicated
    shard within a multi-tenant enabled class. Classes are configured with
    multiTenancyConfig: {enabled: true} and operations specify the tenant.

Key Components:
    - WeaviateMultitenancyIndexingPipeline: Tenant-scoped document indexing with
      native MT
    - WeaviateMultitenancySearchPipeline: Tenant-scoped retrieval with tenant
      context

Isolation Model:
    Weaviate native multi-tenancy provides physical shard isolation:
    - Each tenant has a dedicated shard within the class
    - Class configured with multiTenancyConfig enabled
    - All operations (index, query, delete) include tenant context
    - Built-in tenant isolation at the storage level
    - Automatic shard management per tenant

Usage:
    >>> from vectordb.haystack.multi_tenancy.weaviate import (
    ...     WeaviateMultitenancyIndexingPipeline,
    ...     WeaviateMultitenancySearchPipeline,
    ... )
    >>> # Indexing
    >>> indexer = WeaviateMultitenancyIndexingPipeline(
    ...     "config.yaml", tenant_context=TenantContext(tenant_id="tenant_abc")
    ... )
    >>> result = indexer.run()
    >>> # Search
    >>> search = WeaviateMultitenancySearchPipeline(
    ...     "config.yaml", tenant_context=TenantContext(tenant_id="tenant_abc")
    ... )
    >>> results = search.query("machine learning", top_k=10)

Configuration (YAML):
    weaviate:
      url: "http://localhost:8080"
    class:
      name: "Documents"
      multi_tenancy: true
    tenant:
      id: "default_tenant"
    embedding:
      model: "sentence-transformers/all-MiniLM-L6-v2"

Integration Points:
    - vectordb.haystack.multi_tenancy.common: Shared tenant utilities
    - vectordb.databases.weaviate: WeaviateVectorDB wrapper
    - vectordb.dataloaders: Dataset loading for tenant data
"""

from vectordb.haystack.multi_tenancy.weaviate.indexing import (
    WeaviateMultitenancyIndexingPipeline,
)
from vectordb.haystack.multi_tenancy.weaviate.search import (
    WeaviateMultitenancySearchPipeline,
)


__all__ = [
    "WeaviateMultitenancyIndexingPipeline",
    "WeaviateMultitenancySearchPipeline",
]
