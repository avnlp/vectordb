"""Milvus multi-tenancy implementation with partition key tenant isolation.

This module provides Milvus-specific multi-tenancy pipelines for indexing and
searching documents with tenant-level data isolation. Milvus supports multi-tenancy
through partition key-based filtering where each tenant's data is tagged with a
tenant_id field and filtered using expression queries.

Milvus Multi-Tenancy Strategy:
    Partition key model where each tenant's data is stored in the same collection
    but tagged with a partition key. Collection names follow the base collection
    pattern with automatic tenant filtering via expression: tenant_id == "{tenant_id}"

Key Components:
    - MilvusMultitenancyIndexingPipeline: Tenant-scoped document indexing with
      partition keys
    - MilvusMultitenancySearchPipeline: Tenant-scoped retrieval with partition
      filtering

Isolation Model:
    Milvus partition keys provide logical isolation via expression filtering:
    - Data is tagged with tenant_id field during indexing
    - Queries automatically include tenant_id filter expression
    - Up to 1024 partitions per collection for tenant scalability
    - Automatic data routing based on partition key value

Usage:
    >>> from vectordb.haystack.multi_tenancy.milvus import (
    ...     MilvusMultitenancyIndexingPipeline,
    ...     MilvusMultitenancySearchPipeline,
    ... )
    >>> # Indexing
    >>> indexer = MilvusMultitenancyIndexingPipeline(
    ...     "config.yaml", tenant_context=TenantContext(tenant_id="tenant_abc")
    ... )
    >>> result = indexer.run()
    >>> # Search
    >>> search = MilvusMultitenancySearchPipeline(
    ...     "config.yaml", tenant_context=TenantContext(tenant_id="tenant_abc")
    ... )
    >>> results = search.query("machine learning", top_k=10)

Configuration (YAML):
    milvus:
      uri: "http://localhost:19530"
    collection:
      name: "documents"
    tenant:
      id: "default_tenant"
    embedding:
      model: "sentence-transformers/all-MiniLM-L6-v2"

Integration Points:
    - vectordb.haystack.multi_tenancy.common: Shared tenant utilities
    - vectordb.databases.milvus: MilvusVectorDB wrapper
    - vectordb.dataloaders: Dataset loading for tenant data
"""

from vectordb.haystack.multi_tenancy.milvus.indexing import (
    MilvusMultitenancyIndexingPipeline,
)
from vectordb.haystack.multi_tenancy.milvus.search import (
    MilvusMultitenancySearchPipeline,
)


__all__ = [
    "MilvusMultitenancyIndexingPipeline",
    "MilvusMultitenancySearchPipeline",
]
