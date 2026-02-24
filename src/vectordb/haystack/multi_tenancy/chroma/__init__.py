"""Chroma multi-tenancy implementation with tenant-isolated operations.

This module provides Chroma-specific multi-tenancy pipelines for indexing and
searching documents with tenant-level data isolation. Chroma supports multi-tenancy
through collection-level separation where each tenant gets a dedicated collection.

Chroma Multi-Tenancy Strategy:
    Collection-per-tenant model where each tenant's data is stored in a separate
    Chroma collection. Collection names follow the pattern: {base_name}_{tenant_id}

Key Components:
    - ChromaMultitenancyIndexingPipeline: Tenant-scoped document indexing
    - ChromaMultitenancySearchPipeline: Tenant-scoped retrieval with optional RAG

Isolation Model:
    Chroma collections are fully isolated, providing:
    - No cross-tenant query contamination
    - Efficient per-tenant data management
    - Simple backup/restore per tenant

Usage:
    >>> from vectordb.haystack.multi_tenancy.chroma import (
    ...     ChromaMultitenancyIndexingPipeline,
    ...     ChromaMultitenancySearchPipeline,
    ... )
    >>> # Indexing
    >>> indexer = ChromaMultitenancyIndexingPipeline(
    ...     "config.yaml", tenant_context=TenantContext(tenant_id="tenant_abc")
    ... )
    >>> result = indexer.run()
    >>> # Search
    >>> search = ChromaMultitenancySearchPipeline(
    ...     "config.yaml", tenant_context=TenantContext(tenant_id="tenant_abc")
    ... )
    >>> results = search.query("machine learning", top_k=10)

Configuration (YAML):
    chroma:
      persist_dir: "./chroma_db"
    collection:
      name: "documents"
    tenant:
      id: "default_tenant"
    embedding:
      model: "sentence-transformers/all-MiniLM-L6-v2"

Integration Points:
    - vectordb.haystack.multi_tenancy.common: Shared tenant utilities
    - vectordb.databases.chroma: ChromaVectorDB wrapper
    - vectordb.dataloaders: Dataset loading for tenant data
"""
