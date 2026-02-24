"""Common utilities for multi-tenant vector database pipelines.

This module provides shared infrastructure for implementing tenant-aware vector
database operations. Multi-tenancy ensures data isolation between tenants using
either collection-level separation or metadata-based filtering.

Architecture Overview:
    Tenant Isolation Strategies:
    1. Collection-per-tenant: Separate collection/namespace per tenant
    2. Metadata filtering: Single collection with tenant_id in metadata
    3. Hybrid: Collection prefix with metadata validation

Key Components:
    - tenant_context: TenantContext dataclass for immutable tenant identification
    - config: Configuration loading with environment variable resolution
    - embeddings: Embedder factory with output dimension truncation
    - rag: Haystack RAG pipeline creation utilities
    - timing: Performance metrics collection for pipeline operations
    - types: Typed results for indexing, retrieval, and RAG operations

Tenant Context:
    TenantContext provides immutable tenant identification supporting:
    - Environment variable resolution (TENANT_ID, TENANT_NAME)
    - YAML configuration lookup (tenant.id, tenant.name)
    - Explicit parameter passing (highest priority)
    - Custom metadata for auditing and filtering

Usage:
    >>> from vectordb.haystack.multi_tenancy.common import TenantContext
    >>> # From environment
    >>> os.environ["TENANT_ID"] = "company_abc"
    >>> tenant = TenantContext.from_environment()
    >>> # From config
    >>> config = {"tenant": {"id": "org_123", "name": "Acme Corp"}}
    >>> tenant = TenantContext.from_config(config)
    >>> # Resolution (tries explicit, then env, then config)
    >>> tenant = TenantContext.resolve(tenant_context=explicit, config=config)

Configuration:
    The config module supports ${VAR} and ${VAR:-default} syntax for environment
    variable interpolation in YAML configuration files.

Integration Points:
    - vectordb.haystack.multi_tenancy.*: Database-specific implementations
    - vectordb.dataloaders: Dataset loading for tenant-specific data
    - haystack: Pipeline and Document abstractions
"""
