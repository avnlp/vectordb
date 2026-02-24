"""Tests for multi-tenancy support in Haystack.

This package contains tests for multi-tenancy implementations
in Haystack. These tests verify the ability to isolate data and
queries between different tenants in shared vector database deployments.

Multi-tenancy concepts:
    - Tenant isolation: Complete data separation between tenants
    - Access control: Permission-based document access
    - Resource quotas: Limits on storage and queries per tenant
    - Tenant routing: Directing queries to appropriate tenant spaces

Isolation strategies:
    - Namespace isolation: Separate namespaces per tenant
    - Metadata filtering: Tenant ID in metadata with filters
    - Collection isolation: Separate collections per tenant
    - Partition isolation: Database-level partitioning

Database implementations tested:
    - Chroma: Collection-based tenant isolation
    - Milvus: Partition-based multi-tenancy
    - Pinecone: Namespace-based tenant separation
    - Qdrant: Collection-level tenant isolation
    - Weaviate: Class-based tenant separation

Each implementation tests:
    - Data isolation between tenants
    - Query routing accuracy
    - Cross-tenant access prevention
    - Performance with multiple tenants
"""
