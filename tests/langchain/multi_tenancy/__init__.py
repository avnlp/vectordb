"""Tests for multi-tenancy support in LangChain integrations.

This package contains tests for multi-tenant vector database implementations
in LangChain. Multi-tenancy enables secure data isolation between different
users or organizations sharing the same database infrastructure.

Multi-tenancy approaches:
    - Namespace isolation: Separate namespaces per tenant
    - Metadata filtering: Tenant ID in document metadata
    - Partition separation: Physical partitions per tenant
    - Collection separation: Separate collections per tenant

Database implementations tested:
    - Chroma: Collection-based tenant isolation
    - Milvus: Partition-based multi-tenancy with RBAC
    - Pinecone: Namespace-based tenant separation
    - Qdrant: Payload-based tenant filtering
    - Weaviate: Class-based tenant isolation

Each implementation tests:
    - Tenant data isolation
    - Cross-tenant access prevention
    - Tenant-aware retrieval
    - Resource cleanup per tenant
    - Performance with multiple tenants
"""
