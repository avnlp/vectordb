"""Tests for namespace and collection management in Haystack.

This package contains tests for namespace and collection management
across vector databases in Haystack. Namespaces provide logical isolation
for organizing documents within a single database instance.

Namespace concepts:
    - Logical isolation: Separate document sets within one index
    - Multi-tenancy: Tenant-specific document access
    - Environment separation: Dev/staging/prod isolation
    - Domain organization: Topic or category-based grouping

Namespace operations:
    - Creation and deletion
    - Document scoping to namespaces
    - Cross-namespace queries (where supported)
    - Namespace-level statistics

Database implementations tested:
    - Chroma: Collection-based namespacing
    - Milvus: Partition-based namespaces
    - Pinecone: Native namespace support
    - Qdrant: Collection-based isolation
    - Weaviate: Class-based separation

Each implementation tests:
    - Namespace CRUD operations
    - Document isolation verification
    - Query scoping to namespaces
    - Namespace statistics and metadata
"""
