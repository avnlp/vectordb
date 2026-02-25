"""Multi-tenancy search pipelines for LangChain vector databases.

This module provides search pipelines that execute tenant-scoped queries
across all supported vector databases. Each pipeline ensures data isolation
by restricting searches to tenant-specific namespaces or collections.

Multi-Tenancy Search Architecture:
    All multi-tenancy search pipelines follow a consistent pattern:

    1. Tenant Validation: Verify tenant ID and access permissions

    2. Query Embedding: Convert search query to dense vector using
       the configured embedding model

    3. Tenant-Scoped Search: Execute vector search within the tenant's
       isolated namespace, collection, or partition only

    4. Result Filtering: Apply optional metadata filters within the
       tenant's data boundary

Tenant Isolation Guarantees:
    Each database implementation ensures cross-tenant data leakage prevention:
    - Pinecone: Queries restricted to tenant-specific namespace
    - Weaviate: Searches scoped to tenant's dedicated collection
    - Chroma: Retrieval limited to tenant's isolated collection
    - Milvus: Queries executed within tenant's partition only
    - Qdrant: Searches confined to tenant's collection

Pipeline Consistency:
    All pipelines share identical interfaces:
    - __init__(config_or_path): Initialize from dict or YAML file path
    - search(query, tenant_id, top_k, filters) -> list[Document]: Tenant-scoped search

    This consistency enables easy database switching without code changes.

Supported Databases:
    - ChromaMultiTenancySearchPipeline: Local search with collection isolation
    - PineconeMultiTenancySearchPipeline: Cloud search with namespace isolation
    - MilvusMultiTenancySearchPipeline: Distributed search with partition isolation
    - QdrantMultiTenancySearchPipeline: High-performance with collection isolation
    - WeaviateMultiTenancySearchPipeline: GraphQL search with collection isolation

Security Considerations:
    - Always validate tenant_id before executing searches
    - Ensure tenant_id cannot be spoofed or bypassed
    - Log all tenant-scoped queries for audit purposes
    - Consider rate limiting per tenant to prevent resource exhaustion
    - Implement tenant authentication and authorization checks

Error Handling:
    - ValueError: Raised for invalid tenant_id format or missing tenant
    - PermissionError: Raised when tenant lacks access to requested data
    - RuntimeError: Raised for database connectivity or query execution failures

Example:
    >>> from vectordb.langchain.multi_tenancy.search import (
    ...     PineconeMultiTenancySearchPipeline,
    ... )
    >>> pipeline = PineconeMultiTenancySearchPipeline("configs/pinecone_mt.yaml")
    >>> results = pipeline.search(
    ...     query="machine learning",
    ...     tenant_id="tenant_123",
    ...     top_k=10,
    ... )
    >>> print(f"Found {len(results)} documents for tenant")

Note:
    Import specific pipelines directly from their respective modules rather
    than from this package-level __init__.py.
"""

from vectordb.langchain.multi_tenancy.search.chroma import (
    ChromaMultiTenancySearchPipeline,
)
from vectordb.langchain.multi_tenancy.search.milvus import (
    MilvusMultiTenancySearchPipeline,
)
from vectordb.langchain.multi_tenancy.search.pinecone import (
    PineconeMultiTenancySearchPipeline,
)
from vectordb.langchain.multi_tenancy.search.qdrant import (
    QdrantMultiTenancySearchPipeline,
)
from vectordb.langchain.multi_tenancy.search.weaviate import (
    WeaviateMultiTenancySearchPipeline,
)


__all__ = [
    "ChromaMultiTenancySearchPipeline",
    "MilvusMultiTenancySearchPipeline",
    "PineconeMultiTenancySearchPipeline",
    "QdrantMultiTenancySearchPipeline",
    "WeaviateMultiTenancySearchPipeline",
]
