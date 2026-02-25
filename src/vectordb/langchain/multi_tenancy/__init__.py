"""Multi-tenancy support for LangChain vector database pipelines.

This module provides multi-tenant isolation capabilities across all supported
vector databases, enabling secure multi-user RAG applications where each tenant's
data remains completely isolated.

Multi-Tenancy Concepts:
    Multi-tenancy allows a single vector database instance to serve multiple
    tenants (customers, organizations, users) while maintaining strict data
    isolation between them. This is critical for:

    - SaaS applications serving multiple customers
    - Enterprise platforms with department-level isolation
    - Applications with user-specific knowledge bases

Tenant Isolation Mechanisms:
    Different databases implement tenant isolation through different mechanisms:

    - Pinecone: Namespace-based isolation within a single index
      Each tenant gets a dedicated namespace for both indexing and search

    - Weaviate: Collection-based isolation
      Each tenant gets a dedicated collection (similar to a table)

    - Chroma: Collection-based isolation
      Each tenant gets a dedicated collection within the database

    - Milvus: Partition-based isolation
      Each tenant gets a dedicated partition within a collection

    - Qdrant: Collection-based isolation
      Each tenant gets a dedicated collection

Security Guarantees:
    - Complete data isolation between tenants
    - No cross-tenant query results possible
    - Separate metadata and embeddings per tenant
    - Query scoping enforced at database level

Key Classes:
    Base Class:
        - MultiTenancyPipeline: Abstract base for all multi-tenant pipelines

    Full Pipelines (Indexing + Search):
        - PineconeMultiTenancyPipeline
        - WeaviateMultiTenancyPipeline
        - ChromaMultiTenancyPipeline
        - MilvusMultiTenancyPipeline
        - QdrantMultiTenancyPipeline

    Indexing Pipelines:
        - PineconeMultiTenancyIndexingPipeline
        - WeaviateMultiTenancyIndexingPipeline
        - ChromaMultiTenancyIndexingPipeline
        - MilvusMultiTenancyIndexingPipeline
        - QdrantMultiTenancyIndexingPipeline

    Search Pipelines:
        - PineconeMultiTenancySearchPipeline
        - WeaviateMultiTenancySearchPipeline
        - ChromaMultiTenancySearchPipeline
        - MilvusMultiTenancySearchPipeline
        - QdrantMultiTenancySearchPipeline

Example:
    >>> from vectordb.langchain.multi_tenancy import PineconeMultiTenancyPipeline
    >>> # Initialize multi-tenant pipeline
    >>> pipeline = PineconeMultiTenancyPipeline("config.yaml")
    >>> # Index documents for a specific tenant
    >>> pipeline.index_documents(tenant_id="customer_123", documents=docs)
    >>> # Search within tenant's isolated data
    >>> results = pipeline.search(
    ...     tenant_id="customer_123",
    ...     query="machine learning",
    ...     top_k=5,
    ... )
    >>> # Results only contain customer_123's documents
    >>> for doc in results["documents"]:
    ...     print(doc.page_content)

Note:
    Always specify tenant_id when indexing or searching. The tenant_id is
    used to route operations to the correct isolated namespace/collection.
"""

from vectordb.langchain.multi_tenancy.base import MultiTenancyPipeline
from vectordb.langchain.multi_tenancy.chroma import ChromaMultiTenancyPipeline
from vectordb.langchain.multi_tenancy.indexing import (
    ChromaMultiTenancyIndexingPipeline,
    MilvusMultiTenancyIndexingPipeline,
    PineconeMultiTenancyIndexingPipeline,
    QdrantMultiTenancyIndexingPipeline,
    WeaviateMultiTenancyIndexingPipeline,
)
from vectordb.langchain.multi_tenancy.milvus import MilvusMultiTenancyPipeline
from vectordb.langchain.multi_tenancy.pinecone import PineconeMultiTenancyPipeline
from vectordb.langchain.multi_tenancy.qdrant import QdrantMultiTenancyPipeline
from vectordb.langchain.multi_tenancy.search import (
    ChromaMultiTenancySearchPipeline,
    MilvusMultiTenancySearchPipeline,
    PineconeMultiTenancySearchPipeline,
    QdrantMultiTenancySearchPipeline,
    WeaviateMultiTenancySearchPipeline,
)
from vectordb.langchain.multi_tenancy.weaviate import WeaviateMultiTenancyPipeline


__all__ = [
    "MultiTenancyPipeline",
    "PineconeMultiTenancyPipeline",
    "WeaviateMultiTenancyPipeline",
    "ChromaMultiTenancyPipeline",
    "MilvusMultiTenancyPipeline",
    "QdrantMultiTenancyPipeline",
    "PineconeMultiTenancyIndexingPipeline",
    "WeaviateMultiTenancyIndexingPipeline",
    "ChromaMultiTenancyIndexingPipeline",
    "MilvusMultiTenancyIndexingPipeline",
    "QdrantMultiTenancyIndexingPipeline",
    "PineconeMultiTenancySearchPipeline",
    "WeaviateMultiTenancySearchPipeline",
    "ChromaMultiTenancySearchPipeline",
    "MilvusMultiTenancySearchPipeline",
    "QdrantMultiTenancySearchPipeline",
]
