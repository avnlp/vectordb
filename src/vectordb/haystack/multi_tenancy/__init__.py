"""Multi-tenancy pipelines for vector databases.

This package provides multi-tenancy support for various vector databases
with explicit per-database pipeline implementations.
"""

from vectordb.haystack.multi_tenancy.base import BaseMultitenancyPipeline
from vectordb.haystack.multi_tenancy.chroma.indexing import (
    ChromaMultitenancyIndexingPipeline,
)
from vectordb.haystack.multi_tenancy.chroma.search import (
    ChromaMultitenancySearchPipeline,
)
from vectordb.haystack.multi_tenancy.common.tenant_context import TenantContext
from vectordb.haystack.multi_tenancy.common.types import (
    TenantIndexResult,
    TenantIsolationStrategy,
    TenantQueryResult,
    TenantRAGResult,
    TenantRetrievalResult,
    TenantStats,
    TenantStatus,
)
from vectordb.haystack.multi_tenancy.milvus.indexing import (
    MilvusMultitenancyIndexingPipeline,
)
from vectordb.haystack.multi_tenancy.milvus.search import (
    MilvusMultitenancySearchPipeline,
)
from vectordb.haystack.multi_tenancy.pinecone.indexing import (
    PineconeMultitenancyIndexingPipeline,
)
from vectordb.haystack.multi_tenancy.pinecone.search import (
    PineconeMultitenancySearchPipeline,
)
from vectordb.haystack.multi_tenancy.qdrant.indexing import (
    QdrantMultitenancyIndexingPipeline,
)
from vectordb.haystack.multi_tenancy.qdrant.search import (
    QdrantMultitenancySearchPipeline,
)
from vectordb.haystack.multi_tenancy.weaviate.indexing import (
    WeaviateMultitenancyIndexingPipeline,
)
from vectordb.haystack.multi_tenancy.weaviate.search import (
    WeaviateMultitenancySearchPipeline,
)


__all__ = [
    # Base class
    "BaseMultitenancyPipeline",
    # Milvus pipelines
    "MilvusMultitenancyIndexingPipeline",
    "MilvusMultitenancySearchPipeline",
    # Weaviate pipelines
    "WeaviateMultitenancyIndexingPipeline",
    "WeaviateMultitenancySearchPipeline",
    # Pinecone pipelines
    "PineconeMultitenancyIndexingPipeline",
    "PineconeMultitenancySearchPipeline",
    # Qdrant pipelines
    "QdrantMultitenancyIndexingPipeline",
    "QdrantMultitenancySearchPipeline",
    # Chroma pipelines
    "ChromaMultitenancyIndexingPipeline",
    "ChromaMultitenancySearchPipeline",
    # Common utilities
    "TenantContext",
    "TenantIndexResult",
    "TenantRetrievalResult",
    "TenantRAGResult",
    "TenantQueryResult",
    "TenantIsolationStrategy",
    "TenantStats",
    "TenantStatus",
]
