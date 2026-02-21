"""Namespace support for LangChain vector database pipelines.

This module provides namespace isolation capabilities across all supported
vector databases, enabling data partitioning by dataset, split, or any
logical grouping.

Namespace Isolation Mechanisms:
    Different databases implement namespace isolation differently:

    - Pinecone: Native namespace isolation within a single index
      Each namespace gets a dedicated Pinecone namespace

    - Weaviate: Tenant-based isolation
      Each namespace maps to a Weaviate tenant

    - Chroma: Collection-based isolation
      Each namespace gets a dedicated collection

    - Milvus: Partition key field isolation
      Each namespace uses metadata field filtering

    - Qdrant: Payload-based filtering
      Each namespace uses payload filters on a shared collection

Key Classes:
    Base Class:
        - NamespacePipeline: Abstract base for all namespace pipelines

    Full Pipelines:
        - PineconeNamespacePipeline
        - WeaviateNamespacePipeline
        - ChromaNamespacePipeline
        - MilvusNamespacePipeline
        - QdrantNamespacePipeline

    Indexing Pipelines:
        - PineconeNamespaceIndexingPipeline
        - WeaviateNamespaceIndexingPipeline
        - ChromaNamespaceIndexingPipeline
        - MilvusNamespaceIndexingPipeline
        - QdrantNamespaceIndexingPipeline

    Search Pipelines:
        - PineconeNamespaceSearchPipeline
        - WeaviateNamespaceSearchPipeline
        - ChromaNamespaceSearchPipeline
        - MilvusNamespaceSearchPipeline
        - QdrantNamespaceSearchPipeline

Example:
    >>> from vectordb.langchain.namespaces import PineconeNamespacePipeline
    >>> pipeline = PineconeNamespacePipeline(
    ...     api_key="pc-api-...",
    ...     index_name="my-index",
    ... )
    >>> pipeline.create_namespace("arc_train")
    >>> namespaces = pipeline.list_namespaces()
"""

from vectordb.langchain.namespaces.base import NamespacePipeline
from vectordb.langchain.namespaces.chroma import ChromaNamespacePipeline
from vectordb.langchain.namespaces.indexing import (
    ChromaNamespaceIndexingPipeline,
    MilvusNamespaceIndexingPipeline,
    PineconeNamespaceIndexingPipeline,
    QdrantNamespaceIndexingPipeline,
    WeaviateNamespaceIndexingPipeline,
)
from vectordb.langchain.namespaces.milvus import MilvusNamespacePipeline
from vectordb.langchain.namespaces.pinecone import PineconeNamespacePipeline
from vectordb.langchain.namespaces.qdrant import QdrantNamespacePipeline
from vectordb.langchain.namespaces.search import (
    ChromaNamespaceSearchPipeline,
    MilvusNamespaceSearchPipeline,
    PineconeNamespaceSearchPipeline,
    QdrantNamespaceSearchPipeline,
    WeaviateNamespaceSearchPipeline,
)
from vectordb.langchain.namespaces.types import (
    CrossNamespaceComparison,
    CrossNamespaceResult,
    IsolationStrategy,
    NamespaceConfig,
    NamespaceError,
    NamespaceOperationResult,
    NamespaceQueryResult,
    NamespaceStats,
    TenantStatus,
)
from vectordb.langchain.namespaces.weaviate import WeaviateNamespacePipeline


__all__ = [
    "NamespacePipeline",
    "PineconeNamespacePipeline",
    "WeaviateNamespacePipeline",
    "ChromaNamespacePipeline",
    "MilvusNamespacePipeline",
    "QdrantNamespacePipeline",
    "PineconeNamespaceIndexingPipeline",
    "WeaviateNamespaceIndexingPipeline",
    "ChromaNamespaceIndexingPipeline",
    "MilvusNamespaceIndexingPipeline",
    "QdrantNamespaceIndexingPipeline",
    "PineconeNamespaceSearchPipeline",
    "WeaviateNamespaceSearchPipeline",
    "ChromaNamespaceSearchPipeline",
    "MilvusNamespaceSearchPipeline",
    "QdrantNamespaceSearchPipeline",
    "IsolationStrategy",
    "TenantStatus",
    "NamespaceConfig",
    "NamespaceStats",
    "NamespaceQueryResult",
    "CrossNamespaceComparison",
    "CrossNamespaceResult",
    "NamespaceOperationResult",
    "NamespaceError",
]
