"""Namespace and partition management for Haystack vector database integrations.

This module provides namespace/partition pipelines for 5 vector databases:
- Pinecone: Native namespace parameter
- Weaviate: Native multi-tenancy
- Milvus: Partition key field
- Qdrant: Payload-based filtering
- Chroma: Collection-per-namespace

Each pipeline implements database-specific isolation strategies
using unified VectorDB wrappers.
"""

from vectordb.haystack.namespaces.chroma_namespaces import ChromaNamespacePipeline
from vectordb.haystack.namespaces.milvus_namespaces import MilvusNamespacePipeline
from vectordb.haystack.namespaces.pinecone_namespaces import PineconeNamespacePipeline
from vectordb.haystack.namespaces.qdrant_namespaces import QdrantNamespacePipeline
from vectordb.haystack.namespaces.types import (
    CrossNamespaceComparison,
    CrossNamespaceResult,
    IsolationStrategy,
    NamespaceConfig,
    NamespaceConnectionError,
    NamespaceError,
    NamespaceExistsError,
    NamespaceNameGenerator,
    NamespaceNotFoundError,
    NamespaceOperationNotSupportedError,
    NamespaceOperationResult,
    NamespaceQueryResult,
    NamespaceStats,
    NamespaceTimingMetrics,
    QuerySampler,
    TenantStatus,
)
from vectordb.haystack.namespaces.utils import Timer
from vectordb.haystack.namespaces.weaviate_namespaces import WeaviateNamespacePipeline


__all__ = [
    # Pipeline implementations
    "PineconeNamespacePipeline",
    "WeaviateNamespacePipeline",
    "MilvusNamespacePipeline",
    "QdrantNamespacePipeline",
    "ChromaNamespacePipeline",
    # Enums
    "IsolationStrategy",
    "TenantStatus",
    # Dataclasses
    "NamespaceConfig",
    "NamespaceStats",
    "NamespaceTimingMetrics",
    "NamespaceQueryResult",
    "CrossNamespaceComparison",
    "CrossNamespaceResult",
    "NamespaceOperationResult",
    # Exceptions
    "NamespaceError",
    "NamespaceNotFoundError",
    "NamespaceExistsError",
    "NamespaceConnectionError",
    "NamespaceOperationNotSupportedError",
    # Utilities
    "Timer",
    "NamespaceNameGenerator",
    "QuerySampler",
]
