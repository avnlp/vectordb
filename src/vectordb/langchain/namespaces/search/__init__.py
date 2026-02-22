"""Namespace search pipelines for LangChain vector databases.

This module provides search pipelines that execute namespace-scoped queries
across all supported vector databases. Each pipeline ensures data isolation
by restricting searches to namespace-specific areas.

Supported Databases:
    - ChromaNamespaceSearchPipeline: Collection-scoped search
    - PineconeNamespaceSearchPipeline: Namespace-scoped search
    - MilvusNamespaceSearchPipeline: Partition-scoped search
    - QdrantNamespaceSearchPipeline: Payload-filtered search
    - WeaviateNamespaceSearchPipeline: Tenant-scoped search
"""

from vectordb.langchain.namespaces.search.chroma import (
    ChromaNamespaceSearchPipeline,
)
from vectordb.langchain.namespaces.search.milvus import (
    MilvusNamespaceSearchPipeline,
)
from vectordb.langchain.namespaces.search.pinecone import (
    PineconeNamespaceSearchPipeline,
)
from vectordb.langchain.namespaces.search.qdrant import (
    QdrantNamespaceSearchPipeline,
)
from vectordb.langchain.namespaces.search.weaviate import (
    WeaviateNamespaceSearchPipeline,
)


__all__ = [
    "ChromaNamespaceSearchPipeline",
    "MilvusNamespaceSearchPipeline",
    "PineconeNamespaceSearchPipeline",
    "QdrantNamespaceSearchPipeline",
    "WeaviateNamespaceSearchPipeline",
]
