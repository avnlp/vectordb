"""Namespace indexing pipelines for LangChain vector databases.

This module provides document indexing pipelines that prepare vector stores
for namespace-based data isolation. Each pipeline handles document loading,
embedding generation, and namespace-isolated indexing.

Supported Databases:
    - ChromaNamespaceIndexingPipeline: Collection-per-namespace isolation
    - PineconeNamespaceIndexingPipeline: Native namespace isolation
    - MilvusNamespaceIndexingPipeline: Partition key field isolation
    - QdrantNamespaceIndexingPipeline: Payload-based filtering isolation
    - WeaviateNamespaceIndexingPipeline: Tenant-based isolation
"""

from vectordb.langchain.namespaces.indexing.chroma import (
    ChromaNamespaceIndexingPipeline,
)
from vectordb.langchain.namespaces.indexing.milvus import (
    MilvusNamespaceIndexingPipeline,
)
from vectordb.langchain.namespaces.indexing.pinecone import (
    PineconeNamespaceIndexingPipeline,
)
from vectordb.langchain.namespaces.indexing.qdrant import (
    QdrantNamespaceIndexingPipeline,
)
from vectordb.langchain.namespaces.indexing.weaviate import (
    WeaviateNamespaceIndexingPipeline,
)


__all__ = [
    "ChromaNamespaceIndexingPipeline",
    "MilvusNamespaceIndexingPipeline",
    "PineconeNamespaceIndexingPipeline",
    "QdrantNamespaceIndexingPipeline",
    "WeaviateNamespaceIndexingPipeline",
]
