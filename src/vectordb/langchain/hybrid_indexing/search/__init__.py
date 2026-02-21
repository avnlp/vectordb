"""Hybrid search pipelines for vector databases.

This module exports hybrid search pipeline classes for each supported vector
database. All pipelines implement a consistent interface for executing hybrid
searches that combine dense semantic embeddings with sparse lexical embeddings.

Each pipeline class provides:
    - Initialization from configuration (dict or YAML path)
    - Dense and sparse query embedding generation
    - Hybrid search execution with configurable parameters
    - Optional RAG (Retrieval-Augmented Generation) for answer synthesis

Available Pipelines:
    PineconeHybridSearchPipeline: Pinecone with native hybrid search
    WeaviateHybridSearchPipeline: Weaviate with BM25 + vector fusion
    QdrantHybridSearchPipeline: Qdrant with sparse vector support
    MilvusHybridSearchPipeline: Milvus with sparse vector support
    ChromaHybridSearchPipeline: Chroma (dense search, sparse in metadata)

Usage:
    >>> from vectordb.langchain.hybrid_indexing.search import (
    ...     PineconeHybridSearchPipeline,
    ... )
    >>> pipeline = PineconeHybridSearchPipeline("config.yaml")
    >>> results = pipeline.search("machine learning tutorials", top_k=10)
    >>> for doc in results["documents"]:
    ...     print(doc.page_content)
"""

from vectordb.langchain.hybrid_indexing.search.chroma import (
    ChromaHybridSearchPipeline,
)
from vectordb.langchain.hybrid_indexing.search.milvus import (
    MilvusHybridSearchPipeline,
)
from vectordb.langchain.hybrid_indexing.search.pinecone import (
    PineconeHybridSearchPipeline,
)
from vectordb.langchain.hybrid_indexing.search.qdrant import (
    QdrantHybridSearchPipeline,
)
from vectordb.langchain.hybrid_indexing.search.weaviate import (
    WeaviateHybridSearchPipeline,
)


__all__ = [
    "PineconeHybridSearchPipeline",
    "WeaviateHybridSearchPipeline",
    "ChromaHybridSearchPipeline",
    "MilvusHybridSearchPipeline",
    "QdrantHybridSearchPipeline",
]
