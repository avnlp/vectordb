"""Hybrid indexing pipelines for vector databases.

This module exports hybrid indexing pipeline classes for each supported
vector database. All pipelines implement a consistent interface for indexing
documents with both dense semantic embeddings and sparse lexical embeddings.

Each pipeline class provides:
    - Initialization from configuration (dict or YAML path)
    - Document loading and preprocessing
    - Dense and sparse embedding generation
    - Collection/index creation and management
    - Document upsertion with hybrid embeddings

Indexing Process:
    1. Load documents from configured data source
    2. Generate dense embeddings (semantic meaning)
    3. Generate sparse embeddings (lexical terms)
    4. Create or recreate vector database collection/index
    5. Upsert documents with both embedding types

Available Pipelines:
    PineconeHybridIndexingPipeline: Pinecone native hybrid indexing
    WeaviateHybridIndexingPipeline: Weaviate BM25-ready indexing
    QdrantHybridIndexingPipeline: Qdrant sparse vector indexing
    MilvusHybridIndexingPipeline: Milvus sparse vector indexing
    ChromaHybridIndexingPipeline: Chroma (dense + sparse metadata)

Usage:
    >>> from vectordb.langchain.hybrid_indexing import PineconeHybridIndexingPipeline
    >>> pipeline = PineconeHybridIndexingPipeline("config.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents")
"""

from vectordb.langchain.hybrid_indexing.indexing.chroma import (
    ChromaHybridIndexingPipeline,
)
from vectordb.langchain.hybrid_indexing.indexing.milvus import (
    MilvusHybridIndexingPipeline,
)
from vectordb.langchain.hybrid_indexing.indexing.pinecone import (
    PineconeHybridIndexingPipeline,
)
from vectordb.langchain.hybrid_indexing.indexing.qdrant import (
    QdrantHybridIndexingPipeline,
)
from vectordb.langchain.hybrid_indexing.indexing.weaviate import (
    WeaviateHybridIndexingPipeline,
)


__all__ = [
    "PineconeHybridIndexingPipeline",
    "WeaviateHybridIndexingPipeline",
    "ChromaHybridIndexingPipeline",
    "MilvusHybridIndexingPipeline",
    "QdrantHybridIndexingPipeline",
]
