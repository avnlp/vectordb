"""Contextual compression indexing pipelines for vector databases.

This module provides indexing pipelines that prepare vector stores for
contextual compression-based retrieval. These pipelines handle document
loading, embedding generation, and indexing across all supported databases.

The indexing process is similar to standard indexing but ensures documents
are stored with metadata that enables compression techniques during search:

Indexing Process:
    1. Load documents from configured data source
    2. Generate dense vector embeddings using configured embedder
    3. Store documents with full text preserved for compression
    4. Index documents in vector database with complete metadata

Available Indexing Pipelines:
    - ChromaContextualCompressionIndexingPipeline: Chroma with compression support
    - MilvusContextualCompressionIndexingPipeline: Milvus with compression support
    - PineconeContextualCompressionIndexingPipeline: Pinecone with compression support
    - QdrantContextualCompressionIndexingPipeline: Qdrant with compression support
    - WeaviateContextualCompressionIndexingPipeline: Weaviate with compression support

Note:
    Indexing for contextual compression is nearly identical to standard indexing.
    The compression logic is applied during the search phase, not indexing.
"""

from vectordb.langchain.contextual_compression.indexing.chroma import (
    ChromaContextualCompressionIndexingPipeline,
)
from vectordb.langchain.contextual_compression.indexing.milvus import (
    MilvusContextualCompressionIndexingPipeline,
)
from vectordb.langchain.contextual_compression.indexing.pinecone import (
    PineconeContextualCompressionIndexingPipeline,
)
from vectordb.langchain.contextual_compression.indexing.qdrant import (
    QdrantContextualCompressionIndexingPipeline,
)
from vectordb.langchain.contextual_compression.indexing.weaviate import (
    WeaviateContextualCompressionIndexingPipeline,
)


__all__ = [
    "ChromaContextualCompressionIndexingPipeline",
    "MilvusContextualCompressionIndexingPipeline",
    "PineconeContextualCompressionIndexingPipeline",
    "QdrantContextualCompressionIndexingPipeline",
    "WeaviateContextualCompressionIndexingPipeline",
]
