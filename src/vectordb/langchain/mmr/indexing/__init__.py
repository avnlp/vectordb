"""MMR indexing pipelines for vector databases.

This module provides Maximal Marginal Relevance (MMR) indexing pipelines
that prepare documents for diversity-aware retrieval. These pipelines extend
standard indexing with MMR-specific configurations and metadata handling.

MMR indexing pipelines store documents in a format optimized for the MMR
algorithm, which requires maintaining both embedding vectors and document
content to compute relevance-redundancy trade-offs during search time.

Database-Specific Indexing Classes:
    - ChromaMMRIndexingPipeline: Embedded vector database for local development
        and rapid prototyping with MMR support
    - MilvusMMRIndexingPipeline: Open-source vector database with partition
        namespaces and high-throughput MMR indexing
    - PineconeMMRIndexingPipeline: Cloud-native vector database with
        serverless MMR indexing and automatic scaling
    - QdrantMMRIndexingPipeline: High-performance vector search with
        payload filtering and efficient MMR candidate pre-fetching
    - WeaviateMMRIndexingPipeline: Open-source vector search with GraphQL
        interface and modular AI integrations for MMR workflows

Each pipeline provides:
    - Standard document embedding and storage
    - MMR-specific metadata preservation
    - Configurable chunking strategies for diversity optimization
    - Batch indexing for large document collections
"""

from vectordb.langchain.mmr.indexing.chroma import ChromaMMRIndexingPipeline
from vectordb.langchain.mmr.indexing.milvus import MilvusMMRIndexingPipeline
from vectordb.langchain.mmr.indexing.pinecone import PineconeMMRIndexingPipeline
from vectordb.langchain.mmr.indexing.qdrant import QdrantMMRIndexingPipeline
from vectordb.langchain.mmr.indexing.weaviate import WeaviateMMRIndexingPipeline


__all__ = [
    "ChromaMMRIndexingPipeline",
    "MilvusMMRIndexingPipeline",
    "PineconeMMRIndexingPipeline",
    "QdrantMMRIndexingPipeline",
    "WeaviateMMRIndexingPipeline",
]
