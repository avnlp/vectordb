"""Indexing pipelines for reranking.

This module provides indexing pipelines that prepare document collections
for two-stage retrieval with reranking.

Indexing Process:
    1. Load documents from configured data sources
    2. Generate dense embeddings using bi-encoder models
    3. Create collections/indexes in vector databases
    4. Upsert embedded documents with metadata

The bi-encoder embeddings enable fast approximate nearest neighbor search
during retrieval. These embeddings capture semantic meaning but are computed
independently for queries and documents (unlike cross-encoders).

Configuration:
    Each pipeline requires a YAML configuration specifying:
    - Database connection parameters (host, port, API keys)
    - Embedding model configuration (provider, model name, dimensions)
    - Data source configuration (dataset, limit, preprocessing)
    - Collection/index settings (name, metric, recreation policy)

Supported Databases:
    - Chroma: Local or server-based vector storage
    - Milvus: Distributed vector database
    - Pinecone: Managed cloud vector service
    - Qdrant: On-premise or cloud vector database
    - Weaviate: Vector database with GraphQL interface

Example:
    >>> from vectordb.haystack.reranking.indexing import ChromaRerankingIndexingPipeline
    >>> pipeline = ChromaRerankingIndexingPipeline("config.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents")
"""

from vectordb.haystack.reranking.indexing.chroma import ChromaRerankingIndexingPipeline
from vectordb.haystack.reranking.indexing.milvus import MilvusRerankingIndexingPipeline
from vectordb.haystack.reranking.indexing.pinecone import (
    PineconeRerankingIndexingPipeline,
)
from vectordb.haystack.reranking.indexing.qdrant import QdrantRerankingIndexingPipeline
from vectordb.haystack.reranking.indexing.weaviate import (
    WeaviateRerankingIndexingPipeline,
)


__all__ = [
    "ChromaRerankingIndexingPipeline",
    "MilvusRerankingIndexingPipeline",
    "PineconeRerankingIndexingPipeline",
    "QdrantRerankingIndexingPipeline",
    "WeaviateRerankingIndexingPipeline",
]
