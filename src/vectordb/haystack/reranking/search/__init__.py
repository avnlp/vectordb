"""Search pipelines for reranking.

This module provides search pipelines that implement two-stage retrieval:
1. Dense vector retrieval from the respective database
2. Cross-encoder reranking for precision improvement

Supported Databases:
    - Chroma: Local/embedded vector database
    - Milvus: Distributed vector database for large-scale search
    - Pinecone: Managed vector search service
    - Qdrant: High-performance vector database with hybrid search
    - Weaviate: Vector database with rich query capabilities

Each pipeline follows the same pattern:
    1. Embed the query using a bi-encoder
    2. Retrieve top_k*3 candidates using vector similarity
    3. Rerank candidates using a cross-encoder
    4. Return the top_k results

Example:
    >>> from vectordb.haystack.reranking.search import PineconeRerankingSearchPipeline
    >>> pipeline = PineconeRerankingSearchPipeline("config.yaml")
    >>> results = pipeline.search("transformer architecture", top_k=10)
"""

from vectordb.haystack.reranking.search.chroma import ChromaRerankingSearchPipeline
from vectordb.haystack.reranking.search.milvus import MilvusRerankingSearchPipeline
from vectordb.haystack.reranking.search.pinecone import PineconeRerankingSearchPipeline
from vectordb.haystack.reranking.search.qdrant import QdrantRerankingSearchPipeline
from vectordb.haystack.reranking.search.weaviate import WeaviateRerankingSearchPipeline


__all__ = [
    "ChromaRerankingSearchPipeline",
    "MilvusRerankingSearchPipeline",
    "PineconeRerankingSearchPipeline",
    "QdrantRerankingSearchPipeline",
    "WeaviateRerankingSearchPipeline",
]
