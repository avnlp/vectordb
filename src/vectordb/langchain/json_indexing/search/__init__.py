"""JSON search pipelines for LangChain.

This package provides search implementations for JSON-indexed documents
across different vector database backends. Each implementation supports
metadata filtering with JSON path-based queries and optional RAG-based
answer generation.

Available implementations:
    - MilvusJsonSearchPipeline: Zilliz Cloud / Milvus backend
    - QdrantJsonSearchPipeline: Qdrant backend
    - ChromaJsonSearchPipeline: Chroma backend
    - PineconeJsonSearchPipeline: Pinecone backend
    - WeaviateJsonSearchPipeline: Weaviate backend

Common Features:
    - JSON metadata filtering with nested path support
    - Configurable embedding models via HuggingFace
    - Optional RAG with LLM-based answer generation
    - Unified configuration format across backends

Example:
    >>> config = {
    ...     "embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
    ...     "milvus": {"uri": "http://localhost:19530"},
    ...     "collection": {"name": "json_docs"},
    ...     "rag": {"enabled": False},
    ... }
    >>> pipeline = MilvusJsonSearchPipeline(config)
    >>> results = pipeline.search("machine learning", top_k=5)
"""

from .chroma import ChromaJsonSearchPipeline
from .milvus import MilvusJsonSearchPipeline
from .pinecone import PineconeJsonSearchPipeline
from .qdrant import QdrantJsonSearchPipeline
from .weaviate import WeaviateJsonSearchPipeline


__all__ = [
    "ChromaJsonSearchPipeline",
    "MilvusJsonSearchPipeline",
    "PineconeJsonSearchPipeline",
    "QdrantJsonSearchPipeline",
    "WeaviateJsonSearchPipeline",
]
