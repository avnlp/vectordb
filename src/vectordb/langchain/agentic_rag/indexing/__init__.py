"""Agentic RAG indexing pipelines for all vector databases.

This module provides indexing pipelines that prepare document collections for
agentic RAG retrieval. Each pipeline handles document loading, embedding
generation, and vector store indexing for a specific database.

Indexing Pipeline Architecture:
    All indexing pipelines follow a consistent three-phase pattern:

    1. Document Loading: Uses DataloaderCatalog to load documents from various
       sources (TriviaQA, ARC, PopQA, FactScore, EarningsCall datasets)

    2. Embedding Generation: Uses EmbedderHelper to generate vector embeddings
       using configured embedding models (OpenAI, HuggingFace, etc.)

    3. Vector Store Indexing: Creates or updates the vector store collection/
       index with the embedded documents

Pipeline Consistency:
    All pipelines share identical interfaces:
    - __init__(config_or_path): Initialize from dict or YAML file path
    - run() -> dict: Execute indexing and return statistics

    This consistency allows easy switching between databases without code changes.

Supported Databases:
    - ChromaAgenticRAGIndexingPipeline: Local embedded database
    - PineconeAgenticRAGIndexingPipeline: Managed cloud service
    - MilvusAgenticRAGIndexingPipeline: Scalable open-source
    - QdrantAgenticRAGIndexingPipeline: High-performance with filtering
    - WeaviateAgenticRAGIndexingPipeline: Schema-based with GraphQL

Usage:
    >>> from vectordb.langchain.agentic_rag.indexing import (
    ...     PineconeAgenticRAGIndexingPipeline,
    ... )
    >>> pipeline = PineconeAgenticRAGIndexingPipeline("config.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents")
"""

from vectordb.langchain.agentic_rag.indexing.chroma import (
    ChromaAgenticRAGIndexingPipeline,
)
from vectordb.langchain.agentic_rag.indexing.milvus import (
    MilvusAgenticRAGIndexingPipeline,
)
from vectordb.langchain.agentic_rag.indexing.pinecone import (
    PineconeAgenticRAGIndexingPipeline,
)
from vectordb.langchain.agentic_rag.indexing.qdrant import (
    QdrantAgenticRAGIndexingPipeline,
)
from vectordb.langchain.agentic_rag.indexing.weaviate import (
    WeaviateAgenticRAGIndexingPipeline,
)


__all__ = [
    "PineconeAgenticRAGIndexingPipeline",
    "WeaviateAgenticRAGIndexingPipeline",
    "ChromaAgenticRAGIndexingPipeline",
    "MilvusAgenticRAGIndexingPipeline",
    "QdrantAgenticRAGIndexingPipeline",
]
