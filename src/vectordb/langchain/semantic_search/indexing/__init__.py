"""Semantic search indexing pipelines for vector databases.

This module provides indexing pipeline implementations that prepare vector stores
for semantic (dense) search retrieval. These pipelines handle document loading,
embedding generation, and upsertion to the vector database.

Pipeline Components:
    - Document loader: Load documents from configured dataset (TriviaQA, ARC,
      PopQA, FactScore, EarningsCall)
    - Embedder: Generate dense vector embeddings using configured model
    - Vector store: Store embeddings with metadata for retrieval

Indexing Process:
    1. Load documents from data source using DataloaderCatalog
    2. Generate dense embeddings using EmbedderHelper
    3. Create or recreate collection/index in vector database
    4. Upsert documents with embeddings and metadata

Supported Vector Stores:
    - Chroma: Local embedded vector database with collection-based storage
    - Pinecone: Managed cloud vector database with namespace support
    - Milvus: Open-source scalable vector search with partition support
    - Qdrant: High-performance vector search with payload filtering
    - Weaviate: Graph-based vector search with schema enforcement

Embedding Models:
    Configurable via the embedder section in YAML:
        - sentence-transformers: Local models (all-MiniLM-L6-v2, etc.)
        - openai: OpenAI embedding models (text-embedding-3-small, etc.)
        - huggingface: HuggingFace inference API models

Pipeline Consistency:
    All indexing pipelines share identical interfaces:
        - __init__(config_or_path): Initialize from dict or YAML file path
        - run() -> dict: Execute indexing and return statistics

    This consistency enables easy database switching without code changes.

Example:
    >>> from vectordb.langchain.semantic_search.indexing.chroma import (
    ...     ChromaSemanticIndexingPipeline,
    ... )
    >>> pipeline = ChromaSemanticIndexingPipeline("configs/chroma_triviaqa.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents")
    >>> print(f"Collection: {result['collection_name']}")

See Also:
    vectordb.langchain.semantic_search.search: Semantic search pipelines
    vectordb.utils.embedder_helper: Embedding generation utilities
    vectordb.utils.data_loader_helper: Document loading utilities
"""

from vectordb.langchain.semantic_search.indexing.chroma import (
    ChromaSemanticIndexingPipeline,
)
from vectordb.langchain.semantic_search.indexing.milvus import (
    MilvusSemanticIndexingPipeline,
)
from vectordb.langchain.semantic_search.indexing.pinecone import (
    PineconeSemanticIndexingPipeline,
)
from vectordb.langchain.semantic_search.indexing.qdrant import (
    QdrantSemanticIndexingPipeline,
)
from vectordb.langchain.semantic_search.indexing.weaviate import (
    WeaviateSemanticIndexingPipeline,
)


__all__ = [
    "ChromaSemanticIndexingPipeline",
    "MilvusSemanticIndexingPipeline",
    "PineconeSemanticIndexingPipeline",
    "QdrantSemanticIndexingPipeline",
    "WeaviateSemanticIndexingPipeline",
]
