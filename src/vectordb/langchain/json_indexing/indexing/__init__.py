"""JSON indexing pipelines for LangChain.

This module provides indexing pipelines for JSON document storage across multiple
vector database backends. JSON indexing enables structured data storage with vector
embeddings, allowing semantic search over JSON documents while preserving structured
fields for filtering.

JSON Indexing Features:
    - Nested field extraction: Flatten nested JSON into searchable content
    - Schema validation: Validate documents against expected JSON schemas
    - Field-level indexing: Index specific JSON fields with metadata
    - Dynamic mapping: Auto-detect and index JSON structures

Available Backends:
    - ChromaJsonIndexingPipeline: Local JSON document storage with metadata filtering
    - MilvusJsonIndexingPipeline: Cloud-native JSON indexing with dynamic fields
    - PineconeJsonIndexingPipeline: Managed service with JSON metadata support
    - QdrantJsonIndexingPipeline: On-premise JSON payload indexing
    - WeaviateJsonIndexingPipeline: Graph-vector JSON property storage

Usage:
    >>> from vectordb.langchain.json_indexing.indexing import ChromaJsonIndexingPipeline
    >>> config = {
    ...     "dataloader": {"type": "triviaqa", "split": "test"},
    ...     "embeddings": {"model": "all-MiniLM-L6-v2"},
    ...     "chroma": {"path": "./data", "collection_name": "json_docs"},
    ... }
    >>> pipeline = ChromaJsonIndexingPipeline(config)
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents")

Each pipeline follows a standard workflow:
    1. Load JSON documents from configured dataloader
    2. Extract text content for embedding
    3. Generate embeddings using configured model
    4. Create collection in vector database
    5. Upsert documents with embeddings and JSON metadata
"""

from vectordb.langchain.json_indexing.indexing.chroma import (
    ChromaJsonIndexingPipeline,
)
from vectordb.langchain.json_indexing.indexing.milvus import (
    MilvusJsonIndexingPipeline,
)
from vectordb.langchain.json_indexing.indexing.pinecone import (
    PineconeJsonIndexingPipeline,
)
from vectordb.langchain.json_indexing.indexing.qdrant import (
    QdrantJsonIndexingPipeline,
)
from vectordb.langchain.json_indexing.indexing.weaviate import (
    WeaviateJsonIndexingPipeline,
)


__all__ = [
    "ChromaJsonIndexingPipeline",
    "MilvusJsonIndexingPipeline",
    "PineconeJsonIndexingPipeline",
    "QdrantJsonIndexingPipeline",
    "WeaviateJsonIndexingPipeline",
]
