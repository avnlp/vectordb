"""MMR indexing pipelines for all vector databases.

This module provides Haystack pipeline implementations for indexing documents
with embedding models optimized for MMR-based retrieval. The pipelines handle
document loading, embedding generation, and upsertion to the vector database.

Pipeline Design:
    Each database implementation follows a consistent pattern:
    1. Load documents from configured dataset (via DatasetRegistry)
    2. Generate embeddings using configured embedder (via EmbedderFactory)
    3. Create/manage collection/index in the vector database
    4. Upsert embedded documents to the database

Supported Databases:
    - Chroma: Local embedded vector database with SQLite persistence
    - Pinecone: Managed cloud vector database with namespace support
    - Milvus: Open-source scalable vector search with GPU acceleration
    - Qdrant: High-performance vector search with rich metadata filtering
    - Weaviate: Graph-based vector search with semantic capabilities

Architecture Notes:
    All indexing pipelines share a common 3-stage architecture:
    - Document Loading: Via DataloaderCatalog with configurable limits
    - Embedding Generation: Via EmbedderFactory with model configuration
    - Database Upsertion: Via database-specific VectorDB classes

    This consistency allows easy switching between databases by only
    changing the configuration file, not the application code.

Usage:
    >>> from vectordb.haystack.mmr.indexing import ChromaMmrIndexingPipeline
    >>> pipeline = ChromaMmrIndexingPipeline("config.yaml")
    >>> stats = pipeline.run()
    >>> print(f"Indexed {stats['documents_indexed']} documents")

Note:
    The pipelines use ConfigLoader for YAML configuration and validate
    required fields for each database type at initialization.
"""

from vectordb.haystack.mmr.indexing.chroma import ChromaMmrIndexingPipeline
from vectordb.haystack.mmr.indexing.milvus import MilvusMmrIndexingPipeline
from vectordb.haystack.mmr.indexing.pinecone import PineconeMmrIndexingPipeline
from vectordb.haystack.mmr.indexing.qdrant import QdrantMmrIndexingPipeline
from vectordb.haystack.mmr.indexing.weaviate import WeaviateMmrIndexingPipeline


__all__ = [
    "ChromaMmrIndexingPipeline",
    "MilvusMmrIndexingPipeline",
    "PineconeMmrIndexingPipeline",
    "QdrantMmrIndexingPipeline",
    "WeaviateMmrIndexingPipeline",
]
