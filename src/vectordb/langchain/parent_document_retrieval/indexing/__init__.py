"""Indexing pipelines for parent document retrieval.

This module provides database-specific indexing pipelines for parent document
retrieval using LangChain. Each pipeline implements the same core pattern:

Indexing Pipeline Pattern:
    1. Load documents using configured dataloader
    2. Split documents into chunks using RecursiveCharacterTextSplitter
    3. Store parent documents in ParentDocumentStore
    4. Generate embeddings for chunks using configured embedder
    5. Index chunks in vector database with parent_id metadata
    6. Persist ParentDocumentStore to disk (optional)

Key Design Decisions:
    - Parent documents are stored separately from chunks to enable full-text retrieval
    - Chunks reference parents via parent_id stored in chunk metadata
    - Each parent gets a unique UUID, chunks get unique UUIDs
    - Parent store can be persisted to disk for later use during search

Supported Vector Databases:
    - Pinecone: Cloud-native, serverless vector search
    - Chroma: Lightweight, embeddable vector database
    - Weaviate: Cloud-native, GraphQL-based vector search
    - Milvus: High-performance, distributed vector database
    - Qdrant: Open-source, filterable vector search

Example Usage:
    >>> from vectordb.langchain.parent_document_retrieval.indexing import (
    ...     PineconeParentDocumentRetrievalIndexingPipeline,
    ... )
    >>>
    >>> # Initialize with configuration
    >>> indexer = PineconeParentDocumentRetrievalIndexingPipeline("config.yaml")
    >>>
    >>> # Run indexing pipeline
    >>> result = indexer.run()
    >>> print(f"Indexed {result['documents_indexed']} documents")
    >>> print(f"Created {result['chunks_created']} chunks")
    >>> print(f"Parent store saved to: {result['parent_store_path']}")

Configuration:
    Each pipeline requires a YAML configuration file specifying:
        - Database connection parameters (API keys, URLs, etc.)
        - Embedding model configuration
        - Chunking parameters (chunk_size, chunk_overlap, separators)
        - Dataloader settings (dataset, limit, etc.)
        - Parent store persistence settings (optional)

Note:
    The indexing pipeline must be run before the corresponding search
    pipeline. The parent store file generated during indexing must be
    available to the search pipeline.

Available Classes:
    - ChromaParentDocumentRetrievalIndexingPipeline
    - MilvusParentDocumentRetrievalIndexingPipeline
    - PineconeParentDocumentRetrievalIndexingPipeline
    - QdrantParentDocumentRetrievalIndexingPipeline
    - WeaviateParentDocumentRetrievalIndexingPipeline
"""

from vectordb.langchain.parent_document_retrieval.indexing.chroma import (
    ChromaParentDocumentRetrievalIndexingPipeline,
)
from vectordb.langchain.parent_document_retrieval.indexing.milvus import (
    MilvusParentDocumentRetrievalIndexingPipeline,
)
from vectordb.langchain.parent_document_retrieval.indexing.pinecone import (
    PineconeParentDocumentRetrievalIndexingPipeline,
)
from vectordb.langchain.parent_document_retrieval.indexing.qdrant import (
    QdrantParentDocumentRetrievalIndexingPipeline,
)
from vectordb.langchain.parent_document_retrieval.indexing.weaviate import (
    WeaviateParentDocumentRetrievalIndexingPipeline,
)


__all__ = [
    "ChromaParentDocumentRetrievalIndexingPipeline",
    "MilvusParentDocumentRetrievalIndexingPipeline",
    "PineconeParentDocumentRetrievalIndexingPipeline",
    "QdrantParentDocumentRetrievalIndexingPipeline",
    "WeaviateParentDocumentRetrievalIndexingPipeline",
]
