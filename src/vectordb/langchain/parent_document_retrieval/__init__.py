"""LangChain parent document retrieval implementation for vector databases.

This module implements parent document retrieval using LangChain, enabling
dual-level storage where chunks are indexed for semantic search but full parent
documents are returned for context-rich responses.

Parent document retrieval works by:
    1. Splitting parent documents into smaller chunks
    2. Indexing chunk embeddings for efficient similarity search
    3. Maintaining parent-child mappings in a ParentDocumentStore
    4. Retrieving chunks by similarity, then mapping back to full parents

Supported vector databases:
    - Pinecone: Cloud-native, serverless vector search
    - Chroma: Lightweight, embeddable vector database
    - Weaviate: Cloud-native, GraphQL-based vector search
    - Milvus: High-performance, distributed vector database
    - Qdrant: Open-source, filterable vector search

Example:
    >>> from vectordb.langchain.parent_document_retrieval import (
    ...     PineconeParentDocumentRetrievalIndexingPipeline,
    ...     PineconeParentDocumentRetrievalSearchPipeline,
    ... )
    >>>
    >>> # Index documents
    >>> indexer = PineconeParentDocumentRetrievalIndexingPipeline("config.yaml")
    >>> result = indexer.run()
    >>>
    >>> # Search and retrieve parent documents
    >>> searcher = PineconeParentDocumentRetrievalSearchPipeline("config.yaml")
    >>> results = searcher.search("query text", top_k=5)

Architecture:
    - indexing/: Database-specific indexing pipelines for chunking and storage
    - search/: Database-specific search pipelines for retrieval and RAG
    - parent_store.py: Central store for parent-child document mappings
    - configs/: YAML configuration files for different datasets

Note:
    Parent document retrieval improves context quality by returning complete
documents instead of fragmented chunks, while maintaining search precision
through chunk-level embeddings.
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
from vectordb.langchain.parent_document_retrieval.parent_store import (
    ParentDocumentStore,
)
from vectordb.langchain.parent_document_retrieval.search.chroma import (
    ChromaParentDocumentRetrievalSearchPipeline,
)
from vectordb.langchain.parent_document_retrieval.search.milvus import (
    MilvusParentDocumentRetrievalSearchPipeline,
)
from vectordb.langchain.parent_document_retrieval.search.pinecone import (
    PineconeParentDocumentRetrievalSearchPipeline,
)
from vectordb.langchain.parent_document_retrieval.search.qdrant import (
    QdrantParentDocumentRetrievalSearchPipeline,
)
from vectordb.langchain.parent_document_retrieval.search.weaviate import (
    WeaviateParentDocumentRetrievalSearchPipeline,
)


__all__ = [
    # Indexing pipelines
    "ChromaParentDocumentRetrievalIndexingPipeline",
    "MilvusParentDocumentRetrievalIndexingPipeline",
    "PineconeParentDocumentRetrievalIndexingPipeline",
    "QdrantParentDocumentRetrievalIndexingPipeline",
    "WeaviateParentDocumentRetrievalIndexingPipeline",
    # Search pipelines
    "ChromaParentDocumentRetrievalSearchPipeline",
    "MilvusParentDocumentRetrievalSearchPipeline",
    "PineconeParentDocumentRetrievalSearchPipeline",
    "QdrantParentDocumentRetrievalSearchPipeline",
    "WeaviateParentDocumentRetrievalSearchPipeline",
    # Parent store
    "ParentDocumentStore",
]
