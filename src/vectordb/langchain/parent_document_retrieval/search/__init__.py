"""Search pipelines for parent document retrieval.

This module provides database-specific search pipelines for parent document
retrieval using LangChain. Each pipeline implements the parent document retrieval
pattern: search on chunks, return full parent documents.

Search Pipeline Pattern:
    1. Embed the query using configured embedder
    2. Search vector database for similar chunks (top_k * 2 for redundancy)
    3. Extract chunk IDs from search results
    4. Map chunk IDs to parent documents via ParentDocumentStore
    5. Deduplicate and limit to top_k unique parent documents
    6. Optionally generate RAG answer using retrieved parents

Key Design Decisions:
    - Search retrieves more chunks than needed (top_k * 2) to account for multiple
      chunks from the same parent
    - Parent documents are deduplicated to avoid returning the same document twice
    - RAG generation is optional and only performed if LLM is configured
    - Each pipeline maintains its own ParentDocumentStore instance

Supported Vector Databases:
    - Pinecone: Cloud-native, serverless vector search
    - Chroma: Lightweight, embeddable vector database
    - Weaviate: Cloud-native, GraphQL-based vector search
    - Milvus: High-performance, distributed vector database
    - Qdrant: Open-source, filterable vector search

Example Usage:
    >>> from vectordb.langchain.parent_document_retrieval.search import (
    ...     PineconeParentDocumentRetrievalSearchPipeline,
    ... )
    >>>
    >>> # Initialize with configuration (must include parent_store path)
    >>> searcher = PineconeParentDocumentRetrievalSearchPipeline("config.yaml")
    >>>
    >>> # Execute search
    >>> results = searcher.search(
    ...     query="What is machine learning?",
    ...     top_k=5,
    ...     filters={"category": "technology"},
    ... )
    >>>
    >>> # Access parent documents
    >>> for doc in results["parent_documents"]:
    ...     print(f"Document: {doc['text'][:200]}...")
    >>>
    >>> # Access RAG answer if generated
    >>> if "answer" in results:
    ...     print(f"Answer: {results['answer']}")

Configuration:
    Each pipeline requires a YAML configuration file specifying:
        - Database connection parameters (API keys, URLs, etc.)
        - Embedding model configuration (must match indexing)
        - Parent store path (required for parent lookup)
        - Optional LLM configuration for RAG generation
        - Optional metadata filters

Note:
    The search pipeline requires the parent store file generated during
    indexing. The embedder configuration must match the indexing configuration
    for compatible embeddings.

Available Classes:
    - ChromaParentDocumentRetrievalSearchPipeline
    - MilvusParentDocumentRetrievalSearchPipeline
    - PineconeParentDocumentRetrievalSearchPipeline
    - QdrantParentDocumentRetrievalSearchPipeline
    - WeaviateParentDocumentRetrievalSearchPipeline
"""

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
    "ChromaParentDocumentRetrievalSearchPipeline",
    "MilvusParentDocumentRetrievalSearchPipeline",
    "PineconeParentDocumentRetrievalSearchPipeline",
    "QdrantParentDocumentRetrievalSearchPipeline",
    "WeaviateParentDocumentRetrievalSearchPipeline",
]
