"""Semantic search pipelines for vector databases.

This module provides search pipeline implementations that execute semantic
(dense vector) search against indexed vector stores. These pipelines handle
query embedding, similarity search, and optional RAG answer generation.

Search Pipeline Architecture:
    1. Query Embedding: Convert query text to dense vector using embedder
    2. Similarity Search: Query vector database for top-k similar documents
    3. Result Formatting: Convert database results to LangChain Document objects
    4. Optional RAG: Generate answer using retrieved documents if LLM configured

Similarity Metrics:
    Different vector databases use different similarity metrics:
        - Cosine similarity: Pinecone, Chroma, Qdrant (most common)
        - Dot product: Milvus (configurable)
        - L2 distance: Weaviate (converted to similarity score)

    All pipelines normalize results to a consistent format with similarity scores.

Supported Vector Stores:
    - Chroma: Local search with optional filtering
    - Pinecone: Cloud search with metadata filtering
    - Milvus: Partition-aware search with batch queries
    - Qdrant: Payload-filtered search with scroll API
    - Weaviate: GraphQL-based search with BM25 hybrid option

Search Parameters:
    - query: The search query string
    - top_k: Number of results to return (default: 10)
    - filters: Optional metadata filters (database-specific syntax)

Results Format:
    All pipelines return a dictionary containing:
        - 'documents': List of LangChain Document objects with page_content
        - 'query': The original query string
        - 'answer': Generated answer if RAG/LLM is configured

Usage:
    >>> from vectordb.langchain.semantic_search.search.chroma import (
    ...     ChromaSemanticSearchPipeline,
    ... )
    >>> pipeline = ChromaSemanticSearchPipeline("configs/chroma_triviaqa.yaml")
    >>> results = pipeline.search(
    ...     query="What is deep learning?",
    ...     top_k=5,
    ...     filters={"category": "technology"},
    ... )
    >>> print(f"Found {len(results['documents'])} documents")
    >>> for doc in results["documents"]:
    ...     print(f"Score: {doc.metadata.get('score', 'N/A')}")
    ...     print(f"Content: {doc.page_content[:100]}...")

See Also:
    vectordb.langchain.semantic_search.indexing: Document indexing pipelines
    vectordb.utils.embedder_helper: Query embedding utilities
"""

from vectordb.langchain.semantic_search.search.chroma import (
    ChromaSemanticSearchPipeline,
)
from vectordb.langchain.semantic_search.search.milvus import (
    MilvusSemanticSearchPipeline,
)
from vectordb.langchain.semantic_search.search.pinecone import (
    PineconeSemanticSearchPipeline,
)
from vectordb.langchain.semantic_search.search.qdrant import (
    QdrantSemanticSearchPipeline,
)
from vectordb.langchain.semantic_search.search.weaviate import (
    WeaviateSemanticSearchPipeline,
)


__all__ = [
    "ChromaSemanticSearchPipeline",
    "MilvusSemanticSearchPipeline",
    "PineconeSemanticSearchPipeline",
    "QdrantSemanticSearchPipeline",
    "WeaviateSemanticSearchPipeline",
]
