"""LangChain semantic search implementations for vector databases.

This module provides semantic (dense) search capabilities using LangChain
integrations with all supported vector databases. Semantic search uses dense
vector embeddings to find documents based on semantic meaning rather than
exact keyword matches.

Semantic Search vs Keyword Search:
    - Semantic search captures meaning and context, enabling retrieval of
      conceptually related documents even without keyword overlap
    - Uses dense embeddings (e.g., sentence-transformers, OpenAI embeddings)
      that encode semantic information into high-dimensional vectors
    - Handles synonyms, paraphrases, and cross-lingual queries naturally
    - Ideal for questions where users may not know the exact terminology

Architecture:
    indexing/: Document indexing pipelines for all vector stores
        - ChromaSemanticIndexingPipeline
        - PineconeSemanticIndexingPipeline
        - MilvusSemanticIndexingPipeline
        - QdrantSemanticIndexingPipeline
        - WeaviateSemanticIndexingPipeline

    search/: Semantic search pipelines for all vector stores
        - ChromaSemanticSearchPipeline
        - PineconeSemanticSearchPipeline
        - MilvusSemanticSearchPipeline
        - QdrantSemanticSearchPipeline
        - WeaviateSemanticSearchPipeline

Pipeline Flow:
    Indexing:
        1. Load documents from configured data source
        2. Generate dense embeddings using configured embedder
        3. Create collection/index in vector database
        4. Upsert documents with embeddings

    Search:
        1. Embed query using same embedder as indexing
        2. Perform similarity search in vector database
        3. Return top-k most similar documents
        4. Optionally generate RAG answer using retrieved documents

Supported Vector Databases:
    - Pinecone: Managed cloud service with metadata filtering
    - Chroma: Local embedded database for prototyping
    - Weaviate: Cloud-native with GraphQL interface
    - Milvus: High-performance distributed search
    - Qdrant: Open-source with payload filtering

Usage Example:
    Indexing:
        >>> from vectordb.langchain.semantic_search import (
        ...     ChromaSemanticIndexingPipeline,
        ... )
        >>> indexer = ChromaSemanticIndexingPipeline("config.yaml")
        >>> result = indexer.run()
        >>> print(f"Indexed {result['documents_indexed']} documents")

    Search:
        >>> from vectordb.langchain.semantic_search import ChromaSemanticSearchPipeline
        >>> searcher = ChromaSemanticSearchPipeline("config.yaml")
        >>> results = searcher.search("What is machine learning?", top_k=5)
        >>> for doc in results["documents"]:
        ...     print(doc.page_content[:200])

Configuration:
    Each pipeline requires a YAML configuration specifying:
        - Database connection (API keys, URLs, collection names)
        - Embedding model (provider, model name, dimensions)
        - Optional RAG LLM configuration for answer generation

Note:
    Semantic search requires the same embedding model for both indexing
    and search. Mismatched embedders will produce incompatible vectors
    and poor search results.
"""

from vectordb.langchain.semantic_search.indexing import (
    ChromaSemanticIndexingPipeline,
    MilvusSemanticIndexingPipeline,
    PineconeSemanticIndexingPipeline,
    QdrantSemanticIndexingPipeline,
    WeaviateSemanticIndexingPipeline,
)
from vectordb.langchain.semantic_search.search import (
    ChromaSemanticSearchPipeline,
    MilvusSemanticSearchPipeline,
    PineconeSemanticSearchPipeline,
    QdrantSemanticSearchPipeline,
    WeaviateSemanticSearchPipeline,
)


__all__ = [
    # Indexing pipelines
    "ChromaSemanticIndexingPipeline",
    "MilvusSemanticIndexingPipeline",
    "PineconeSemanticIndexingPipeline",
    "QdrantSemanticIndexingPipeline",
    "WeaviateSemanticIndexingPipeline",
    # Search pipelines
    "ChromaSemanticSearchPipeline",
    "MilvusSemanticSearchPipeline",
    "PineconeSemanticSearchPipeline",
    "QdrantSemanticSearchPipeline",
    "WeaviateSemanticSearchPipeline",
]
