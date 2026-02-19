"""Tests for Haystack integrations.

This package contains comprehensive tests for all Haystack pipeline
integrations with vector databases. Tests cover:

Database integrations:
    - Chroma: Local vector database
    - Milvus: Cloud-native vector database
    - Pinecone: Managed vector database
    - Qdrant: High-performance vector database
    - Weaviate: Graph-vector database

Haystack components tested:
    - Document stores and embedders
    - MMR (Maximal Marginal Relevance) search pipelines
    - Contextual compression pipelines
    - Metadata filtering and querying
    - Reranking components
    - Diversity filtering and ranking
    - Semantic search pipelines
    - Agentic RAG components

Each database integration is tested for:
    - Indexing: Document storage and embedding
    - Search: Similarity retrieval with various filters
    - Pipeline composition: Integration with Haystack pipelines
"""
