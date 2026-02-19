"""Haystack integrations for vector database operations.

This module provides pipeline components and integration classes for building
RAG (Retrieval-Augmented Generation) applications using the Haystack framework.
It supports multiple vector databases with consistent APIs across all backends.

Vector Database Support:
    - Pinecone: Cloud-native vector database with sparse-dense hybrid search
    - Weaviate: Open-source vector search with GraphQL and Graph-OOD
    - Chroma: Embedded vector database for local development
    - Milvus: Open-source vector database with partition namespaces
    - Qdrant: High-performance vector search with payload filtering

Features:
    - Dense retrieval with configurable embedding models
    - Hybrid search with sparse embeddings (BM25-style)
    - MMR (Maximal Marginal Relevance) for diversity
    - Parent document retrieval for hierarchical chunking
    - Cost-optimized pipelines for production
    - Context compression and query enhancement
    - Agentic routing and self-reflection loops

Usage:
    >>> from vectordb.haystack import ChromaVectorDB, PineconeVectorDB
    >>> from vectordb.haystack.mmr import ChromaMmrSearchPipeline
"""
