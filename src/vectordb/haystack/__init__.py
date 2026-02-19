"""Haystack integrations for vector database operations.

This module provides pipeline components and integration classes for building
RAG (Retrieval-Augmented Generation) applications using the Haystack framework.
It supports multiple vector databases with consistent APIs across all backends.

Features:
    - Dense retrieval with configurable embedding models
    - Hybrid search with sparse embeddings (BM25-style)
    - MMR (Maximal Marginal Relevance) for diversity
    - Parent document retrieval for hierarchical chunking
    - Cost-optimized pipelines for production
    - Context compression and query enhancement
    - Agentic routing and self-reflection loops

Usage:
    >>> from vectordb.databases.chroma import ChromaVectorDB
    >>> from vectordb.haystack.mmr import ChromaMmrSearchPipeline
"""
