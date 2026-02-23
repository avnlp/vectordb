"""Tests for reranking pipelines in LangChain.

This package contains tests for document reranking implementations in LangChain.
Reranking improves retrieval precision by applying a secondary scoring model
to re-order initial search results based on query-document relevance.

Reranking approaches tested:
    - Cross-encoder reranking: Bi-directional attention for query-doc pairs
    - ColBERT reranking: Late interaction with MaxSim scoring
    - Cohere reranking: Cloud-based neural reranker API
    - FlashRank reranking: Lightweight local reranker models

Two-stage retrieval pipeline:
    1. Initial retrieval: Fast vector similarity search (high recall)
    2. Reranking: Precise relevance scoring (high precision)

Database implementations tested:
    - Chroma: Local reranking with cross-encoder models
    - Milvus: Cloud-native retrieval with reranking integration
    - Pinecone: Managed service with post-retrieval reranking
    - Qdrant: On-premise reranking with payload filtering
    - Weaviate: Graph-vector hybrid with reranking layer

Each implementation tests:
    - Initial retrieval followed by reranking
    - Reranker model initialization and inference
    - Top-k selection after reranking
    - RAG integration with reranked results
    - Performance characteristics of two-stage retrieval
"""
