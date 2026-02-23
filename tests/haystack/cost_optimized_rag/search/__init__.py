"""Tests for cost-optimized RAG database searchers.

This package contains integration tests for vector database searcher
implementations. Each searcher provides a unified interface for
retrieval across Chroma, Milvus, Pinecone, Qdrant, and Weaviate.

Searchers Tested:
    - ChromaSearcher: Local and persistent ChromaDB backend
        * Document retrieval with similarity search
        * Hybrid search support (where available)
        * Collection management and configuration

    - MilvusSearcher: Distributed Milvus vector database
        * High-performance similarity search
        * Partition and collection management
        * Scalable retrieval for large datasets

    - PineconeSearcher: Managed Pinecone service
        * Namespace-based query routing
        * Metadata filtering capabilities
        * Serverless and pod-based indexes

    - QdrantSearcher: Qdrant vector search engine
        * Payload filtering and search
        * Collection-level optimizations
        * Sparse vector support for hybrid search

    - WeaviateSearcher: Weaviate hybrid search engine
        * GraphQL query interface
        * Vector and BM25 hybrid search
        * Modular AI integrations

Test Coverage:
    - Initialization: Config loading, client setup, error handling
    - Search: Query execution, result formatting, top_k handling
    - Hybrid Search: Dense + sparse fusion, weight tuning
    - Edge Cases: Empty results, connection failures, timeouts
    - Cost Optimization: Query routing, caching, batch operations

Testing Infrastructure:
    Tests use mocked clients to avoid external dependencies in CI/CD.
    Integration tests marked with @pytest.mark.integration_test verify
    actual database connectivity in dedicated test environments.

Configuration:
    Searchers accept YAML configuration with database-specific settings.
    Tests validate configuration parsing and environment variable
    substitution for secure credential management.

Common Interface:
    All searchers implement the same search(query, top_k) signature,
    enabling database-agnostic RAG pipelines. Tests verify interface
    consistency across all implementations.
"""
