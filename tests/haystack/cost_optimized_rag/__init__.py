"""Tests for cost-optimized RAG pipelines in Haystack.

This package contains tests for cost-optimized RAG (Retrieval-Augmented Generation)
implementations in Haystack. These tests verify strategies for reducing API costs
and latency while maintaining retrieval quality.

Cost optimization strategies:
    - Query classification: Route queries to appropriate retrieval strategies
    - Tiered retrieval: Use cheaper retrievers for simple queries
    - Response caching: Cache frequent query results
    - Batch processing: Group multiple queries for efficiency
    - Model selection: Use smaller models for simpler tasks

Implementation components:
    - Smart routing based on query complexity
    - Fallback mechanisms for failed retrievals
    - Dynamic retrieval depth adjustment
    - Embedding caching and reuse

Database implementations tested:
    - Chroma: Cost-optimized retrieval with caching
    - Milvus: Tiered search strategies
    - Pinecone: Namespace-based routing
    - Qdrant: Collection-level optimization
    - Weaviate: GraphQL query optimization

Each implementation tests:
    - Cost reduction metrics
    - Latency improvements
    - Quality preservation
    - Fallback behavior
"""
