"""Tests for query enhancement pipelines in Haystack.

This package contains tests for query enhancement implementations
in Haystack. Query enhancement improves retrieval quality by transforming
or expanding queries before search.

Enhancement techniques:
    - Multi-query: Generate multiple query variations
    - HyDE: Hypothetical Document Embeddings for query expansion
    - Step-back: Abstract queries for broader context retrieval
    - Query decomposition: Break complex queries into sub-queries

Query routing:
    - Classification-based: Route to appropriate retrieval strategy
    - Complexity analysis: Adjust retrieval depth based on query
    - Intent detection: Identify query type for targeted retrieval

Database implementations tested:
    - Chroma: Local query enhancement
    - Milvus: Multi-query parallel search
    - Pinecone: Namespace-aware query routing
    - Qdrant: Payload-filtered query expansion
    - Weaviate: GraphQL query transformation

Each implementation tests:
    - Enhancement quality and diversity
    - Retrieval improvement metrics
    - Latency impact of enhancement
    - Integration with RAG pipelines
"""
