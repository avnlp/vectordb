"""Tests for query enhancement pipelines (LangChain).

This package contains tests for query enhancement implementations in LangChain.
Query enhancement improves retrieval quality by transforming or expanding queries
before searching the vector database.

Enhancement techniques tested:
    - Multi-query expansion: Generate variations of the original query
    - HyDE (Hypothetical Document Embeddings): Generate hypothetical answer
    - Step-back prompting: Generate broader context query
    - Query decomposition: Break complex queries into sub-queries
    - Relevance feedback: Use previous results to refine query

Database implementations tested:
    - Chroma: Enhanced queries with local search
    - Milvus: Cloud-native query processing
    - Pinecone: Managed service with query routing
    - Qdrant: On-premise query enhancement
    - Weaviate: Graph-vector hybrid queries

Each implementation tests:
    - Query transformation accuracy
    - Retrieval improvement metrics
    - Latency impact of enhancement
    - Integration with LangChain chains
    - Fallback behavior on enhancement failure
"""
