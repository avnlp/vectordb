"""Tests for diversity filtering indexing and search pipelines.

This package contains integration tests for unified pipelines that add
diversity filtering (MMR) to vector database retrieval across Chroma,
Milvus, Pinecone, Qdrant, and Weaviate.

Pipeline Architecture:
    - Indexing Pipelines: Document embedding and storage
        * Dataset loading via DataloaderCatalog
        * Document embedding using sentence-transformers
        * Vector database indexing with metadata
        * Progress tracking and error handling

    - Search Pipelines: Retrieval with diversity filtering
        * Query embedding for semantic search
        * Candidate retrieval from vector database (top_k_candidates)
        * MMR (Maximum Margin Relevance) diversity ranking
        * Optional RAG response generation with diverse context

Databases Tested:
    - Qdrant: Payload filtering, sparse vectors, hybrid search
    - Pinecone: Namespace routing, metadata filtering, serverless indexes
    - Weaviate: GraphQL queries, hybrid BM25+vector search, modular AI
    - Chroma: Local persistent storage, simple embedding workflows
    - Milvus: Distributed architecture, high-scale retrieval

Diversity Filtering Integration:
    All search pipelines integrate SentenceTransformersDiversityRanker:
    1. Retrieve top_k_candidates from vector DB (default 100)
    2. Apply MMR algorithm with configurable lambda parameter
    3. Return top_k diverse results covering multiple semantic aspects

Test Coverage:
    - End-to-end indexing workflows with sample data
    - Search with various query types and filters
    - MMR diversity quality and coverage metrics
    - RAG generation with diverse context documents
    - Error handling for connection failures and timeouts
    - Configuration loading and validation

Testing Strategy:
    Unit tests use mocks for fast execution without database dependencies.
    Integration tests (marked @pytest.mark.integration_test) verify actual
    database connectivity and query execution in test environments.

Configuration Testing:
    - YAML configuration loading for each database
    - Environment variable substitution for credentials
    - Database-specific parameter validation
    - Invalid configuration error handling

Dataset Support:
    - TriviaQA: Question-answer pairs for knowledge retrieval
    - ARC: Science exam questions for reasoning evaluation
    - PopQA: Popular entity questions for fact checking
    - FactScore: Document-level fact verification
    - Earnings Calls: Financial document analysis
"""
