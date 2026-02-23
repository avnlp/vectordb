"""Tests for cost-optimized RAG base modules.

This package contains tests for foundational components that support
cost-efficient RAG pipeline implementations. These tests verify the core
functionality for result fusion and retrieval quality metrics.

Test Coverage:
    - ResultFuser: Hybrid search result fusion strategies
        * Reciprocal Rank Fusion (RRF) for combining dense and sparse results
        * Weighted fusion with configurable balance parameters
        * Merge strategies for multi-source search results
    - RetrievalMetrics: Quality assessment without external dependencies
        * MRR (Mean Reciprocal Rank): Position of first relevant result
        * Recall@K: Coverage of relevant documents in top-K
        * Precision@K: Accuracy of retrieved documents
        * NDCG@K: Ranking quality with position weighting
        * MAP@K: Mean Average Precision across queries
    - MetricsAggregator: Batch metric computation and summarization
        * Per-query metric tracking
        * Aggregate statistics across query sets
        * Report generation for evaluation workflows

Testing Strategy:
    Unit tests cover edge cases and boundary conditions for each metric
    computation. Fusion tests verify correct score calculations and proper
    handling of empty or partial results. Metric tests validate mathematical
    correctness against known values.

Integration Points:
    These base modules are imported by searcher implementations across all
    vector databases (Chroma, Milvus, Pinecone, Qdrant, Weaviate). Tests
    ensure consistent behavior regardless of which database is used.

Dependencies:
    - pytest for test framework
    - No external database dependencies (pure unit tests)
    - Tests run quickly without network or service requirements
"""
