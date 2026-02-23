"""Tests for cost-optimized RAG evaluation framework.

This package contains tests for the RAG evaluation system that measures
retrieval quality and cost efficiency across vector database implementations.

RAGEvaluator Tests:
    - Initialization: Config loading, searcher integration, logger setup
    - Single Query Evaluation: Latency measurement, metric computation
    - Batch Evaluation: Multi-query processing, aggregate statistics
    - Report Generation: Summary statistics, trend analysis

Evaluation Metrics Validated:
    - MRR (Mean Reciprocal Rank): First relevant result position
    - Recall@K: Proportion of relevant documents retrieved
    - Precision@K: Accuracy of retrieved documents
    - NDCG@K: Ranking quality with logarithmic discounting
    - MAP@K (Mean Average Precision): Precision averaged across ranks
    - Latency: Query execution time in milliseconds

Cost-Aware Testing:
    - Evaluation sampling to reduce API costs during testing
    - Synthetic data generation for load testing
    - Progressive evaluation from coarse to fine granularity
    - Benchmark cost estimation without full execution

Test Fixtures:
    - MockRAGConfig: Configuration objects for test scenarios
    - MockSearcher: Simulated search results for deterministic testing
    - SampleQueries: Representative query sets with relevance judgments

Evaluation Workflows Tested:
    - Development Phase: Quick smoke tests (10-50 queries)
    - Production Validation: Full test suite (1000+ queries)
    - Regression Testing: CI/CD automated evaluation with thresholds
    - A/B Testing: Comparison of retrieval strategies

Integration Points:
    Evaluator integrates with MetricsAggregator from base modules for
    consistent metric computation. Results feed into cost optimization
    decisions and quality monitoring dashboards.

Validation Strategy:
    Unit tests use mocks for fast, deterministic execution.
    Integration tests verify end-to-end workflows with real databases.
    All metrics are validated against reference implementations.
"""
