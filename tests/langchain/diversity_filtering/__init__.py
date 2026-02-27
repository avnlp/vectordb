"""Tests for diversity filtering pipelines (LangChain).

This package contains tests for diversity filtering implementations in
LangChain. Diversity filtering improves result quality by ensuring retrieved
documents cover different aspects of the query topic rather than being
semantically similar to each other.

Diversity algorithms tested:
    - MMR (Maximal Marginal Relevance): Balances relevance and diversity
    - Clustering-based: Groups documents and selects representatives
    - Coverage-based: Ensures topic coverage across results
    - Distance-based: Filters by minimum inter-document distance

Database implementations tested:
    - Chroma: Local diversity filtering with LangChain retrievers
    - Milvus: Cloud-native MMR with partition support
    - Pinecone: Managed service diversity with metadata filtering
    - Qdrant: On-premise diversity with payload filtering
    - Weaviate: Graph-vector hybrid diversity filtering

Each implementation tests:
    - Diversity metric calculation
    - Relevance-diversity trade-off (lambda parameter)
    - Result set coverage improvement
    - Performance vs. pure similarity search
    - Integration with LangChain retrievers
"""
