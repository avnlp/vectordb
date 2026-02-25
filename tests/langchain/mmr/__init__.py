"""Tests for Maximal Marginal Relevance (MMR) pipelines (LangChain).

This package contains tests for MMR-based retrieval implementations in
LangChain. MMR balances relevance to the query with diversity among results
to provide comprehensive coverage of the topic.

MMR formula:
    MMR = argmax[λ * Sim(d, q) - (1-λ) * max(Sim(d, d'))]
    where:
        - λ (lambda): Trade-off parameter (0=diversity, 1=relevance)
        - Sim(d, q): Similarity between document and query
        - Sim(d, d'): Similarity to already selected documents

Database implementations tested:
    - Chroma: Local MMR with LangChain retrievers
    - Milvus: Cloud-native MMR with partition support
    - Pinecone: Managed service MMR with namespace filtering
    - Qdrant: On-premise MMR with payload filtering
    - Weaviate: Graph-vector hybrid MMR

Each implementation tests:
    - Lambda parameter tuning
    - Relevance-diversity trade-off
    - Top-k selection with diversity
    - Performance vs. pure similarity search
    - Integration with LangChain chains
"""
