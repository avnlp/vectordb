"""Tests for Maximal Marginal Relevance (MMR) search pipelines.

This package contains tests for MMR-based retrieval pipelines that balance
relevance and diversity in search results. MMR helps reduce redundancy while
maintaining high relevance to the query.

MMR algorithm:
    MMR = argmax[λ * Sim(d, q) - (1-λ) * max(Sim(d, d'))]
    where:
        - λ: Trade-off parameter between relevance and diversity
        - Sim(d, q): Similarity between document and query
        - Sim(d, d'): Similarity between candidate and selected documents

Database implementations tested:
    - Chroma MMR: Local database with MMR reranking
    - Milvus MMR: Cloud-native MMR with partition support
    - Pinecone MMR: Managed service MMR with metadata filtering
    - Qdrant MMR: High-performance MMR with payload filtering
    - Weaviate MMR: Graph-vector hybrid MMR

Each implementation tests:
    - Indexing: Document storage with embeddings
    - MMR search: Diverse result retrieval
    - Lambda parameter: Relevance-diversity trade-off
    - Top-k selection: Number of results
    - Integration with Haystack pipelines
"""
