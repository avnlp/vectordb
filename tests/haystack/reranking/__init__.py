"""Tests for reranking in Haystack.

This package contains tests for reranking implementations in Haystack.
Reranking improves retrieval quality by applying a more sophisticated
scoring model to initial search results.

Reranking workflow:
    1. Initial retrieval: Fast vector similarity search
    2. Candidate selection: Top-K results from initial search
    3. Reranking: Apply cross-encoder or LLM-based scoring
    4. Final ranking: Return top-N reranked results

Reranking approaches:
    - Cross-encoders: BERT-based models for query-document scoring
    - LLM-based: Use large language models for relevance scoring
    - ColBERT: Late interaction for efficient reranking
    - Learned sparse: Neural sparse retrieval models

Benefits of reranking:
    - Improved precision over vector similarity alone
    - Better handling of semantic nuances
    - Query-aware document scoring
    - Reduced false positives

Database implementations tested:
    - Chroma: Post-retrieval reranking pipeline
    - Milvus: Multi-stage retrieval with reranking
    - Pinecone: Metadata-based reranking
    - Qdrant: Score modification and reranking
    - Weaviate: GraphQL result reranking

Each implementation tests:
    - Reranking accuracy improvement
    - Latency impact
    - Integration with retrieval pipelines
    - Score normalization
"""
