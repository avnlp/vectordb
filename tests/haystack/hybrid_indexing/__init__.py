"""Tests for hybrid indexing pipelines in Haystack.

This package contains tests for hybrid indexing implementations that
combine dense and sparse vector representations. Hybrid indexing enables
search strategies that leverage both semantic and lexical matching.

Hybrid indexing components:
    - Dense vectors: Semantic embeddings from transformer models
    - Sparse vectors: BM25/TF-IDF term frequency representations
    - Fusion layer: Combines sparse and dense scores

Indexing strategies:
    - Parallel indexing: Index dense and sparse separately
    - Single-index hybrid: Store both representations together
    - Delayed sparse: Generate sparse vectors at query time

Database implementations tested:
    - Chroma: Metadata-based hybrid storage
    - Milvus: Native hybrid vector support
    - Pinecone: Sparse-dense index integration
    - Qdrant: Multi-vector hybrid indexing
    - Weaviate: BM25 + vector hybrid

Each implementation tests:
    - Hybrid document indexing
    - Vector storage and retrieval
    - Index consistency and integrity
    - Query-time fusion preparation
"""
