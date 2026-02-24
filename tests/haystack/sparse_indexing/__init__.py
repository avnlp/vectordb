"""Tests for sparse indexing and hybrid search pipelines.

This package contains tests for sparse (BM25/TF-IDF) indexing and hybrid search
pipelines that combine sparse and dense retrieval methods. Hybrid search provides
better recall by leveraging both semantic understanding and lexical matching.

Sparse indexing methods:
    - BM25: Probabilistic ranking based on term frequency
    - TF-IDF: Term frequency-inverse document frequency weighting

Hybrid search approaches:
    - Linear combination: α * sparse_score + (1-α) * dense_score
    - Reciprocal Rank Fusion (RRF): Combines rankings from both methods
    - Query classification: Route to sparse or dense based on query type

Database implementations tested:
    - Chroma: Sparse vectors with metadata filtering
    - Milvus: Hybrid search with partition support
    - Pinecone: Sparse-dense vectors in single index
    - Qdrant: Sparse vectors with payload filtering
    - Weaviate: Hybrid search with GraphQL interface

Each implementation tests:
    - Sparse vector generation and indexing
    - Hybrid query processing
    - Fusion algorithm accuracy
    - Performance vs. pure dense/sparse search
"""
