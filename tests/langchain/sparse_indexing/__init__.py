"""Tests for sparse indexing in LangChain.

This package contains tests for sparse (BM25/TF-IDF) indexing implementations
in LangChain. Sparse indexing provides lexical matching capabilities that
complement semantic (dense) retrieval.

Sparse indexing methods:
    - BM25: Probabilistic ranking based on term frequency and document length
    - TF-IDF: Term frequency-inverse document frequency weighting
    - Splade: Learned sparse representations with neural expansion

Integration with LangChain:
    - Custom retrievers for sparse search
    - Hybrid search combining sparse and dense
    - Document store adapters for sparse vectors

Database implementations tested:
    - Chroma: Sparse vectors with metadata
    - Milvus: Sparse vector collections
    - Pinecone: Sparse-dense hybrid vectors
    - Qdrant: Sparse vectors with payloads
    - Weaviate: Hybrid sparse-dense search

Each implementation tests:
    - Sparse vector generation
    - Lexical matching accuracy
    - Integration with LangChain retrievers
    - Performance characteristics
"""
