"""Tests for hybrid indexing pipelines (LangChain).

This package contains tests for hybrid indexing implementations in LangChain.
Hybrid indexing combines sparse (BM25/TF-IDF) and dense (embedding) retrieval
to achieve better recall and precision than either method alone.

Hybrid search approaches:
    - Linear combination: α * sparse_score + (1-α) * dense_score
    - Reciprocal Rank Fusion (RRF): Combines rankings from both methods
    - Query routing: Routes to sparse or dense based on query classification
    - Ensemble retrieval: Merges results from multiple retrievers

Database implementations tested:
    - Chroma: Hybrid vectors with metadata filtering
    - Milvus: Multi-vector search with partition support
    - Pinecone: Sparse-dense vectors in unified index
    - Qdrant: Hybrid search with payload filtering
    - Weaviate: Hybrid fusion with GraphQL interface

Each implementation tests:
    - Sparse vector generation (BM25/Splade)
    - Dense vector encoding
    - Fusion algorithm effectiveness
    - Query-time hybrid scoring
    - Performance vs. pure sparse/dense search
    - Integration with LangChain retrievers
"""
