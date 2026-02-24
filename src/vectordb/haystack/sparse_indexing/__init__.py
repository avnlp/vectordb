"""Sparse indexing pipelines for keyword/BM25-style search.

Sparse indexing enables lexical (keyword-based) retrieval as an alternative or
complement to dense semantic search. While dense embeddings capture meaning,
sparse representations excel at exact term matching and handling domain-specific
vocabulary that embedding models may not understand.

Sparse Embedding Approaches:

SPLADE (Sparse Lexical and Expansion Model):
    A learned sparse representation that uses a pretrained language model to
    predict term importance weights. Unlike traditional BM25, SPLADE learns
    to expand documents with semantically related terms, bridging the gap
    between lexical and semantic matching.

    Properties:
    - Produces sparse vectors (mostly zeros, a few non-zero term weights)
    - Supports term expansion (document mentions "car", index includes "vehicle")
    - Compatible with inverted index structures for efficient retrieval

BM25 (Best Matching 25):
    Classical probabilistic retrieval model based on term frequency and inverse
    document frequency. Weaviate provides native BM25 support without requiring
    external sparse embeddings.

    Scoring: BM25(D, Q) = sum(IDF(qi) * TF(qi, D) * (k1 + 1) / (TF + k1 * ...))

Database-Specific Implementation:
    - Pinecone: Sparse vectors via sparse_values parameter
    - Milvus: SPARSE_FLOAT_VECTOR field with SPARSE_INVERTED_INDEX
    - Qdrant: Sparse vectors with named vector configuration
    - Weaviate: Native BM25 (no sparse embedder needed)
    - Chroma: Not supported in open-source version (Chroma Cloud only)

Usage:
    >>> from vectordb.haystack.sparse_indexing import MilvusSparseIndexingPipeline
    >>> indexer = MilvusSparseIndexingPipeline("config.yaml")
    >>> result = indexer.run()  # Index documents with sparse embeddings

    >>> from vectordb.haystack.sparse_indexing import MilvusSparseSearchPipeline
    >>> searcher = MilvusSparseSearchPipeline("config.yaml")
    >>> results = searcher.run(query="machine learning", top_k=10)
"""

from .indexing import (
    ChromaSparseIndexingPipeline,
    MilvusSparseIndexingPipeline,
    PineconeSparseIndexingPipeline,
    QdrantSparseIndexingPipeline,
    WeaviateBM25IndexingPipeline,
)
from .search import (
    ChromaSparseSearchPipeline,
    MilvusSparseSearchPipeline,
    PineconeSparseSearchPipeline,
    QdrantSparseSearchPipeline,
    WeaviateBM25SearchPipeline,
)


__all__ = [
    # Indexing pipelines
    "PineconeSparseIndexingPipeline",
    "MilvusSparseIndexingPipeline",
    "QdrantSparseIndexingPipeline",
    "WeaviateBM25IndexingPipeline",
    "ChromaSparseIndexingPipeline",
    # Search pipelines
    "PineconeSparseSearchPipeline",
    "MilvusSparseSearchPipeline",
    "QdrantSparseSearchPipeline",
    "WeaviateBM25SearchPipeline",
    "ChromaSparseSearchPipeline",
]
