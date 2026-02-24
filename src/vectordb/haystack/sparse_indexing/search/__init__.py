"""Search pipelines for sparse vector retrieval with BM25/SPLADE embeddings.

This module provides Haystack-compatible search pipelines for sparse-only vector
search using keyword-based retrieval models. Unlike dense embeddings that capture
semantic meaning, sparse embeddings capture term-level relevance for precise
keyword matching.

Architecture:
    Each database implementation follows the same search flow:
    1. Embed query using the same sparse embedder as indexing
    2. Convert query embedding to database-specific sparse format
    3. Query vector database with sparse vector only
    4. Convert results to Haystack Documents with relevance scores

Sparse search excels at:
    - Exact term and phrase matching
    - User queries with specific keywords
    - Scenarios where explainability matters
    - Complementing semantic search in hybrid pipelines

Key Components:
    - PineconeSparseSearchPipeline: Pinecone sparse_values format search
    - MilvusSparseSearchPipeline: Milvus sparse vector search
    - QdrantSparseSearchPipeline: Qdrant sparse vector search
    - WeaviateBM25SearchPipeline: Weaviate BM25/BM25S search
    - ChromaSparseSearchPipeline: Chroma sparse embedding search

Usage:
    >>> from vectordb.haystack.sparse_indexing.search import (
    ...     PineconeSparseSearchPipeline,
    ... )
    >>> search = PineconeSparseSearchPipeline("config.yaml")
    >>> results = search.search("machine learning algorithms", top_k=10)
    >>> for doc in results["documents"]:
    ...     print(f"Score: {doc.meta['score']:.3f} - {doc.content[:100]}...")

Integration Points:
    - vectordb.haystack.sparse_indexing.indexing: Complementary indexing pipelines
    - vectordb.haystack.utils: ConfigLoader, EmbedderFactory
    - haystack.dataclasses.SparseEmbedding: Haystack sparse embedding type
"""

from .chroma import ChromaSparseSearchPipeline
from .milvus import MilvusSparseSearchPipeline
from .pinecone import PineconeSparseSearchPipeline
from .qdrant import QdrantSparseSearchPipeline
from .weaviate import WeaviateBM25SearchPipeline


__all__ = [
    "ChromaSparseSearchPipeline",
    "MilvusSparseSearchPipeline",
    "PineconeSparseSearchPipeline",
    "QdrantSparseSearchPipeline",
    "WeaviateBM25SearchPipeline",
]
