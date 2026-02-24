"""Indexing pipelines for sparse vector indexing with BM25/SPLADE embeddings.

This module provides Haystack-compatible indexing pipelines for sparse-only vector
indexing using keyword-based retrieval models (BM25, SPLADE). Sparse embeddings
represent documents as term-weighted vectors where only non-zero dimensions are
stored, enabling efficient keyword-style search without dense vectors.

Architecture:
    Each database implementation follows the same pattern:
    1. Load documents from configured dataloader
    2. Generate sparse embeddings using SPLADE models
    3. Convert to database-specific sparse format
    4. Upsert to vector database with metadata

Sparse indexing is ideal for:
    - Keyword-based search without embedding models
    - Exact term matching scenarios
    - Hybrid search with sparse+dense (where supported)
    - Memory-efficient indexing of large corpora

Key Components:
    - PineconeSparseIndexingPipeline: Pinecone sparse_values format indexing
    - MilvusSparseIndexingPipeline: Milvus sparse vector indexing
    - QdrantSparseIndexingPipeline: Qdrant sparse vector indexing
    - WeaviateBM25IndexingPipeline: Weaviate BM25/BM25S indexing
    - ChromaSparseIndexingPipeline: Chroma sparse embedding indexing

Usage:
    >>> from vectordb.haystack.sparse_indexing.indexing import (
    ...     PineconeSparseIndexingPipeline,
    ... )
    >>> pipeline = PineconeSparseIndexingPipeline("config.yaml")
    >>> pipeline.create_index(dimension=1, metric="dotproduct")
    >>> result = pipeline.run()

Integration Points:
    - vectordb.haystack.sparse_indexing.search: Complementary search pipelines
    - vectordb.haystack.utils: ConfigLoader, EmbedderFactory
    - haystack.dataclasses.SparseEmbedding: Haystack sparse embedding type
"""

from .chroma import ChromaSparseIndexingPipeline
from .milvus import MilvusSparseIndexingPipeline
from .pinecone import PineconeSparseIndexingPipeline
from .qdrant import QdrantSparseIndexingPipeline
from .weaviate import WeaviateBM25IndexingPipeline


__all__ = [
    "ChromaSparseIndexingPipeline",
    "MilvusSparseIndexingPipeline",
    "PineconeSparseIndexingPipeline",
    "QdrantSparseIndexingPipeline",
    "WeaviateBM25IndexingPipeline",
]
