"""Sparse indexing implementations for vector databases.

This module provides sparse indexing and search pipelines for keyword-based retrieval.

Example:
    >>> from vectordb.langchain.sparse_indexing import ChromaSparseSearchPipeline
    >>> pipeline = ChromaSparseSearchPipeline("config.yaml")
    >>> results = pipeline.search("keyword query", top_k=10)
"""

from vectordb.langchain.sparse_indexing.indexing import (
    ChromaSparseIndexingPipeline,
    MilvusSparseIndexingPipeline,
    PineconeSparseIndexingPipeline,
    QdrantSparseIndexingPipeline,
    WeaviateSparseIndexingPipeline,
)
from vectordb.langchain.sparse_indexing.search import (
    ChromaSparseSearchPipeline,
    MilvusSparseSearchPipeline,
    PineconeSparseSearchPipeline,
    QdrantSparseSearchPipeline,
    WeaviateSparseSearchPipeline,
)


__all__ = [
    "ChromaSparseIndexingPipeline",
    "MilvusSparseIndexingPipeline",
    "PineconeSparseIndexingPipeline",
    "QdrantSparseIndexingPipeline",
    "WeaviateSparseIndexingPipeline",
    "ChromaSparseSearchPipeline",
    "MilvusSparseSearchPipeline",
    "PineconeSparseSearchPipeline",
    "QdrantSparseSearchPipeline",
    "WeaviateSparseSearchPipeline",
]
