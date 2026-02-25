"""Query enhancement implementations for vector databases.

This module provides query enhancement pipelines that generate multiple
query variations for improved retrieval quality.

Example:
    >>> from vectordb.langchain.query_enhancement import (
    ...     ChromaQueryEnhancementSearchPipeline,
    ... )
    >>> pipeline = ChromaQueryEnhancementSearchPipeline("config.yaml")
    >>> results = pipeline.search(
    ...     "What is quantum computing?", enhancement_mode="multi_query"
    ... )
"""

from vectordb.langchain.query_enhancement.indexing import (
    ChromaQueryEnhancementIndexingPipeline,
    MilvusQueryEnhancementIndexingPipeline,
    PineconeQueryEnhancementIndexingPipeline,
    QdrantQueryEnhancementIndexingPipeline,
    WeaviateQueryEnhancementIndexingPipeline,
)
from vectordb.langchain.query_enhancement.search import (
    ChromaQueryEnhancementSearchPipeline,
    MilvusQueryEnhancementSearchPipeline,
    PineconeQueryEnhancementSearchPipeline,
    QdrantQueryEnhancementSearchPipeline,
    WeaviateQueryEnhancementSearchPipeline,
)


__all__ = [
    "ChromaQueryEnhancementIndexingPipeline",
    "MilvusQueryEnhancementIndexingPipeline",
    "PineconeQueryEnhancementIndexingPipeline",
    "QdrantQueryEnhancementIndexingPipeline",
    "WeaviateQueryEnhancementIndexingPipeline",
    "ChromaQueryEnhancementSearchPipeline",
    "MilvusQueryEnhancementSearchPipeline",
    "PineconeQueryEnhancementSearchPipeline",
    "QdrantQueryEnhancementSearchPipeline",
    "WeaviateQueryEnhancementSearchPipeline",
]
