"""Metadata filtering pipelines for vector database integrations.

This module provides indexing and search pipelines for metadata filtering
across multiple vector databases (Milvus, Qdrant, Pinecone, Chroma, Weaviate).

Quick Start:
    # Indexing
    from vectordb.haystack.metadata_filtering.indexing import (
        MilvusMetadataFilteringIndexingPipeline
    )

    pipeline = MilvusMetadataFilteringIndexingPipeline("config.yaml")
    result = pipeline.run()

    # Searching
    from vectordb.haystack.metadata_filtering.search import (
        MilvusMetadataFilteringSearchPipeline
    )

    pipeline = MilvusMetadataFilteringSearchPipeline("config.yaml")
    results = pipeline.search(query="your query")
"""

from vectordb.haystack.metadata_filtering.common import (
    FilterCondition,
    FilteredQueryResult,
    FilterField,
    FilterSpec,
    Timer,
    TimingMetrics,
)
from vectordb.haystack.metadata_filtering.indexing import (
    ChromaMetadataFilteringIndexingPipeline,
    MilvusMetadataFilteringIndexingPipeline,
    PineconeMetadataFilteringIndexingPipeline,
    QdrantMetadataFilteringIndexingPipeline,
    WeaviateMetadataFilteringIndexingPipeline,
)
from vectordb.haystack.metadata_filtering.search import (
    ChromaMetadataFilteringSearchPipeline,
    MilvusMetadataFilteringSearchPipeline,
    PineconeMetadataFilteringSearchPipeline,
    QdrantMetadataFilteringSearchPipeline,
    WeaviateMetadataFilteringSearchPipeline,
)


__all__ = [
    # Common types
    "Timer",
    "FilterField",
    "FilterCondition",
    "FilterSpec",
    "TimingMetrics",
    "FilteredQueryResult",
    # Indexing pipelines
    "MilvusMetadataFilteringIndexingPipeline",
    "QdrantMetadataFilteringIndexingPipeline",
    "PineconeMetadataFilteringIndexingPipeline",
    "ChromaMetadataFilteringIndexingPipeline",
    "WeaviateMetadataFilteringIndexingPipeline",
    # Search pipelines
    "MilvusMetadataFilteringSearchPipeline",
    "QdrantMetadataFilteringSearchPipeline",
    "PineconeMetadataFilteringSearchPipeline",
    "ChromaMetadataFilteringSearchPipeline",
    "WeaviateMetadataFilteringSearchPipeline",
]
