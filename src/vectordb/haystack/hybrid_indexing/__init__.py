"""Hybrid search pipelines for vector databases.

This module provides unified hybrid (dense + sparse) search pipelines
for Milvus, Pinecone, Qdrant, Weaviate, and Chroma.

Features:
- Config-driven (YAML) - no CLI scripts
- Native Haystack embedders (no FastEmbed dependency)
- SPLADE sparse embeddings via sentence-transformers
- Per-database implementations for optimal performance
- Shared utilities: config, dataloader, embeddings, fusion, evaluation

Example:
    >>> from vectordb.haystack.hybrid_indexing import (
    ...     MilvusHybridIndexingPipeline,
    ... )
    >>> indexer = MilvusHybridIndexingPipeline(
    ...     config_path="configs/milvus/triviaqa.yaml"
    ... )
    >>> result = indexer.run()

    >>> from vectordb.haystack.hybrid_indexing import (
    ...     MilvusHybridSearchPipeline,
    ... )
    >>> searcher = MilvusHybridSearchPipeline(
    ...     config_path="configs/milvus/triviaqa.yaml"
    ... )
    >>> results = searcher.run(query="What is X?", top_k=10)
"""

from vectordb.haystack.hybrid_indexing.indexing import (
    ChromaHybridIndexingPipeline,
    MilvusHybridIndexingPipeline,
    PineconeHybridIndexingPipeline,
    QdrantHybridIndexingPipeline,
    WeaviateHybridIndexingPipeline,
)
from vectordb.haystack.hybrid_indexing.search import (
    ChromaHybridSearchPipeline,
    MilvusHybridSearchPipeline,
    PineconeHybridSearchPipeline,
    QdrantHybridSearchPipeline,
    WeaviateHybridSearchPipeline,
)


__all__ = [
    # Indexing
    "MilvusHybridIndexingPipeline",
    "PineconeHybridIndexingPipeline",
    "QdrantHybridIndexingPipeline",
    "WeaviateHybridIndexingPipeline",
    "ChromaHybridIndexingPipeline",
    # Search
    "MilvusHybridSearchPipeline",
    "PineconeHybridSearchPipeline",
    "QdrantHybridSearchPipeline",
    "WeaviateHybridSearchPipeline",
    "ChromaHybridSearchPipeline",
]
