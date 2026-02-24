"""Indexing pipelines for query-enhanced RAG with multi-strategy support.

Provides unified indexing interfaces across vector databases with consistent
document ingestion for query enhancement retrieval strategies.

Supported Databases:
    - Qdrant: Cloud-native with payload indexing, optimized for hybrid search
    - Milvus: Enterprise-scale with GPU acceleration, partitioning support
    - Weaviate: GraphQL-native, modular AI integrations, semantic caching
    - Pinecone: Fully managed serverless with metadata filtering
    - Chroma: Local-first development, zero cloud costs for prototyping

Query Enhancement Strategies:
    - Multi-Query: Generates N query variations to capture different phrasings
      and aspects of the original question, improving recall
    - HyDE (Hypothetical Document Embeddings): Generates hypothetical answer
      documents that are embedded and used for similarity search
    - Step-Back: Abstracts specific questions to higher-level conceptual
      queries that retrieve broader, more contextual information

Lazy Loading Pattern:
    Indexers use lazy loading to avoid import errors for optional dependencies.
    Only import the indexer you need; unused database clients won't be loaded.

Example:
    >>> from vectordb.haystack.query_enhancement.indexing import QdrantIndexer
    >>> indexer = QdrantIndexer("config.yaml")
    >>> indexer.run()  # Index documents for query-enhanced retrieval

Performance Notes:
    - Documents embedded once and cached for reuse across query strategies
    - Indexing is pipeline setup cost; query enhancement happens at search time
    - Batch processing optimizes embedding API usage for large document sets
"""


def __getattr__(name: str) -> object:
    """Lazy load indexers to avoid import errors for optional dependencies."""
    if name == "QdrantIndexer":
        from vectordb.haystack.query_enhancement.indexing.qdrant import (
            QdrantQueryEnhancementIndexingPipeline as QdrantIndexer,
        )

        return QdrantIndexer
    if name == "MilvusIndexer":
        from vectordb.haystack.query_enhancement.indexing.milvus import (
            MilvusQueryEnhancementIndexingPipeline as MilvusIndexer,
        )

        return MilvusIndexer
    if name == "WeaviateIndexer":
        from vectordb.haystack.query_enhancement.indexing.weaviate import (
            WeaviateQueryEnhancementIndexingPipeline as WeaviateIndexer,
        )

        return WeaviateIndexer
    if name == "PineconeIndexer":
        from vectordb.haystack.query_enhancement.indexing.pinecone import (
            PineconeQueryEnhancementIndexingPipeline as PineconeIndexer,
        )

        return PineconeIndexer
    if name == "ChromaIndexer":
        from vectordb.haystack.query_enhancement.indexing.chroma import (
            ChromaQueryEnhancementIndexingPipeline as ChromaIndexer,
        )

        return ChromaIndexer
    msg = f"module {name!r} not found"
    raise AttributeError(msg)


__all__ = [
    "ChromaIndexer",
    "MilvusIndexer",
    "PineconeIndexer",
    "QdrantIndexer",
    "WeaviateIndexer",
]
