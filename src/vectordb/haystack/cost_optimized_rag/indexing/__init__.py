"""Database indexers for cost-optimized RAG pipelines.

Provides unified indexing interfaces across vector databases with
consistent batch processing for cost-efficient embedding generation.

Supported Databases:
    - Qdrant: Cloud-native with payload indexing, quantization support
    - Milvus: Enterprise-scale with partitioning, GPU index support
    - Weaviate: GraphQL interface, modular AI integrations
    - Pinecone: Fully managed serverless, per-query pricing
    - Chroma: Local-first, zero cloud costs for development

Cost Optimization in Indexing:
    - Batch embedding reduces per-document API overhead
    - Quantization (scalar/binary/PQ) cuts storage costs 2-4x
    - Partitioning enables selective loading for multi-tenant scenarios
    - Sparse indexing pre-filter avoids expensive dense embeddings

Lazy Loading Pattern:
    Indexers use lazy loading to avoid import errors for optional dependencies.
    Only import the indexer you need; unused database clients won't be loaded.

Example:
    >>> from vectordb.haystack.cost_optimized_rag.indexing import QdrantIndexer
    >>> indexer = QdrantIndexer("config.yaml")
    >>> indexer.run()  # Batch embed and upsert all documents

Performance Notes:
    - Batch size default: 32 (optimal for most embedding APIs)
    - Indexing is one-time cost; query-time optimizations provide ongoing savings
    - Consider quantization for large collections (>1M documents)
"""


def __getattr__(name: str) -> object:
    """Lazy load indexers to avoid import errors for optional dependencies."""
    if name == "QdrantIndexer":
        from vectordb.haystack.cost_optimized_rag.indexing.qdrant_indexer import (
            QdrantIndexer,
        )

        return QdrantIndexer
    if name == "MilvusIndexer":
        from vectordb.haystack.cost_optimized_rag.indexing.milvus_indexer import (
            MilvusIndexer,
        )

        return MilvusIndexer
    if name == "WeaviateIndexer":
        from vectordb.haystack.cost_optimized_rag.indexing.weaviate_indexer import (
            WeaviateIndexer,
        )

        return WeaviateIndexer
    if name == "PineconeIndexer":
        from vectordb.haystack.cost_optimized_rag.indexing.pinecone_indexer import (
            PineconeIndexer,
        )

        return PineconeIndexer
    if name == "ChromaIndexer":
        from vectordb.haystack.cost_optimized_rag.indexing.chroma_indexer import (
            ChromaIndexer,
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
