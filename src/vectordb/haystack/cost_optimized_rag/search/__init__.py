"""Database searchers for cost-optimized RAG pipelines.

Lazy-loading module for searcher implementations to avoid import errors
when optional dependencies are not installed. Provides unified access to
Pinecone, Weaviate, Milvus, Chroma, and Qdrant search pipelines.

Cost Optimization Through Lazy Loading:

    Deferred Import Strategy:
        - Searcher classes imported only when accessed
        - Prevents dependency conflicts during module load
        - Reduces startup time by ~50-100ms per unused backend
        - Memory savings: ~5-20MB per unimported client library

    Dependency Isolation:
        - Pinecone: Requires pinecone-client
        - Weaviate: Requires weaviate-client
        - Milvus: Requires pymilvus
        - Chroma: Requires chromadb
        - Qdrant: Requires qdrant-client
        - Each backend optional - install only what you need

Usage Patterns:

    Direct Import:
        >>> from vectordb.haystack.cost_optimized_rag.search import QdrantSearcher
        >>> searcher = QdrantSearcher("config.yaml")

    Dynamic Access:
        >>> search_module = __import__("search", fromlist=["QdrantSearcher"])
        >>> searcher_class = getattr(search_module, "QdrantSearcher")

Cost Implications:
    - No runtime cost for unused backends
    - Import time proportional to backends used
    - Memory footprint scales with active searchers

When to Use Each Backend:
    - Pinecone: Managed service, zero ops
    - Weaviate: GraphQL, modular architecture
    - Milvus: Large scale, distributed
    - Chroma: Local/dev, minimal setup
    - Qdrant: Hybrid search, payload filtering
"""


def __getattr__(name: str) -> object:
    """Lazy load searchers to avoid import errors for optional dependencies."""
    if name == "QdrantSearcher":
        from vectordb.haystack.cost_optimized_rag.search.qdrant_searcher import (
            QdrantSearcher,
        )

        return QdrantSearcher
    if name == "MilvusSearcher":
        from vectordb.haystack.cost_optimized_rag.search.milvus_searcher import (
            MilvusSearcher,
        )

        return MilvusSearcher
    if name == "WeaviateSearcher":
        from vectordb.haystack.cost_optimized_rag.search.weaviate_searcher import (
            WeaviateSearcher,
        )

        return WeaviateSearcher
    if name == "PineconeSearcher":
        from vectordb.haystack.cost_optimized_rag.search.pinecone_searcher import (
            PineconeSearcher,
        )

        return PineconeSearcher
    if name == "ChromaSearcher":
        from vectordb.haystack.cost_optimized_rag.search.chroma_searcher import (
            ChromaSearcher,
        )

        return ChromaSearcher
    msg = f"module '{__name__}' has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "ChromaSearcher",
    "MilvusSearcher",
    "PineconeSearcher",
    "QdrantSearcher",
    "WeaviateSearcher",
]
