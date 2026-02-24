"""Search pipelines with query enhancement for improved retrieval quality.

Provides unified search interfaces across vector databases with integrated
query enhancement strategies: Multi-Query, HyDE, and Step-Back prompting.

Supported Databases:
    - Qdrant: Fast hybrid search with payload filtering, ideal for multi-query
    - Milvus: GPU-accelerated similarity search for large-scale deployments
    - Weaviate: Native hybrid search combining BM25 and vector similarity
    - Pinecone: Metadata-filtered vector search with automatic indexing
    - Chroma: Local development with persistent embeddings storage

Query Enhancement Strategies:
    - Multi-Query: Expands single query into N variations, retrieves for each,
      then fuses results using Reciprocal Rank Fusion (RRF) for diversity
    - HyDE (Hypothetical Document Embeddings): Uses LLM to generate hypothetical
      answer documents, embeds them, retrieves by vector similarity to answers
    - Step-Back: Reformulates specific queries into abstract concepts,
      retrieves broader context, then filters to specific details

Lazy Loading Pattern:
    Searchers use lazy loading to avoid import errors for optional dependencies.
    Only import the searcher you need; unused database clients won't be loaded.

Example:
    >>> from vectordb.haystack.query_enhancement.search import QdrantSearcher
    >>> searcher = QdrantSearcher("config.yaml")
    >>> results = searcher.run("What are the benefits of RAG?")

Performance Notes:
    - Multi-Query: N x embedding cost + LLM generation cost
    - HyDE: M x LLM generation + embedding cost (no query embedding needed)
    - Step-Back: 1 x LLM abstraction + 1 x standard retrieval
"""


def __getattr__(name: str) -> object:
    """Lazy load searchers to avoid import errors for optional dependencies."""
    if name == "QdrantSearcher":
        from vectordb.haystack.query_enhancement.search.qdrant import (
            QdrantQueryEnhancementSearchPipeline as QdrantSearcher,
        )

        return QdrantSearcher
    if name == "MilvusSearcher":
        from vectordb.haystack.query_enhancement.search.milvus import (
            MilvusQueryEnhancementSearchPipeline as MilvusSearcher,
        )

        return MilvusSearcher
    if name == "WeaviateSearcher":
        from vectordb.haystack.query_enhancement.search.weaviate import (
            WeaviateQueryEnhancementSearchPipeline as WeaviateSearcher,
        )

        return WeaviateSearcher
    if name == "PineconeSearcher":
        from vectordb.haystack.query_enhancement.search.pinecone import (
            PineconeQueryEnhancementSearchPipeline as PineconeSearcher,
        )

        return PineconeSearcher
    if name == "ChromaSearcher":
        from vectordb.haystack.query_enhancement.search.chroma import (
            ChromaQueryEnhancementSearchPipeline as ChromaSearcher,
        )

        return ChromaSearcher
    msg = f"module {name!r} not found"
    raise AttributeError(msg)


__all__ = [
    "ChromaSearcher",
    "MilvusSearcher",
    "PineconeSearcher",
    "QdrantSearcher",
    "WeaviateSearcher",
]
