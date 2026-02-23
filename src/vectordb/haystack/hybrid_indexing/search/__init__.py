"""Hybrid search pipelines for vector databases.

This module provides unified search pipelines for hybrid (dense + sparse)
retrieval across multiple vector database backends.

Search Pipeline Architecture:
    Each search pipeline follows the same pattern:
    1. Load and validate configuration
    2. Initialize dense and optional sparse embedders
    3. Embed the query using both embedding types
    4. Execute hybrid search (native or manual fusion)
    5. Return ranked results with metadata

Query Embedding Process:
    - Dense embeddings capture the semantic intent of the query
    - Sparse embeddings capture specific keywords and terms
    - Both representations are computed in parallel when available
    - Sparse embeddings are optional; dense-only search is supported

Search Strategies by Backend:
    - Milvus: Native hybrid_search() with RRF ranking
    - Pinecone: Native hybrid_search() with alpha weighting
    - Qdrant: Native search() with search_type="hybrid"
    - Weaviate: Native hybrid_search() combining BM25 + vector
    - Chroma: Manual RRF fusion of separate dense and sparse searches

Result Format:
    All pipelines return a dictionary with:
    - documents: List of ranked Document objects with scores
    - query: Original query string
    - db: Database identifier string

Example:
    >>> from vectordb.haystack.hybrid_indexing.search import (
    ...     MilvusHybridSearchPipeline,
    ... )
    >>> searcher = MilvusHybridSearchPipeline(
    ...     config_path="configs/milvus/triviaqa.yaml"
    ... )
    >>> results = searcher.run(query="What is machine learning?", top_k=10)
    >>> for doc in results["documents"]:
    ...     print(f"{doc.score:.3f}: {doc.content[:100]}...")
"""

from vectordb.haystack.hybrid_indexing.search.chroma import (
    ChromaHybridSearchPipeline,
)
from vectordb.haystack.hybrid_indexing.search.milvus import (
    MilvusHybridSearchPipeline,
)
from vectordb.haystack.hybrid_indexing.search.pinecone import (
    PineconeHybridSearchPipeline,
)
from vectordb.haystack.hybrid_indexing.search.qdrant import (
    QdrantHybridSearchPipeline,
)
from vectordb.haystack.hybrid_indexing.search.weaviate import (
    WeaviateHybridSearchPipeline,
)


__all__ = [
    "ChromaHybridSearchPipeline",
    "MilvusHybridSearchPipeline",
    "PineconeHybridSearchPipeline",
    "QdrantHybridSearchPipeline",
    "WeaviateHybridSearchPipeline",
]
