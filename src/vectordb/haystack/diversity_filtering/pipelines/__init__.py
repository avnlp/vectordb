"""Diversity filtering pipelines for all supported vector databases.

Provides unified indexing and search pipelines with diversity filtering
across Qdrant, Pinecone, Weaviate, Chroma, and Milvus.

Pipeline Architecture:
- Indexing: Loads datasets via DatasetRegistry, embeds documents using
  sentence-transformers, and indexes into vector database collections.
- Search: Embeds query, retrieves candidates using vector similarity,
  applies diversity filtering (MMR by default), and optionally generates
  RAG responses using the diverse document subset.

Diversity Filtering Integration:
Each search pipeline integrates SentenceTransformersDiversityRanker to apply
Maximum Margin Relevance (MMR) filtering. The pipeline:
1. Retrieves top_k_candidates from vector DB (default 100)
2. Applies MMR with configurable lambda trade-off parameter
3. Returns top_k diverse results covering multiple semantic aspects

Database-Specific Pipelines:
- qdrant_indexing/qdrant_search: Qdrant vector database
- pinecone_indexing/pinecone_search: Pinecone managed service
- weaviate_indexing/weaviate_search: Weaviate hybrid search engine
- chroma_indexing/chroma_search: Chroma local/persistent database
- milvus_indexing/milvus_search: Milvus distributed vector database

All pipelines support optional RAG generation with dataset-specific prompts.
"""

from vectordb.haystack.diversity_filtering.pipelines import (
    chroma_indexing,
    chroma_search,
    milvus_indexing,
    milvus_search,
    pinecone_indexing,
    pinecone_search,
    qdrant_indexing,
    qdrant_search,
    weaviate_indexing,
    weaviate_search,
)


__all__ = [
    "qdrant_indexing",
    "qdrant_search",
    "pinecone_indexing",
    "pinecone_search",
    "weaviate_indexing",
    "weaviate_search",
    "chroma_indexing",
    "chroma_search",
    "milvus_indexing",
    "milvus_search",
]
