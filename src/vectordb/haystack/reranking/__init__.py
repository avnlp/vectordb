"""Reranking pipelines for vector databases.

Reranking is a two-stage retrieval technique that improves search result quality:

1. **First Stage (Retrieval)**: Fast vector similarity search retrieves a larger
   candidate set (typically 3x the desired results) using bi-encoder embeddings.
2. **Second Stage (Reranking)**: A more accurate cross-encoder model reorders the
   candidates by computing relevance scores for each query-document pair.

Cross-Encoder vs Bi-Encoder:
    - Bi-encoders encode query and documents independently, enabling fast
      approximate nearest neighbor search via vector databases.
    - Cross-encoders process query-document pairs together, capturing richer
      semantic interactions but requiring O(n) forward passes for n candidates.

When to Use Reranking:
    - High precision requirements (e.g., question answering, fact verification)
    - Latency is acceptable (reranking adds 100-500ms overhead)
    - Query-document relevance requires deep semantic understanding
    - Cost trade-off is favorable (expensive reranking on few candidates)

Architecture:
    - Indexing pipelines: Embed documents and store in vector databases
    - Search pipelines: Retrieve candidates with vector search, then rerank

Example:
    >>> from vectordb.haystack.reranking import ChromaRerankingSearchPipeline
    >>> pipeline = ChromaRerankingSearchPipeline("config.yaml")
    >>> results = pipeline.search("machine learning basics", top_k=5)
"""

# Indexing pipelines
from vectordb.haystack.reranking.indexing.chroma import ChromaRerankingIndexingPipeline
from vectordb.haystack.reranking.indexing.milvus import MilvusRerankingIndexingPipeline
from vectordb.haystack.reranking.indexing.pinecone import (
    PineconeRerankingIndexingPipeline,
)
from vectordb.haystack.reranking.indexing.qdrant import QdrantRerankingIndexingPipeline
from vectordb.haystack.reranking.indexing.weaviate import (
    WeaviateRerankingIndexingPipeline,
)

# Search pipelines
from vectordb.haystack.reranking.search.chroma import ChromaRerankingSearchPipeline
from vectordb.haystack.reranking.search.milvus import MilvusRerankingSearchPipeline
from vectordb.haystack.reranking.search.pinecone import PineconeRerankingSearchPipeline
from vectordb.haystack.reranking.search.qdrant import QdrantRerankingSearchPipeline
from vectordb.haystack.reranking.search.weaviate import WeaviateRerankingSearchPipeline


__all__ = [
    # Indexing pipelines
    "ChromaRerankingIndexingPipeline",
    "MilvusRerankingIndexingPipeline",
    "PineconeRerankingIndexingPipeline",
    "QdrantRerankingIndexingPipeline",
    "WeaviateRerankingIndexingPipeline",
    # Search pipelines
    "ChromaRerankingSearchPipeline",
    "MilvusRerankingSearchPipeline",
    "PineconeRerankingSearchPipeline",
    "QdrantRerankingSearchPipeline",
    "WeaviateRerankingSearchPipeline",
]
