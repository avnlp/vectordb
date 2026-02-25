"""MMR (Maximal Marginal Relevance) for diversity-aware retrieval.

This module provides MMR pipelines that balance relevance with diversity during
document retrieval. Unlike standard similarity search that returns the most similar
documents, MMR reduces redundancy by penalizing documents that are too similar to
already-selected results, ensuring a diverse set of perspectives.

The MMR formula: MMR = λ * Relevance - (1 - λ) * Redundancy

Key Classes:
    Indexing Pipelines:
        - ChromaMMRIndexingPipeline: MMR indexing for Chroma vector database
        - MilvusMMRIndexingPipeline: MMR indexing for Milvus vector database
        - PineconeMMRIndexingPipeline: MMR indexing for Pinecone vector database
        - QdrantMMRIndexingPipeline: MMR indexing for Qdrant vector database
        - WeaviateMMRIndexingPipeline: MMR indexing for Weaviate vector database

    Search Pipelines:
        - ChromaMMRSearchPipeline: MMR search for Chroma vector database
        - MilvusMMRSearchPipeline: MMR search for Milvus vector database
        - PineconeMMRSearchPipeline: MMR search for Pinecone vector database
        - QdrantMMRSearchPipeline: MMR search for Qdrant vector database
        - WeaviateMMRSearchPipeline: MMR search for Weaviate vector database

Usage:
    >>> from vectordb.langchain.mmr import ChromaMMRSearchPipeline
    >>> pipeline = ChromaMMRSearchPipeline("config.yaml")
    >>> # Fetch 50 candidates, return top 5 diverse results
    >>> results = pipeline.search(
    ...     "artificial intelligence applications", top_k=10, mmr_k=5, lambda_param=0.5
    ... )

The Lambda Parameter:
    - λ = 1.0: Only relevance, no diversity (same as standard similarity search)
    - λ = 0.5: Balanced trade-off between relevance and diversity (default)
    - λ = 0.0: Only diversity, no relevance (selects maximally different documents)
    - Typical range: 0.3 to 0.7 depending on use case
"""

from vectordb.langchain.mmr.indexing import (
    ChromaMMRIndexingPipeline,
    MilvusMMRIndexingPipeline,
    PineconeMMRIndexingPipeline,
    QdrantMMRIndexingPipeline,
    WeaviateMMRIndexingPipeline,
)
from vectordb.langchain.mmr.search import (
    ChromaMMRSearchPipeline,
    MilvusMMRSearchPipeline,
    PineconeMMRSearchPipeline,
    QdrantMMRSearchPipeline,
    WeaviateMMRSearchPipeline,
)


__all__ = [
    "ChromaMMRIndexingPipeline",
    "MilvusMMRIndexingPipeline",
    "PineconeMMRIndexingPipeline",
    "QdrantMMRIndexingPipeline",
    "WeaviateMMRIndexingPipeline",
    "ChromaMMRSearchPipeline",
    "MilvusMMRSearchPipeline",
    "PineconeMMRSearchPipeline",
    "QdrantMMRSearchPipeline",
    "WeaviateMMRSearchPipeline",
]
