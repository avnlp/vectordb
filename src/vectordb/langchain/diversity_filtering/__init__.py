"""LangChain diversity filtering implementations for vector databases.

This module provides diversity filtering pipelines that balance relevance with
diversity in search results. Unlike standard semantic search which returns the
k most similar documents, diversity filtering ensures results cover different
aspects of the query topic, reducing redundancy and improving information coverage.

Diversity Filtering Concepts:

    1. MMR - Maximal Marginal Relevance (default):
       - Balances query relevance with inter-document diversity
       - Uses lambda parameter to control relevance-diversity trade-off
       - Configurable: max_documents, lambda_param (default: 0.5)
       - Best for: Retrieval where both relevance and diversity matter

    2. Clustering-based:
       - Groups retrieved documents into N clusters using embeddings
       - Samples M documents from each cluster
       - Configurable: num_clusters, samples_per_cluster
       - Best for: Ensuring coverage of distinct topic areas

Integration with MMR:
    Diversity filtering uses MMR as its default post-processing strategy for
    retrieved candidates, with clustering available as an alternative strategy
    when explicit topic coverage is preferred.

Architecture:
    indexing/: Document indexing pipelines for all vector stores
        - ChromaDiversityFilteringIndexingPipeline
        - PineconeDiversityFilteringIndexingPipeline
        - MilvusDiversityFilteringIndexingPipeline
        - QdrantDiversityFilteringIndexingPipeline
        - WeaviateDiversityFilteringIndexingPipeline

    search/: Diversity filtering search pipelines for all vector stores
        - ChromaDiversityFilteringSearchPipeline
        - PineconeDiversityFilteringSearchPipeline
        - MilvusDiversityFilteringSearchPipeline
        - QdrantDiversityFilteringSearchPipeline
        - WeaviateDiversityFilteringSearchPipeline

Pipeline Flow:
    Indexing:
        1. Load documents from configured data source
        2. Generate dense embeddings using configured embedder
        3. Create collection/index in vector database
        4. Upsert documents with embeddings

    Search:
        1. Embed query using same embedder as indexing
        2. Over-fetch: Retrieve 3x top_k candidates from vector database
        3. Re-embed: Generate embeddings for retrieved documents
        4. Apply diversity filtering (MMR or clustering method)
        5. Return top_k diverse documents
        6. Optionally generate RAG answer using diverse documents

Supported Vector Databases:
    - Pinecone: Managed cloud service with metadata filtering
    - Chroma: Local embedded database for prototyping
    - Weaviate: Cloud-native with GraphQL interface
    - Milvus: High-performance distributed search
    - Qdrant: Open-source with payload filtering

Configuration Schema:
    Required:
        - Database connection (API keys, URLs, collection names)
        - Embedding model (provider, model name, dimensions)
    Optional:
        - diversity.method: "mmr" or "clustering"
        - diversity.max_documents: Max docs for MMR method
        - diversity.lambda_param: Relevance-diversity trade-off (0.0-1.0)
        - diversity.num_clusters: Number of clusters for clustering method
        - diversity.samples_per_cluster: Docs per cluster
        - rag: Optional LLM configuration for answer generation

Usage Example:
    Indexing:
        >>> from vectordb.langchain.diversity_filtering import (
        ...     ChromaDiversityFilteringIndexingPipeline,
        ... )
        >>> indexer = ChromaDiversityFilteringIndexingPipeline("config.yaml")
        >>> result = indexer.run()
        >>> print(f"Indexed {result['documents_indexed']} documents")

    Search:
        >>> from vectordb.langchain.diversity_filtering import (
        ...     ChromaDiversityFilteringSearchPipeline,
        ... )
        >>> searcher = ChromaDiversityFilteringSearchPipeline("config.yaml")
        >>> results = searcher.search("machine learning applications", top_k=5)
        >>> for doc in results["documents"]:
        ...     print(f"Diverse result: {doc.page_content[:200]}")

Note:
    Diversity filtering requires the same embedding model for both indexing
    and search. The algorithm computes inter-document similarities using
    embeddings, so consistent embedders are essential for accurate diversity
    calculations.
"""

from vectordb.langchain.diversity_filtering.indexing import (
    ChromaDiversityFilteringIndexingPipeline,
    MilvusDiversityFilteringIndexingPipeline,
    PineconeDiversityFilteringIndexingPipeline,
    QdrantDiversityFilteringIndexingPipeline,
    WeaviateDiversityFilteringIndexingPipeline,
)
from vectordb.langchain.diversity_filtering.search import (
    ChromaDiversityFilteringSearchPipeline,
    MilvusDiversityFilteringSearchPipeline,
    PineconeDiversityFilteringSearchPipeline,
    QdrantDiversityFilteringSearchPipeline,
    WeaviateDiversityFilteringSearchPipeline,
)


__all__ = [
    # Indexing pipelines
    "ChromaDiversityFilteringIndexingPipeline",
    "MilvusDiversityFilteringIndexingPipeline",
    "PineconeDiversityFilteringIndexingPipeline",
    "QdrantDiversityFilteringIndexingPipeline",
    "WeaviateDiversityFilteringIndexingPipeline",
    # Search pipelines
    "ChromaDiversityFilteringSearchPipeline",
    "MilvusDiversityFilteringSearchPipeline",
    "PineconeDiversityFilteringSearchPipeline",
    "QdrantDiversityFilteringSearchPipeline",
    "WeaviateDiversityFilteringSearchPipeline",
]
