"""LangChain diversity filtering implementations for vector databases.

This module provides diversity filtering pipelines that balance relevance with
diversity in search results. Unlike standard semantic search which returns the
k most similar documents, diversity filtering ensures results cover different
aspects of the query topic, reducing redundancy and improving information coverage.

Diversity Filtering Concepts:

    1. Threshold-based (default):
       - Iteratively selects documents that are most relevant to the query
       - Filters out documents with similarity > threshold to already-selected docs
       - Configurable: max_documents, similarity_threshold (default: 0.7)
       - Best for: Fine-grained control over diversity vs relevance trade-off

    2. Clustering-based:
       - Groups retrieved documents into N clusters using embeddings
       - Samples M documents from each cluster
       - Configurable: num_clusters, samples_per_cluster
       - Best for: Ensuring coverage of distinct topic areas

Integration with MMR:
    Diversity filtering complements MMR (Maximal Marginal Relevance) by providing
    post-processing diversity algorithms. While MMR balances relevance and diversity
    during retrieval using a lambda parameter, diversity filtering operates on
    retrieved candidates to select diverse subsets using either:
    - Inter-document similarity thresholds
    - Clustering algorithms on document embeddings

    Together, they provide both retrieval-time and post-processing diversity control.

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
        4. Apply diversity filtering (threshold or clustering method)
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
        - diversity.method: "threshold" or "clustering"
        - diversity.max_documents: Max docs for threshold method
        - diversity.similarity_threshold: Similarity cutoff (0.0-1.0)
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
