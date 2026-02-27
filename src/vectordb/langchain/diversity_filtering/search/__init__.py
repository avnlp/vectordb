"""Diversity filtering search pipelines for vector databases.

This module provides search pipelines with diversity-aware retrieval capabilities.
Diversity filtering post-processes search results to ensure the returned documents
cover different aspects of the query, reducing redundancy and improving
information coverage.

Diversity Filtering Methods:
    1. Threshold-based (default):
       - Iteratively selects documents most relevant to the query
       - Filters out documents with similarity > threshold to already-selected docs
       - Configurable: max_documents, similarity_threshold (default: 0.7)
       - Best for: Fine-grained control over diversity vs relevance trade-off

    2. Clustering-based:
       - Groups retrieved documents into N clusters using embeddings
       - Samples M documents from each cluster
       - Configurable: num_clusters, samples_per_cluster
       - Best for: Ensuring coverage of distinct topic areas

Pipeline Flow:
    1. Query Embedding: Convert query text to dense vector
    2. Over-fetch: Retrieve 3x top_k candidates from vector database
    3. Re-embedding: Generate embeddings for retrieved documents
    4. Diversity Filtering: Apply threshold or clustering method
    5. Limit: Return top_k diverse documents
    6. Optional RAG: Generate answer using diverse documents

Supported Vector Databases:
    - ChromaDiversityFilteringSearchPipeline: Local embedded database
    - PineconeDiversityFilteringSearchPipeline: Managed cloud service
    - MilvusDiversityFilteringSearchPipeline: High-performance distributed
    - QdrantDiversityFilteringSearchPipeline: Open-source with filtering
    - WeaviateDiversityFilteringSearchPipeline: Cloud-native GraphQL

Configuration:
    Each pipeline requires a YAML configuration specifying:
        - Database connection (API keys, URLs, collection names)
        - Embedding model (provider, model name, dimensions)
        - Diversity method and parameters
        - Optional RAG LLM configuration for answer generation

Usage Example:
    >>> from vectordb.langchain.diversity_filtering.search import (
    ...     PineconeDiversityFilteringSearchPipeline,
    ... )
    >>> pipeline = PineconeDiversityFilteringSearchPipeline("config.yaml")
    >>> results = pipeline.search("machine learning applications", top_k=5)
    >>> for doc in results["documents"]:
    ...     print(f"Diverse result: {doc.page_content[:200]}")

Note:
    Diversity filtering requires the same embedding model used during indexing.
    The algorithm computes inter-document similarities using embeddings retrieved
    at search time, so consistent embedders are essential for accurate results.
"""

from vectordb.langchain.diversity_filtering.search.chroma import (
    ChromaDiversityFilteringSearchPipeline,
)
from vectordb.langchain.diversity_filtering.search.milvus import (
    MilvusDiversityFilteringSearchPipeline,
)
from vectordb.langchain.diversity_filtering.search.pinecone import (
    PineconeDiversityFilteringSearchPipeline,
)
from vectordb.langchain.diversity_filtering.search.qdrant import (
    QdrantDiversityFilteringSearchPipeline,
)
from vectordb.langchain.diversity_filtering.search.weaviate import (
    WeaviateDiversityFilteringSearchPipeline,
)


__all__ = [
    "ChromaDiversityFilteringSearchPipeline",
    "MilvusDiversityFilteringSearchPipeline",
    "PineconeDiversityFilteringSearchPipeline",
    "QdrantDiversityFilteringSearchPipeline",
    "WeaviateDiversityFilteringSearchPipeline",
]
