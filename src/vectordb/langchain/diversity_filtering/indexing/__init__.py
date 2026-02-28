"""Diversity filtering indexing pipelines for vector databases.

This module provides indexing pipelines optimized for diversity filtering search.
Diversity filtering ensures search results are not only relevant but also cover
different aspects of the query, reducing redundancy and improving information
coverage.

Indexing Requirements:
    The indexing pipeline stores document embeddings that will be used during
    search for both query matching AND inter-document similarity calculations.
    No special metadata is required - diversity is computed at search time.

Pipeline Flow:
    1. Load configuration and validate database settings
    2. Initialize embedder for dense vector generation
    3. Connect to vector database
    4. Load documents from configured data source
    5. Generate embeddings for all documents
    6. Create or recreate collection/index
    7. Upsert documents with embeddings

Supported Vector Databases:
    - ChromaDiversityFilteringIndexingPipeline: Local embedded database
    - PineconeDiversityFilteringIndexingPipeline: Managed cloud service
    - MilvusDiversityFilteringIndexingPipeline: High-performance distributed
    - QdrantDiversityFilteringIndexingPipeline: Open-source with filtering
    - WeaviateDiversityFilteringIndexingPipeline: Cloud-native GraphQL

Configuration:
    Each pipeline requires a YAML configuration specifying:
        - Database connection (API keys, URLs, collection names)
        - Embedding model (provider, model name, dimensions)
        - Optional: recreate flag, namespace settings

Usage Example:
    >>> from vectordb.langchain.diversity_filtering.indexing import (
    ...     PineconeDiversityFilteringIndexingPipeline,
    ... )
    >>> pipeline = PineconeDiversityFilteringIndexingPipeline("config.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents")

Note:
    Use the same embedding model for indexing that will be used for search.
    Diversity filtering computes inter-document similarities using embeddings,
    so consistent embedders are essential for accurate results.
"""

from vectordb.langchain.diversity_filtering.indexing.chroma import (
    ChromaDiversityFilteringIndexingPipeline,
)
from vectordb.langchain.diversity_filtering.indexing.milvus import (
    MilvusDiversityFilteringIndexingPipeline,
)
from vectordb.langchain.diversity_filtering.indexing.pinecone import (
    PineconeDiversityFilteringIndexingPipeline,
)
from vectordb.langchain.diversity_filtering.indexing.qdrant import (
    QdrantDiversityFilteringIndexingPipeline,
)
from vectordb.langchain.diversity_filtering.indexing.weaviate import (
    WeaviateDiversityFilteringIndexingPipeline,
)


__all__ = [
    "ChromaDiversityFilteringIndexingPipeline",
    "MilvusDiversityFilteringIndexingPipeline",
    "PineconeDiversityFilteringIndexingPipeline",
    "QdrantDiversityFilteringIndexingPipeline",
    "WeaviateDiversityFilteringIndexingPipeline",
]
