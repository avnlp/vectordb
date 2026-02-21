"""Metadata filtering search pipelines for vector databases.

This module provides search pipelines that support metadata-based filtering
during retrieval operations. Metadata filtering enables precise document
selection using document attributes like source, category, date, or custom
metadata fields.

Search Pipeline Architecture:
    1. Query Embedding: Convert search query to dense vector using embedder
    2. Metadata Filtering: Apply filters to constrain search scope
    3. Similarity Search: Query vector database for top-k similar documents
    4. Result Formatting: Convert database results to LangChain Documents

Metadata Filter Syntax:
    Different vector databases support different filter syntax:
        - Pinecone: Metadata filters with exact match operators
        - Chroma: Where clause with comparison operators
        - Weaviate: GraphQL-style where filters
        - Qdrant: Payload-based filtering
        - Milvus: Expression-based filtering

    Common filter patterns:
        - Equality: {"category": "technology"}
        - Range: {"year": {"$gte": 2020}}
        - IN clause: {"author": {"$in": ["Alice", "Bob"]}}
        - Text search: {"content": {"$contains": "keyword"}}

Supported Vector Databases:
    - PineconeMetadataFilteringSearchPipeline
    - ChromaMetadataFilteringSearchPipeline
    - WeaviateMetadataFilteringSearchPipeline
    - QdrantMetadataFilteringSearchPipeline
    - MilvusMetadataFilteringSearchPipeline

Performance Considerations:
    - Metadata filtering reduces the search space, improving both relevance
      and latency for targeted queries
    - Index metadata fields that are frequently used in filters
    - Avoid filters with high-cardinality values for best performance

Usage:
    >>> from vectordb.langchain.metadata_filtering.search.chroma import (
    ...     ChromaMetadataFilteringSearchPipeline,
    ... )
    >>> pipeline = ChromaMetadataFilteringSearchPipeline("config.yaml")
    >>> results = pipeline.search(
    ...     query="machine learning",
    ...     top_k=10,
    ...     filters={"category": "technology", "year": 2023},
    ... )
    >>> print(f"Found {len(results['documents'])} filtered documents")

See Also:
    vectordb.langchain.metadata_filtering.indexing: Indexing pipelines
"""

__all__ = []
