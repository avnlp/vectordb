"""Metadata filtering indexing pipelines for vector databases.

This module provides indexing pipelines that prepare document collections
for metadata-filtered retrieval. These pipelines handle document loading,
embedding generation, and vector store indexing with metadata support.

Indexing Pipeline Architecture:
    All indexing pipelines follow a consistent pattern:
        1. Document Loading: Load documents from configured data source
        2. Metadata Extraction: Extract or preserve document metadata
        3. Embedding Generation: Generate dense vectors for documents
        4. Vector Store Indexing: Store documents with metadata in database

Metadata Handling:
    - Metadata is preserved and upserted alongside document content
    - Some databases index metadata fields for filtering (Pinecone, Weaviate)
    - Others require explicit filter configuration (Chroma, Qdrant, Milvus)
    - Nested metadata is flattened or stored as JSON depending on database

Supported Vector Databases:
    - PineconeMetadataFilteringIndexingPipeline
    - ChromaMetadataFilteringIndexingPipeline
    - WeaviateMetadataFilteringIndexingPipeline
    - QdrantMetadataFilteringIndexingPipeline
    - MilvusMetadataFilteringIndexingPipeline

Configuration:
    Each pipeline requires standard configuration plus optional metadata settings:
        metadata:
          # Fields to index for filtering (database-specific)
          indexed_fields: ["category", "year", "author"]

          # Default metadata values
          defaults:
            source: "default_source"
            version: 1

Pipeline Consistency:
    All pipelines share identical interfaces:
        - __init__(config_or_path): Initialize from dict or YAML file path
        - run() -> dict: Execute indexing and return statistics

Usage:
    >>> from vectordb.langchain.metadata_filtering.indexing.chroma import (
    ...     ChromaMetadataFilteringIndexingPipeline,
    ... )
    >>> pipeline = ChromaMetadataFilteringIndexingPipeline("config.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents with metadata")

See Also:
    vectordb.langchain.metadata_filtering.search: Search pipelines with filtering
"""

__all__ = []
