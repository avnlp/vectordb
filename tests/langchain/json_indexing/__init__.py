"""Tests for JSON indexing pipelines in LangChain.

This package contains tests for JSON document indexing implementations in
LangChain. JSON indexing enables structured data storage and retrieval with
embedded vector representations for semantic search capabilities.

JSON indexing features tested:
    - Nested field extraction: Flatten nested JSON into searchable content
    - Schema validation: Validate documents against expected JSON schemas
    - Field-level indexing: Index specific JSON fields with metadata
    - Dynamic mapping: Auto-detect and index JSON structures

Database implementations tested:
    - Chroma: Local JSON document storage with metadata filtering
    - Milvus: Cloud-native JSON indexing with dynamic fields
    - Pinecone: Managed service with JSON metadata support
    - Qdrant: On-premise JSON payload indexing
    - Weaviate: Graph-vector JSON property storage

Each implementation tests:
    - JSON document ingestion and parsing
    - Field extraction and embedding generation
    - Metadata-based filtering on JSON fields
    - Search result reconstruction with original JSON structure
    - Integration with LangChain document loaders
"""
