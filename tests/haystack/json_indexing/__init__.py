"""Tests for JSON indexing pipelines in Haystack.

This package contains tests for JSON document indexing implementations
in Haystack. These tests verify the ability to index, query, and retrieve
structured JSON data from vector databases.

JSON indexing capabilities:
    - Nested field extraction and flattening
    - Schema inference and validation
    - Type-aware embedding generation
    - Metadata preservation from JSON structure

Supported JSON patterns:
    - Flat key-value objects
    - Nested objects with arbitrary depth
    - Arrays and lists
    - Mixed types and null values

Database implementations tested:
    - Chroma: JSON documents with metadata filtering
    - Milvus: Dynamic field support for JSON
    - Pinecone: Metadata-based JSON storage
    - Qdrant: Payload-based JSON indexing
    - Weaviate: GraphQL JSON querying

Each implementation tests:
    - JSON document ingestion
    - Nested field querying
    - Schema evolution handling
    - Filter operations on JSON fields
"""
