"""Tests for metadata filtering pipelines in Haystack.

This package contains tests for metadata filtering implementations
in Haystack. These tests verify the ability to filter vector search
results based on document metadata attributes.

Metadata filtering capabilities:
    - Exact match filtering (equals, not equals)
    - Range filtering (greater than, less than, between)
    - List membership (in, not in)
    - String operations (contains, starts with, ends with)
    - Logical operators (AND, OR, NOT)

Filter expression formats:
    - Dictionary-based filters for simple databases
    - SQL-like expressions for advanced queries
    - JSONPath for nested metadata
    - Custom DSL for complex conditions

Database implementations tested:
    - Chroma: Metadata dictionary filtering
    - Milvus: Boolean expression filters
    - Pinecone: Metadata key-value filtering
    - Qdrant: Payload filtering with conditions
    - Weaviate: GraphQL where filters

Each implementation tests:
    - Single and compound filters
    - Performance with large metadata sets
    - Filter pushdown optimization
    - Edge cases and special characters
"""
