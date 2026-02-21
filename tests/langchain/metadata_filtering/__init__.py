"""Tests for metadata filtering pipelines in LangChain.

This package contains tests for metadata-based filtering implementations in
LangChain. Metadata filtering enables precise document retrieval by combining
vector similarity search with attribute-based constraints.

Filtering capabilities tested:
    - Equality filters: Match exact metadata values
    - Range filters: Numeric comparisons (gt, lt, gte, lte)
    - Set membership: IN/NOT IN operators for multi-value matching
    - Compound filters: AND/OR combinations of filter conditions
    - Nested field access: Filter on nested metadata structures

Database implementations tested:
    - Chroma: Local filtering with where clauses
    - Milvus: Expression-based filtering with partition support
    - Pinecone: Managed service with metadata filter API
    - Qdrant: Payload-based filtering with complex conditions
    - Weaviate: GraphQL where filters with property paths

Each implementation tests:
    - Filter expression construction and validation
    - Combined vector search with metadata constraints
    - Performance impact of complex filter conditions
    - Edge cases for null/missing metadata fields
    - Integration with LangChain retriever filters
"""
