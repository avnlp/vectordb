"""Tests for shared vector database wrappers.

This package contains tests for the shared vector database wrapper classes
in ``vectordb.databases``. These wrappers provide a unified API used by
both Haystack and LangChain integrations.

Database wrappers tested:
    - ChromaVectorDB: Local persistent vector database
    - MilvusVectorDB: Cloud-native vector database with high performance
    - PineconeVectorDB: Managed vector database service
    - QdrantVectorDB: High-performance vector search engine
    - WeaviateVectorDB: Graph-vector hybrid database

Each wrapper is tested for:
    - Document storage and retrieval operations
    - Embedding-based similarity search
    - Metadata filtering and querying
    - Collection/index management
    - Error handling and edge cases
"""
