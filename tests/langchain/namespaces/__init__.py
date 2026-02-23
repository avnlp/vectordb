"""Tests for namespace support in LangChain integrations.

This package contains tests for namespace-based vector database implementations
in LangChain. Namespaces enable data isolation between different partitions
sharing the same database infrastructure.

Namespace approaches:
    - Native namespace: Pinecone namespaces (zero overhead)
    - Partition separation: Physical partitions per namespace
    - Collection separation: Separate collections per namespace
    - Payload filtering: Metadata-based namespace isolation

Database implementations tested:
    - Pinecone: Native namespace parameter (zero overhead per namespace)
    - Chroma: Collection-based namespace isolation
    - Milvus: Partition-based namespace separation
    - Qdrant: Payload-based namespace filtering
    - Weaviate: Class-based namespace isolation

Each implementation tests:
    - Namespace data isolation
    - Cross-namespace query prevention
    - Namespace-aware retrieval
    - Resource cleanup per namespace
    - Performance with multiple namespaces
"""
