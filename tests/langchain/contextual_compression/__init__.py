"""Tests for contextual compression pipelines (LangChain).

This package contains tests for contextual compression implementations in
LangChain. Contextual compression reduces token usage and noise by filtering
or compressing retrieved documents based on their relevance to the query.

Compression methods tested:
    - LLM-based compression: Uses LLM to extract relevant portions
    - Embedding-based compression: Filters by query-document similarity
    - Reranking compression: Reorders and truncates based on relevance scores
    - Token-based compression: Truncates documents to token limits

Database implementations tested:
    - Chroma: Local compression with LangChain Document objects
    - Milvus: Cloud-native compression with partition support
    - Pinecone: Managed service compression with metadata filtering
    - Qdrant: On-premise compression with payload filtering
    - Weaviate: Graph-vector hybrid compression

Each implementation tests:
    - Base retriever functionality
    - Compression pipeline integration
    - Token reduction effectiveness
    - Relevance preservation after compression
    - Integration with LangChain chains and LCEL
"""
