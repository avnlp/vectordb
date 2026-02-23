"""Tests for contextual compression pipelines in Haystack.

This package contains tests for contextual compression implementations
in Haystack. Contextual compression reduces token usage and improves
relevance by compressing retrieved documents before LLM generation.

Compression strategies:
    - Abstractive: Summarize context into concise form
    - Extractive: Select most relevant sentences/passages
    - Relevance filtering: Remove low-relevance content

Benefits of compression:
    - Reduced token costs for LLM API calls
    - Improved signal-to-noise ratio in context
    - Faster inference with smaller inputs
    - Better answer quality by removing noise

Database implementations tested:
    - Chroma: Compression with local retrieval
    - Milvus: Tiered compression strategies
    - Pinecone: Metadata-aware compression
    - Qdrant: Payload-based relevance filtering
    - Weaviate: GraphQL context compression

Each implementation tests:
    - Compression quality and fidelity
    - Token reduction metrics
    - Answer quality preservation
    - Latency impact
"""
