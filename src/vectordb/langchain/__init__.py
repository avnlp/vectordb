"""LangChain integrations for vector database operations.

This module provides native LangChain components and pipelines for building
RAG applications using the LangChain framework. It mirrors the Haystack
integrations with LangChain-specific APIs and abstractions.

Vector Database Support:
    - Pinecone: Managed vector database with hybrid search capabilities
    - Weaviate: Open-source vector search with schema-based filtering
    - Chroma: Embedded database for prototyping and local use
    - Milvus: Scalable vector search with partition support
    - Qdrant: High-performance search with payload indexing

Features:
    - Semantic search with dense embeddings
    - MMR (Maximal Marginal Relevance) diversity ranking
    - Query enhancement with multi-query and HyDE techniques
    - Context compression for token optimization
    - Agentic routing for multi-step reasoning
    - Multi-tenancy with namespace isolation

Usage:
    >>> from langchain_openai import OpenAIEmbeddings
    >>> from vectordb.langchain import ChromaVectorStore
    >>> from vectordb.langchain.mmr import ChromaMmrRetriever
"""
