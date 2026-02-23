"""VectorDB: Vector database integrations for NLP and RAG applications.

This package provides unified interfaces for working with multiple vector databases
including Pinecone, Weaviate, Chroma, Milvus, and Qdrant. It offers both Haystack and
LangChain integrations with support for 13+ advanced RAG features.

Features:
    - Dense and hybrid search with sparse embeddings
    - Maximal Marginal Relevance (MMR) for diversity-aware retrieval
    - Parent document retrieval for hierarchical chunking
    - Cost-optimized RAG pipelines
    - Context compression and query enhancement
    - Agentic routing for multi-step reasoning
    - Comprehensive evaluation metrics

Architecture:
    - vectordb/databases/: Shared vector database wrappers
      (Chroma, Milvus, Pinecone, Qdrant, Weaviate)
    - vectordb/haystack/: Haystack pipeline components and integrations
    - vectordb/langchain/: LangChain native integrations
    - vectordb/dataloaders/: Dataset loaders for RAG evaluation
    - vectordb/utils/: Shared utilities for document conversion,
      logging, config, and metrics

Example:
    >>> from vectordb.dataloaders import DataloaderCatalog
    >>> from vectordb.utils import RetrievalMetrics
"""
