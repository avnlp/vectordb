"""Hybrid indexing pipelines for LangChain vector databases.

This module provides hybrid search capabilities combining dense and sparse
embeddings for enhanced document retrieval. Hybrid search leverages the
strengths of both approaches:

- Dense embeddings: Capture semantic meaning through neural network-based
  vector representations (e.g., sentence-transformers, OpenAI embeddings).
  Excellent for understanding query intent and semantic similarity.

- Sparse embeddings: Capture exact lexical matches through traditional
  bag-of-words or TF-IDF representations. Critical for matching specific
  keywords, rare terms, and exact phrases.

The hybrid approach fuses results from both embedding types, typically using
a weighted combination (alpha parameter) to balance semantic and lexical
relevance. This improves recall for queries where either dense or sparse
search alone might miss relevant documents.

Database Support:
    - Pinecone: Native hybrid search with separate dense/sparse vectors
    - Weaviate: BM25 + vector search fusion
    - Qdrant: Native sparse vector support
    - Milvus: Native sparse vector support
    - Chroma: Dense only (sparse stored in metadata)

Example:
    >>> from vectordb.langchain.hybrid_indexing import PineconeHybridIndexingPipeline
    >>> pipeline = PineconeHybridIndexingPipeline("config.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents")
"""
