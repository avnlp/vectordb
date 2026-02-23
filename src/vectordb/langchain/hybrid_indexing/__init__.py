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

The hybrid approach fuses results from both embedding types using database-
specific strategies:

- Reciprocal Rank Fusion (RRF): Used by Qdrant, Milvus, and Pinecone to
  combine ranked results from dense and sparse searches.
- Alpha-weighted blending: Used by Weaviate to balance BM25 and vector
  similarity scores with a configurable weight parameter.

This improves recall for queries where either dense or sparse search alone
might miss relevant documents.

Database Support:
    - Pinecone: Native hybrid search with separate dense/sparse vectors
    - Weaviate: BM25 + vector search fusion
    - Qdrant: Native sparse vector support with RRF fusion
    - Milvus: Native sparse vector support with RRF fusion
    - Chroma: Dense only (sparse stored in metadata)

Example:
    >>> from vectordb.langchain.hybrid_indexing import PineconeHybridIndexingPipeline
    >>> pipeline = PineconeHybridIndexingPipeline("config.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents")
"""
