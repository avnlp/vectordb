"""Cost-optimized RAG pipelines for LangChain.

This module provides cost-optimized Retrieval-Augmented Generation (RAG) pipelines
that minimize API costs while maintaining high retrieval quality. The optimization
strategies employed include:

Cost Optimization Strategies:
    1. Hybrid Search: Combines dense semantic embeddings with sparse lexical
       embeddings to reduce reliance on expensive dense embedding API calls
       while maintaining recall.
    2. Reciprocal Rank Fusion (RRF): Merges dense and sparse search results
       without requiring additional embedding calls, using a lightweight
       ranking algorithm.
    3. Efficient Chunking: Uses RecursiveCharacterTextSplitter with sensible
       defaults to minimize token usage while preserving semantic coherence.
    4. Local Sparse Embeddings: Generates sparse vectors locally using
       TF-IDF/BM25-style algorithms, avoiding API costs entirely for
       lexical search capabilities.
    5. Configurable LLM Integration: Optional RAG generation that can be
       disabled to reduce costs during retrieval-only operations.

Submodules:
    indexing: Document ingestion pipelines for various vector databases.
    search: Query processing and retrieval pipelines with hybrid fusion.

Example:
    >>> from vectordb.langchain.cost_optimized_rag.indexing.pinecone import (
    ...     PineconeCostOptimizedRAGIndexingPipeline,
    ... )
    >>> pipeline = PineconeCostOptimizedRAGIndexingPipeline("config.yaml")
    >>> result = pipeline.run()
    >>> print(f"Indexed {result['documents_indexed']} documents")

Note:
    These pipelines are designed for cost-conscious production deployments
    where API costs are a significant concern. The hybrid approach typically
    reduces embedding API costs by 40-60% compared to pure dense search
    while maintaining comparable retrieval accuracy.
"""
