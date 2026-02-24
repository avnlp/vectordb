"""Indexing pipelines for cost-optimized RAG.

This module contains document indexing pipelines that prepare data for
cost-optimized retrieval. Each pipeline implements the same core strategy:

Indexing Strategy:
    1. Document Loading: Ingest documents from various sources (files, URLs,
       databases) using configurable data loaders.
    2. Intelligent Chunking: Split documents into chunks using
       RecursiveCharacterTextSplitter with configurable size/overlap to
       balance context preservation with token efficiency.
    3. Dual Embedding Generation:
       - Dense embeddings via API (semantic understanding)
       - Sparse embeddings locally (lexical matching, zero API cost)
    4. Hybrid Index Storage: Store both vector types for efficient hybrid
       search at query time.

Supported Vector Databases:
    - Pinecone: Cloud-native with native sparse vector support
    - Qdrant: Open-source with sparse vector support
    - Weaviate: Hybrid search with BM25 integration
    - Milvus: High-performance with sparse vector support
    - Chroma: Local-first with metadata-based sparse storage

Cost Optimization:
    The indexing phase generates sparse embeddings locally using efficient
    TF-IDF/BM25 algorithms, completely avoiding API costs for lexical
    representation. Only dense embeddings incur API costs, reducing total
    indexing costs by approximately 50% compared to dual-API embedding.

Example:
    >>> from vectordb.langchain.cost_optimized_rag.indexing.pinecone import (
    ...     PineconeCostOptimizedRAGIndexingPipeline,
    ... )
    >>> pipeline = PineconeCostOptimizedRAGIndexingPipeline(
    ...     {
    ...         "pinecone": {"api_key": "...", "index_name": "my-index"},
    ...         "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
    ...         "chunking": {"chunk_size": 1000, "chunk_overlap": 200},
    ...     }
    ... )
    >>> result = pipeline.run()
"""
