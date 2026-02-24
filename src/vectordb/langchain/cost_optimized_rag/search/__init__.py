"""Search pipelines for cost-optimized RAG.

This module contains query processing and retrieval pipelines that implement
cost-optimized hybrid search across various vector databases.

Search Strategy:
    1. Query Embedding: Generate both dense and sparse embeddings for the
       query. Dense embeddings use API for semantic understanding, sparse
       embeddings generated locally at zero cost.
    2. Parallel Retrieval: Execute dense and sparse searches concurrently
       against the vector database to minimize latency.
    3. Reciprocal Rank Fusion (RRF): Merge results from both searches using
       RRF algorithm with configurable parameters. This requires no additional
       API calls and runs locally in milliseconds.
    4. Result Deduplication: Remove duplicate documents that appear in both
       search results, preserving highest-ranked instance.
    5. Optional RAG Generation: If LLM is configured, generate contextual
       answer using retrieved documents.

Cost Benefits:
    - Single dense embedding per query (vs. multiple for re-ranking)
    - Zero-cost sparse query embedding (local algorithm)
    - Zero-cost fusion algorithm (local computation)
    - Optional LLM generation (can be disabled for retrieval-only)

Configuration:
    Search behavior is controlled via the 'search' configuration section:
    - rrf_k: RRF fusion parameter (default: 60)
    - alpha: Hybrid weight for Weaviate (default: 0.5)

Example:
    >>> from vectordb.langchain.cost_optimized_rag.search.pinecone import (
    ...     PineconeCostOptimizedRAGSearchPipeline,
    ... )
    >>> pipeline = PineconeCostOptimizedRAGSearchPipeline("config.yaml")
    >>> result = pipeline.search("What is machine learning?", top_k=5)
    >>> for doc in result["documents"]:
    ...     print(doc["text"][:100])
"""
