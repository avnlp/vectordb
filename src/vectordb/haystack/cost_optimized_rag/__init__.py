"""Cost-optimized RAG pipelines for production vector databases.

This module provides Retrieval-Augmented Generation (RAG) pipelines optimized for
cost efficiency when working with managed vector database services that charge by
request unit (RU) or compute unit (CU).

Cost Optimization Strategies:

    1. Tiered Retrieval Architecture
       - Stage 1: BM25 sparse retrieval filters irrelevant documents without
         embedding computation cost
       - Stage 2: Dense vector search operates on reduced candidate set
       - Reduces embedding API calls by 60-80% compared to pure dense retrieval

    2. Token Usage Optimization
       - Fixed-size chunking with configurable overlap prevents oversized
         context windows that increase LLM costs
       - Prompt templates optimized for conciseness to minimize input tokens
       - Aggressive top-k reduction (default: 10 documents vs. typical 50-100)

    3. Caching and Reuse
       - Embedding cache prevents redundant embedding computation for
         frequently accessed documents
       - Query result cache for common queries reduces database calls
       - Batch embedding operations amortize API call overhead

    4. Performance vs Cost Trade-offs
       - Configurable reranking with smaller models (MiniLM-L-6-v2) vs
         larger models (ms-marco-MiniLM-L-12-v2) based on accuracy requirements
       - Quantization options (scalar/binary/PQ) reduce storage costs
       - Partitioning enables selective index loading for multi-tenant scenarios

    5. Database-Specific Optimizations
       - Pinecone: Namespace isolation for cost attribution
       - Qdrant: Payload indexing for efficient metadata filtering
       - Milvus: IVF_FLAT index balances recall with query cost
       - Weaviate: Class-level vectorization control
       - Chroma: Local storage eliminates cloud egress fees

Supported Datasets:
    - TriviaQA: General knowledge QA (reading comprehension)
    - ARC: Science reasoning QA (multiple choice)
    - PopQA: Factoid entity QA (knowledge probing)
    - FactScore: Fact verification (precision evaluation)
    - Earnings Calls: Financial transcript QA (domain-specific)

Usage:
    >>> from vectordb.haystack.cost_optimized_rag import (
    ...     QdrantSearchPipeline,
    ... )
    >>> pipeline = QdrantSearchPipeline("config.yaml")
    >>> result = pipeline.search(query, top_k=5)  # Minimize tokens sent to LLM
    >>> rag_result = pipeline.search_with_rag(query, top_k=3)  # Ultra-compact context
"""
