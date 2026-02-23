"""Tests for agentic RAG pipelines in LangChain.

This package contains tests for agent-based Retrieval-Augmented Generation
implementations in LangChain. Agentic RAG uses autonomous decision-making
to determine retrieval strategies, answer quality, and when to stop iterating.

Agentic RAG components tested:
    - AgenticRouter: LLM-based action routing (search/reflect/generate)
    - ContextCompressor: Document compression for token efficiency
    - QueryEnhancer: Query transformation and expansion
    - ReflectionAgent: Answer quality verification

Agentic loop workflow:
    1. Route: Determine next action based on current state
    2. Search: Retrieve relevant documents from vector store
    3. Compress: Reduce context to most relevant portions
    4. Reflect: Verify answer quality and completeness
    5. Generate: Produce final answer when ready

Database implementations tested:
    - Chroma: Local agentic RAG with iterative retrieval
    - Milvus: Cloud-native agentic pipelines
    - Pinecone: Managed service agentic retrieval
    - Qdrant: On-premise agentic RAG with filtering
    - Weaviate: Graph-vector agentic search

Each implementation tests:
    - Pipeline initialization and configuration
    - Iterative retrieval with action routing
    - Context compression modes (reranking, LLM extraction)
    - Maximum iteration handling and fallback behavior
    - RAG generation with compressed context
"""
