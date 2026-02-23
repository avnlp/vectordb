"""Tests for agentic RAG pipelines in Haystack.

This package contains tests for agentic RAG (Retrieval-Augmented Generation)
implementations in Haystack. Agentic RAG extends traditional RAG with
autonomous decision-making capabilities for improved answer quality.

Agentic capabilities:
    - Tool selection: Choose appropriate retrieval tools per query
    - Self-reflection: Evaluate and improve answer quality
    - Multi-step reasoning: Chain retrieval and reasoning steps
    - Error recovery: Handle retrieval failures gracefully

Agent components:
    - AgenticRouter: Routes queries to appropriate tools
    - SelfReflector: Evaluates and refines generated answers
    - ToolExecutor: Executes selected retrieval tools
    - AnswerSynthesizer: Combines retrieved context into answers

Database implementations tested:
    - Chroma: Local agentic retrieval
    - Milvus: Cloud-native agent integration
    - Pinecone: Managed service with agent routing
    - Qdrant: High-performance agentic search
    - Weaviate: Graph-vector agentic retrieval

Each implementation tests:
    - Tool selection accuracy
    - Self-reflection loop behavior
    - Answer quality improvement
    - Error handling and fallbacks
"""
