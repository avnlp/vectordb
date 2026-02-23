"""Agentic RAG pipelines for LangChain.

This module provides agentic RAG (Retrieval-Augmented Generation) pipelines
that combine multi-turn reasoning with vector search, document compression,
answer reflection, and RAG generation.

Agentic RAG Architecture:
    Agentic RAG extends traditional RAG by introducing an intelligent agent
    that can make decisions about how to best answer a query. The agent
    operates in a loop, choosing between three actions:

    1. Search: Retrieve documents from the vector database to gather information
    2. Reflect: Evaluate the current answer quality and identify gaps
    3. Generate: Produce the final answer when sufficient information exists

    This loop continues until either the agent decides to generate an answer
    or a maximum iteration limit is reached.

Key Components:
    - AgenticRouter: LLM-based component that decides the next action
    - ContextCompressor: Filters/summarizes retrieved documents
    - Vector Store: Provides document retrieval capabilities
    - LLM: Generates answers and reflection insights

Pipeline Types:
    Search Pipelines (search/):
        - ChromaAgenticRAGPipeline
        - MilvusAgenticRAGPipeline
        - PineconeAgenticRAGPipeline
        - QdrantAgenticRAGPipeline
        - WeaviateAgenticRAGPipeline

    Indexing Pipelines (indexing/):
        - ChromaAgenticRAGIndexingPipeline
        - MilvusAgenticRAGIndexingPipeline
        - PineconeAgenticRAGIndexingPipeline
        - QdrantAgenticRAGIndexingPipeline
        - WeaviateAgenticRAGIndexingPipeline

Configuration:
    Each pipeline is configured via YAML files specifying:
    - Vector database connection parameters
    - Embedding model configuration
    - LLM settings for routing and generation
    - Compression settings (reranking or LLM extraction)
    - Agentic loop parameters (max iterations, etc.)

Usage:
    >>> from vectordb.langchain.agentic_rag import PineconeAgenticRAGPipeline
    >>> pipeline = PineconeAgenticRAGPipeline("config.yaml")
    >>> result = pipeline.run("What is quantum computing?")
    >>> print(result["final_answer"])
    >>> print(f"Steps: {result['intermediate_steps']}")
"""

from vectordb.langchain.agentic_rag.base import AgenticRAGPipeline
from vectordb.langchain.agentic_rag.indexing import (
    ChromaAgenticRAGIndexingPipeline,
    MilvusAgenticRAGIndexingPipeline,
    PineconeAgenticRAGIndexingPipeline,
    QdrantAgenticRAGIndexingPipeline,
    WeaviateAgenticRAGIndexingPipeline,
)
from vectordb.langchain.agentic_rag.search import (
    ChromaAgenticRAGPipeline,
    MilvusAgenticRAGPipeline,
    PineconeAgenticRAGPipeline,
    QdrantAgenticRAGPipeline,
    WeaviateAgenticRAGPipeline,
)


__all__ = [
    "AgenticRAGPipeline",
    "PineconeAgenticRAGPipeline",
    "WeaviateAgenticRAGPipeline",
    "ChromaAgenticRAGPipeline",
    "MilvusAgenticRAGPipeline",
    "QdrantAgenticRAGPipeline",
    "PineconeAgenticRAGIndexingPipeline",
    "WeaviateAgenticRAGIndexingPipeline",
    "ChromaAgenticRAGIndexingPipeline",
    "MilvusAgenticRAGIndexingPipeline",
    "QdrantAgenticRAGIndexingPipeline",
]
