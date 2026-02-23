"""Agentic RAG (Retrieval-Augmented Generation) pipelines for Haystack.

Agentic RAG extends traditional RAG by introducing an LLM-powered agent that
makes autonomous decisions about how to answer queries through multi-step
reasoning and tool selection.

Key Concepts:

Multi-step Reasoning:
    Complex queries are decomposed into sub-queries that the agent
    processes iteratively, gathering information from multiple sources
    before synthesizing a final answer.

LLM-based Routing:
    The agent analyzes each query to determine the optimal tool:
    - retrieval: Vector database lookup for factual questions
    - web_search: External web search for current information
    - calculation: Mathematical or logical reasoning
    - reasoning: Complex multi-hop questions requiring synthesis

Self-reflection and Iteration:
    After generating an initial answer, the agent evaluates quality by:
    1. Assessing completeness (covers all aspects of the query)
    2. Checking factual accuracy against retrieved context
    3. Measuring coherence and clarity
    If quality falls below threshold, the agent refines through additional
    retrieval or rewrites the answer with better context.

This module provides implementations for all supported vector databases:
Pinecone, Weaviate, Chroma, Milvus, and Qdrant.
"""

from vectordb.haystack.agentic_rag.base import BaseAgenticRAGPipeline
from vectordb.haystack.agentic_rag.chroma_agentic_rag import (
    ChromaAgenticRAGPipeline,
)
from vectordb.haystack.agentic_rag.milvus_agentic_rag import (
    MilvusAgenticRAGPipeline,
)
from vectordb.haystack.agentic_rag.pinecone_agentic_rag import (
    PineconeAgenticRAGPipeline,
)
from vectordb.haystack.agentic_rag.qdrant_agentic_rag import (
    QdrantAgenticRAGPipeline,
)
from vectordb.haystack.agentic_rag.weaviate_agentic_rag import (
    WeaviateAgenticRAGPipeline,
)


__all__ = [
    "BaseAgenticRAGPipeline",
    "ChromaAgenticRAGPipeline",
    "MilvusAgenticRAGPipeline",
    "PineconeAgenticRAGPipeline",
    "QdrantAgenticRAGPipeline",
    "WeaviateAgenticRAGPipeline",
]
