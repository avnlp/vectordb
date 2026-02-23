"""LangChain components for advanced search and RAG features.

This module provides LangChain-native implementations of advanced RAG components
that can be composed into end-to-end retrieval pipelines.

Components:
    - QueryEnhancer: Multi-query generation, HyDE, and step-back prompting
    - ContextCompressor: Abstractive and extractive summarization
    - AgenticRouter: LLM-based routing for retrieval/reflection/generation

These components work with any LangChain-compatible vector store and LLM.

Usage:
    >>> from vectordb.langchain.components import QueryEnhancer, ContextCompressor
    >>> enhancer = QueryEnhancer(model="llama-3.3-70b-versatile")
    >>> variations = enhancer.enhance_query("What is quantum computing?", "multi_query")
"""

from vectordb.langchain.components.agentic_router import AgenticRouter
from vectordb.langchain.components.context_compressor import ContextCompressor
from vectordb.langchain.components.query_enhancer import QueryEnhancer


__all__ = [
    "QueryEnhancer",
    "ContextCompressor",
    "AgenticRouter",
]
