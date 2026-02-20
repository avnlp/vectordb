"""Shared components for Advanced RAG pipelines.

Provides reusable, composable components for query enhancement, result merging,
context compression, evaluation, and agentic routing. These components are designed
to work with any Haystack-compatible vector database and LLM.

Components Provided:
    - QueryEnhancer: Multi-query generation, HyDE, and step-back prompting
    - ResultMerger: Reciprocal Rank Fusion (RRF) and weighted fusion
    - ContextCompressor: Query-focused document summarization
    - DeepEvalEvaluator: Contextual precision, recall, faithfulness metrics
    - AgenticRouter: LLM-based tool selection and self-reflection

Note:
    Embedders and rerankers use native Haystack components directly.
    Use vectordb.haystack.utils for factory classes that create these components
    with proper configuration from YAML files.

Example:
    >>> from vectordb.haystack.components import QueryEnhancer, ResultMerger
    >>> enhancer = QueryEnhancer(model="llama-3.3-70b-versatile")
    >>> queries = enhancer.enhance_query("neural networks", "multi_query")
"""

from vectordb.haystack.components.agentic_router import AgenticRouter
from vectordb.haystack.components.context_compressor import ContextCompressor
from vectordb.haystack.components.evaluators import DeepEvalEvaluator
from vectordb.haystack.components.query_enhancer import QueryEnhancer
from vectordb.haystack.components.result_merger import ResultMerger


__all__ = [
    "AgenticRouter",
    "ContextCompressor",
    "DeepEvalEvaluator",
    "QueryEnhancer",
    "ResultMerger",
]
