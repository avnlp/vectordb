"""Utility modules for cost-optimized RAG.

Shared utilities for document loading, result formatting, and prompt
templates. Provides common functionality across all vector database
implementations with cost-conscious design.

Utility Cost Optimization:

    Document Loading:
        - Streaming loading for large datasets
        - Configurable limits prevent memory issues
        - DatasetRegistry caching reduces repeated loads
        - Conversion utilities minimize format overhead

    Result Formatting:
        - Optional embedding exclusion (default)
        - Metadata filtering reduces response size
        - Score normalization for comparison

    Prompt Templates:
        - Concise templates minimize token usage
        - Jinja2 for conditional content
        - No redundant whitespace
        - Optimized for LLM context windows

Design Principles:

    Lazy Evaluation:
        - Documents loaded only when needed
        - Embeddings excluded by default
        - Logging at appropriate levels

    Memory Efficiency:
        - Streaming where possible
        - Batch processing preferred
        - Clear ownership of large objects

    Cost Awareness:
        - Token counts matter
        - Data transfer minimized
        - Compute only when necessary
"""

from vectordb.haystack.cost_optimized_rag.utils.common import (
    create_logger,
    format_search_results,
    load_documents_from_config,
)
from vectordb.haystack.cost_optimized_rag.utils.prompt_templates import (
    COST_OPTIMIZED_RAG_TEMPLATE,
    RAG_ANSWER_TEMPLATE,
    RAG_ANSWER_WITH_SOURCES_TEMPLATE,
)


__all__ = [
    "create_logger",
    "format_search_results",
    "load_documents_from_config",
    "COST_OPTIMIZED_RAG_TEMPLATE",
    "RAG_ANSWER_TEMPLATE",
    "RAG_ANSWER_WITH_SOURCES_TEMPLATE",
]
