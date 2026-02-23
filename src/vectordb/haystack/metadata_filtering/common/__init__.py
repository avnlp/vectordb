"""Shared utilities for metadata filtering pipelines.

This module provides common functionality across all vector database
implementations, including configuration loading, data loading, embeddings,
filtering, and RAG generation.
"""

from vectordb.haystack.metadata_filtering.common.config import (
    load_metadata_filtering_config,
)
from vectordb.haystack.metadata_filtering.common.dataloader import (
    load_documents_from_config,
)
from vectordb.haystack.metadata_filtering.common.embeddings import (
    get_document_embedder,
    get_text_embedder,
)
from vectordb.haystack.metadata_filtering.common.filters import (
    filter_spec_to_canonical_dict,
    parse_filter_from_config,
)
from vectordb.haystack.metadata_filtering.common.rag import (
    create_rag_generator,
    generate_answer,
)
from vectordb.haystack.metadata_filtering.common.timer import Timer
from vectordb.haystack.metadata_filtering.common.types import (
    FilterCondition,
    FilteredQueryResult,
    FilterField,
    FilterSpec,
    TimingMetrics,
)


__all__ = [
    "load_metadata_filtering_config",
    "load_documents_from_config",
    "get_document_embedder",
    "get_text_embedder",
    "parse_filter_from_config",
    "filter_spec_to_canonical_dict",
    "create_rag_generator",
    "generate_answer",
    "Timer",
    "FilterField",
    "FilterCondition",
    "FilterSpec",
    "TimingMetrics",
    "FilteredQueryResult",
]
