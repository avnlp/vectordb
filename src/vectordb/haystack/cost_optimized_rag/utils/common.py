"""Common utilities for cost-optimized RAG pipelines.

Document loading, logging, and result formatting utilities designed
for memory efficiency and minimal overhead. Optimized for batch
processing and cost-conscious deployments.

Document Loading Cost Optimization:

    Dataset Registry Integration:
        - Unified interface for multiple datasets
        - TriviaQA, ARC, PopQA, FactScore, EarningsCall
        - Streaming loading prevents memory spikes
        - Configurable limits for testing/prototyping

    Document Conversion:
        - Haystack Document format standardization
        - Metadata preservation for filtering
        - Batch conversion reduces overhead
        - Lazy evaluation where possible

    Memory Management:
        - limit parameter caps document count
        - Streaming for large collections
        - Clear references for GC
        - No caching unless configured

Result Formatting Strategies:

    Embedding Exclusion:
        - Default: Exclude embeddings (large)
        - Optional: Include when needed
        - Reduces response size 90%+
        - Network and serialization savings

    Metadata Filtering:
        - Score extracted and flattened
        - Other metadata preserved
        - Reduces nesting depth
        - Faster serialization

Logging Efficiency:

    Structured Logging:
        - LoggerFactory with configurable levels
        - Component-specific loggers
        - Appropriate verbosity per environment
        - No debug logging in production
"""

import logging
from typing import Any

from haystack import Document

from vectordb.dataloaders import DataloaderCatalog
from vectordb.haystack.cost_optimized_rag.base.config import RAGConfig
from vectordb.utils.logging import LoggerFactory


def load_documents_from_config(config: RAGConfig) -> list[Document]:
    """Load documents from dataset registry with configurable limits.

    Supports multiple datasets with automatic type detection.
    Applies limit for memory-constrained environments.

    Cost Optimization:
        - Streaming prevents memory spikes
        - Limit caps document count
        - Conversion is batched
        - No intermediate caching

    Supported Datasets:
        - triviaqa → trivia_qa
        - arc → ai2_arc
        - popqa → akariasai/PopQA
        - factscore → dskar/FActScore
        - earnings_calls → lamini/earnings-calls-qa

    Args:
        config: RAGConfig with dataloader.type and optional limit.

    Returns:
        List of Haystack Document objects.
    """
    dataloader_config = config.dataloader
    dataloader_type = dataloader_config.type.lower()

    dataset_name_mapping = {
        "triviaqa": "trivia_qa",
        "arc": "ai2_arc",
        "popqa": "akariasai/PopQA",
        "factscore": "dskar/FActScore",
        "earnings_calls": "lamini/earnings-calls-qa",
    }

    dataset_name = dataloader_config.dataset_name or dataset_name_mapping.get(
        dataloader_type
    )
    split = dataloader_config.split

    loader = DataloaderCatalog.create(
        dataloader_type,
        split=split,
        limit=dataloader_config.limit,
        dataset_id=dataset_name,
    )
    return loader.load().to_haystack()

    # Apply limit for memory control


def create_logger(config: RAGConfig) -> logging.Logger:
    """Create structured logger from configuration.

    Args:
        config: RAGConfig with logging.name and logging.level.

    Returns:
        Configured Logger instance.
    """
    factory = LoggerFactory(
        config.logging.name,
        log_level=getattr(logging, config.logging.level.upper()),
    )
    return factory.get_logger()


def format_search_results(
    documents: list[Document],
    include_embeddings: bool = False,
) -> list[dict[str, Any]]:
    """Format Documents as search results with optional embedding control.

    Default excludes embeddings to minimize response size.
    Embeddings add ~90% to response size (768 dims × 4 bytes).

    Cost Optimization:
        - Exclude embeddings by default (90% size reduction)
        - Flatten score for easy access
        - Filter metadata to remove redundant score

    Args:
        documents: List of Haystack Documents.
        include_embeddings: Whether to include embedding vectors.
            Default False to minimize response size.

    Returns:
        List of result dicts with id, content, score, metadata.
    """
    results = []
    for doc in documents:
        result = {
            "id": doc.id,
            "content": doc.content,
            "score": doc.score if doc.score is not None else doc.meta.get("score", 0.0),
            "metadata": {k: v for k, v in doc.meta.items() if k != "score"},
        }
        if include_embeddings and doc.embedding:
            result["embedding"] = doc.embedding
        results.append(result)
    return results
