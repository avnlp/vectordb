"""Utility modules for Haystack vector database pipelines.

This module consolidates helper classes and utilities used across all Haystack
pipeline implementations. These utilities handle common operations like configuration
loading, embedding generation, reranking, and result fusion.

Utility Classes:
    ConfigLoader: YAML configuration parsing with environment variable resolution.
        Handles loading configuration from file paths or dictionaries and validates
        required keys for each vector database backend.

    DataloaderCatalog: Document loading from standard evaluation datasets including
        TriviaQA, ARC, PopQA, FactScore, and EarningsCall. Creates dataloaders
        and converts to Haystack Documents.

    DiversificationHelper: Semantic diversification algorithms for reducing redundancy
        in retrieved document sets. Implements greedy selection based on embedding
        similarity.

    EmbedderFactory: Embedding model initialization using Haystack's native
        SentenceTransformers components. Creates document and text embedders
        with automatic warm-up.

    DocumentFilter: Metadata-based filtering utilities for post-retrieval document
        filtering when database-level filtering is insufficient.

    ResultMerger: Multi-source result fusion algorithms including Reciprocal Rank
        Fusion (RRF) and weighted merging. Essential for hybrid search pipelines.

    RAGHelper: LLM initialization and prompt formatting for RAG generation. Uses
        Haystack's OpenAIGenerator with Groq-compatible endpoints.

    RerankerFactory: Cross-encoder reranking using Haystack's native components.
        Provides factory methods for creating rerankers from configuration.

Usage:
    >>> from vectordb.haystack.utils import EmbedderFactory, RerankerFactory
    >>> config = {"embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2"}}
    >>> embedder = EmbedderFactory.create_document_embedder(config)
    >>> text_embedder = EmbedderFactory.create_text_embedder(config)
"""

from vectordb.haystack.utils.diversification import DiversificationHelper
from vectordb.haystack.utils.embeddings import EmbedderFactory
from vectordb.haystack.utils.filters import DocumentFilter
from vectordb.haystack.utils.fusion import ResultMerger
from vectordb.haystack.utils.rag import RAGHelper
from vectordb.haystack.utils.reranker import RerankerFactory
from vectordb.utils.config_loader import ConfigLoader


__all__ = [
    "ConfigLoader",
    "DiversificationHelper",
    "EmbedderFactory",
    "DocumentFilter",
    "RAGHelper",
    "RerankerFactory",
    "ResultMerger",
]
