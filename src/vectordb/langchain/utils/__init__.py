"""Utility modules for LangChain vector database pipelines.

This module consolidates helper classes and utilities used across all LangChain
pipeline implementations. These utilities handle common operations like configuration
loading, embedding generation, reranking, and result fusion.

Utility Classes:
    ConfigLoader: YAML configuration parsing with schema validation. Handles loading
        configuration from file paths or dictionaries and validates required keys
        for each vector database backend.

    DiversificationHelper: Semantic diversification algorithms for reducing
    redundancy in retrieved document sets. Implements both greedy selection
    and clustering-based approaches for selecting diverse documents.

    EmbedderHelper: Embedding model initialization and inference for both documents
        and queries. Wraps HuggingFace sentence-transformers with batch processing
        support.

    DocumentFilter: Metadata-based filtering with support for various operators
        (equals, contains, gt/lt, in/not_in) and nested JSON path queries. Enables
        post-retrieval filtering when database-level filtering is insufficient.

    ResultMerger: Multi-source result fusion algorithms including Reciprocal Rank
        Fusion (RRF) and weighted merging. Essential for hybrid search pipelines
        that combine dense and sparse retrieval results.

    MMRHelper: Maximal Marginal Relevance implementation for diversity-aware reranking.
        Balances relevance with diversity using configurable lambda parameter.

    RAGHelper: LLM initialization and prompt formatting for RAG generation. Handles
        context assembly and answer generation using Groq-hosted LLMs.

    RerankerHelper: Cross-encoder reranking using HuggingFace models. Provides
        methods for reranking with and without score preservation.

    SparseEmbedder: SPLADE-based sparse embedding generation for hybrid search.
        Generates token-weight dictionaries for lexical retrieval.

Usage:
    >>> from vectordb.langchain.utils import EmbedderHelper, RerankerHelper
    >>> config = {"embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2"}}
    >>> embedder = EmbedderHelper.create_embedder(config)
    >>> query_embedding = EmbedderHelper.embed_query(embedder, "What is ML?")
"""

from vectordb.langchain.utils.config import ConfigLoader
from vectordb.langchain.utils.diversification import DiversificationHelper
from vectordb.langchain.utils.document_converter import HaystackToLangchainConverter
from vectordb.langchain.utils.embeddings import EmbedderHelper
from vectordb.langchain.utils.filters import DocumentFilter
from vectordb.langchain.utils.fusion import ResultMerger
from vectordb.langchain.utils.mmr import MMRHelper
from vectordb.langchain.utils.rag import RAGHelper
from vectordb.langchain.utils.reranker import RerankerHelper
from vectordb.langchain.utils.sparse_embeddings import SparseEmbedder


__all__ = [
    "ConfigLoader",
    "DiversificationHelper",
    "HaystackToLangchainConverter",
    "EmbedderHelper",
    "DocumentFilter",
    "ResultMerger",
    "MMRHelper",
    "RAGHelper",
    "RerankerHelper",
    "SparseEmbedder",
]
