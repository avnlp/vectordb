"""Embedding utilities for LangChain vector database pipelines.

This module provides helper methods for creating and using HuggingFace embedding
models within LangChain pipelines. It abstracts the embedding model initialization
and inference operations commonly needed across all vector database integrations.

Embedding Models:
    The module uses HuggingFaceEmbeddings from langchain-huggingface, which supports
    any sentence-transformers model. Common choices include:

    - sentence-transformers/all-MiniLM-L6-v2: Fast, 384-dimensional embeddings
    - sentence-transformers/all-mpnet-base-v2: Higher quality, 768-dimensional
    - BAAI/bge-small-en-v1.5: Strong performance on retrieval benchmarks
    - intfloat/e5-small-v2: Instruction-tuned for query/document distinction

Configuration:
    Embedding configuration is specified in YAML under the 'embeddings' key:

    .. code-block:: yaml

        embeddings:
          model: sentence-transformers/all-MiniLM-L6-v2
          device: cpu  # or cuda
          batch_size: 32

Device Selection:
    Use 'cuda' for GPU acceleration when available. GPU embedding is significantly
    faster for batch operations (10-50x speedup). CPU is suitable for development
    or when GPU is unavailable.

Usage:
    >>> from vectordb.langchain.utils import EmbedderHelper
    >>> config = {"embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2"}}
    >>> embedder = EmbedderHelper.create_embedder(config)
    >>> query_embedding = EmbedderHelper.embed_query(embedder, "machine learning")
    >>> docs, embeddings = EmbedderHelper.embed_documents(embedder, documents)
"""

from typing import Any

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


class EmbedderHelper:
    """Helper class for HuggingFace embedding model operations.

    This class provides static methods for creating embedding model instances
    and generating embeddings for both queries and documents. It encapsulates
    the configuration parsing and model initialization logic shared across
    all pipeline implementations.
    """

    @classmethod
    def create_embedder(cls, config: dict[str, Any]) -> HuggingFaceEmbeddings:
        """Create HuggingFaceEmbeddings from config.

        Args:
            config: Configuration dictionary with embeddings section.

        Returns:
            HuggingFaceEmbeddings instance.
        """
        embeddings_config = config.get("embeddings", {})
        model = embeddings_config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        device = embeddings_config.get("device", "cpu")
        batch_size = embeddings_config.get("batch_size", 32)

        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": device},
            encode_kwargs={"batch_size": batch_size},
        )

    @classmethod
    def embed_documents(
        cls, embedder: HuggingFaceEmbeddings, documents: list[Document]
    ) -> tuple[list[Document], list[list[float]]]:
        """Embed documents and return with embeddings.

        Args:
            embedder: HuggingFaceEmbeddings instance.
            documents: List of LangChain Document objects.

        Returns:
            Tuple of (documents, embeddings).
        """
        texts = [doc.page_content for doc in documents]
        embeddings = embedder.embed_documents(texts)
        return documents, embeddings

    @classmethod
    def embed_query(cls, embedder: HuggingFaceEmbeddings, query: str) -> list[float]:
        """Embed a single query.

        Args:
            embedder: HuggingFaceEmbeddings instance.
            query: Query text.

        Returns:
            Query embedding.
        """
        return embedder.embed_query(query)
