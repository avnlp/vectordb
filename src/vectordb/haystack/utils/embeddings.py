"""Embedding factory for Haystack pipeline components.

This module provides factory methods for creating and configuring Haystack's
native SentenceTransformers embedders. It supports both dense and sparse
embedding models with automatic warm-up for production readiness.

Key Features:
    - Document Embedder: Batch embedding of Haystack Documents for indexing
    - Text Embedder: Single-query embedding for retrieval operations
    - Sparse Embedders: SPLADE/token-weight models for hybrid search
    - Automatic Warm-up: Pre-loads models to avoid cold start latency

Configuration:
    The factory reads from a standardized config structure:

    embeddings:
      model: "sentence-transformers/all-MiniLM-L6-v2"
      device: "cuda"  # optional, defaults to auto-detection
      batch_size: 32  # optional, for document embedder

    sparse:
      model: "naver/splade-cocondenser-ensembledistil"

Usage:
    >>> from vectordb.haystack.utils import EmbedderFactory
    >>> config = {"embeddings": {"model": "all-MiniLM-L6-v2"}}
    >>> doc_embedder = EmbedderFactory.create_document_embedder(config)
    >>> text_embedder = EmbedderFactory.create_text_embedder(config)
    >>> dim = EmbedderFactory.get_embedding_dimension(doc_embedder)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from haystack import Document
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)


if TYPE_CHECKING:
    from haystack.components.embedders import (
        SentenceTransformersSparseDocumentEmbedder,
        SentenceTransformersSparseTextEmbedder,
    )


class EmbedderFactory:
    """Factory for creating Haystack embedder components.

    All model names must be specified in full in the config.
    No alias resolution is performed.

    Example config:
        embeddings:
          model: "sentence-transformers/all-MiniLM-L6-v2"
          device: "cpu"
          batch_size: 32
    """

    @classmethod
    def create_document_embedder(
        cls, config: dict[str, Any]
    ) -> SentenceTransformersDocumentEmbedder:
        """Create and warm up a document embedder.

        Args:
            config: Configuration with 'embeddings' section containing
                   'model' (required), 'device' (optional), 'batch_size' (optional).

        Returns:
            Warmed-up SentenceTransformersDocumentEmbedder.

        Raises:
            KeyError: If 'embeddings.model' is not specified.
        """
        embeddings_config = config.get("embeddings", {})
        model = embeddings_config["model"]
        device = embeddings_config.get("device")
        batch_size = embeddings_config.get("batch_size", 32)

        kwargs: dict[str, Any] = {"model": model, "batch_size": batch_size}
        if device is not None:
            kwargs["device"] = device

        embedder = SentenceTransformersDocumentEmbedder(**kwargs)
        embedder.warm_up()
        return embedder

    @classmethod
    def create_text_embedder(
        cls, config: dict[str, Any]
    ) -> SentenceTransformersTextEmbedder:
        """Create and warm up a text embedder.

        Args:
            config: Configuration with 'embeddings' section containing
                   'model' (required), 'device' (optional).

        Returns:
            Warmed-up SentenceTransformersTextEmbedder.

        Raises:
            KeyError: If 'embeddings.model' is not specified.
        """
        embeddings_config = config.get("embeddings", {})
        model = embeddings_config["model"]
        device = embeddings_config.get("device")

        kwargs: dict[str, Any] = {"model": model}
        if device is not None:
            kwargs["device"] = device

        embedder = SentenceTransformersTextEmbedder(**kwargs)
        embedder.warm_up()
        return embedder

    @classmethod
    def create_sparse_document_embedder(
        cls, config: dict[str, Any]
    ) -> "SentenceTransformersSparseDocumentEmbedder":
        """Create and warm up a sparse document embedder.

        Args:
            config: Configuration with 'sparse' section containing 'model' (required).

        Returns:
            Warmed-up SentenceTransformersSparseDocumentEmbedder.

        Raises:
            KeyError: If 'sparse.model' is not specified.
        """
        from haystack.components.embedders import (
            SentenceTransformersSparseDocumentEmbedder,
        )

        sparse_config = config.get("sparse", {})
        model = sparse_config["model"]

        embedder = SentenceTransformersSparseDocumentEmbedder(model=model)
        embedder.warm_up()
        return embedder

    @classmethod
    def create_sparse_text_embedder(
        cls, config: dict[str, Any]
    ) -> "SentenceTransformersSparseTextEmbedder":
        """Create and warm up a sparse text embedder.

        Args:
            config: Configuration with 'sparse' section containing 'model' (required).

        Returns:
            Warmed-up SentenceTransformersSparseTextEmbedder.

        Raises:
            KeyError: If 'sparse.model' is not specified.
        """
        from haystack.components.embedders import SentenceTransformersSparseTextEmbedder

        sparse_config = config.get("sparse", {})
        model = sparse_config["model"]

        embedder = SentenceTransformersSparseTextEmbedder(model=model)
        embedder.warm_up()
        return embedder

    @classmethod
    def get_embedding_dimension(
        cls, embedder: SentenceTransformersDocumentEmbedder
    ) -> int:
        """Get embedding dimension by running a sample document.

        Args:
            embedder: A warmed-up document embedder.

        Returns:
            Dimension of embeddings produced by the model.
        """
        sample = Document(content="dimension check")
        result = embedder.run(documents=[sample])
        embedding = result["documents"][0].embedding
        if embedding is None:
            msg = "Embedder did not produce embeddings"
            raise ValueError(msg)
        return len(embedding)
