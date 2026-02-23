"""Embedder initialization utilities for metadata filtering pipelines.

Provides functions to create and warm up document and text embedders.
"""

from typing import Any

from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)


__all__ = ["get_document_embedder", "get_text_embedder"]


def get_document_embedder(
    config: dict[str, Any],
) -> SentenceTransformersDocumentEmbedder:
    """Create and warm up a document embedder from config.

    Configuration should include an 'embeddings' section with:
    - model: HuggingFace model ID (default: sentence-transformers/all-MiniLM-L6-v2)

    Args:
        config: Configuration dictionary.

    Returns:
        Initialized and warmed up SentenceTransformersDocumentEmbedder.
    """
    embeddings_config = config.get("embeddings", {})
    model_name = embeddings_config.get(
        "model",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    embedder = SentenceTransformersDocumentEmbedder(model=model_name)
    embedder.warm_up()

    return embedder


def get_text_embedder(
    config: dict[str, Any],
) -> SentenceTransformersTextEmbedder:
    """Create and warm up a text embedder from config.

    Configuration should include an 'embeddings' section with:
    - model: HuggingFace model ID (default: sentence-transformers/all-MiniLM-L6-v2)

    Args:
        config: Configuration dictionary.

    Returns:
        Initialized and warmed up SentenceTransformersTextEmbedder.
    """
    embeddings_config = config.get("embeddings", {})
    model_name = embeddings_config.get(
        "model",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    embedder = SentenceTransformersTextEmbedder(model=model_name)
    embedder.warm_up()

    return embedder
