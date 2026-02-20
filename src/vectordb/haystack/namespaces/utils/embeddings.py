"""Embedding utilities for namespace pipelines."""

from __future__ import annotations

from typing import Any

from haystack import Document
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)


DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EMBEDDING_DIMENSION = 1024


def get_document_embedder(
    config: dict[str, Any],
) -> SentenceTransformersDocumentEmbedder:
    """Create a document embedder from configuration.

    Args:
        config: Configuration dictionary with embedding settings.

    Returns:
        Initialized and warmed-up document embedder.
    """
    embedding_config = config.get("embedding", {})
    model = embedding_config.get("model", DEFAULT_EMBEDDING_MODEL)

    embedder = SentenceTransformersDocumentEmbedder(
        model=model,
        trust_remote_code=True,
    )
    embedder.warm_up()
    return embedder


def get_text_embedder(
    config: dict[str, Any],
) -> SentenceTransformersTextEmbedder:
    """Create a text embedder from configuration.

    Args:
        config: Configuration dictionary with embedding settings.

    Returns:
        Initialized and warmed-up text embedder.
    """
    embedding_config = config.get("embedding", {})
    model = embedding_config.get("model", DEFAULT_EMBEDDING_MODEL)

    embedder = SentenceTransformersTextEmbedder(
        model=model,
        trust_remote_code=True,
    )
    embedder.warm_up()
    return embedder


def truncate_embeddings(
    documents: list[Document],
    output_dimension: int | None,
) -> list[Document]:
    """Truncate document embeddings to specified dimension.

    Args:
        documents: Documents with embeddings.
        output_dimension: Target dimension (None = no truncation).

    Returns:
        Documents with truncated embeddings.
    """
    if output_dimension is None:
        return documents

    for doc in documents:
        if doc.embedding is not None:
            doc.embedding = doc.embedding[:output_dimension]
    return documents
