"""Embedder creation utilities using native Haystack components."""

from __future__ import annotations

from typing import Any

from haystack import Document
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)


DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def create_document_embedder(
    config: dict[str, Any] | None = None,
) -> SentenceTransformersDocumentEmbedder:
    """Create document embedder from config.

    Args:
        config: Configuration dictionary with 'embedding' section.

    Returns:
        Configured SentenceTransformersDocumentEmbedder instance.
    """
    if config is None:
        config = {}

    embedding_config = config.get("embedding", {})
    model = embedding_config.get("model", DEFAULT_EMBEDDING_MODEL)
    trust_remote_code = embedding_config.get("trust_remote_code", True)
    batch_size = embedding_config.get("batch_size", 32)

    return SentenceTransformersDocumentEmbedder(
        model=model,
        trust_remote_code=trust_remote_code,
        batch_size=batch_size,
    )


def create_text_embedder(
    config: dict[str, Any] | None = None,
) -> SentenceTransformersTextEmbedder:
    """Create text embedder from config.

    Args:
        config: Configuration dictionary with 'embedding' section.

    Returns:
        Configured SentenceTransformersTextEmbedder instance.
    """
    if config is None:
        config = {}

    embedding_config = config.get("embedding", {})
    model = embedding_config.get("model", DEFAULT_EMBEDDING_MODEL)
    trust_remote_code = embedding_config.get("trust_remote_code", True)
    prefix = embedding_config.get("query_prefix", "")

    return SentenceTransformersTextEmbedder(
        model=model,
        trust_remote_code=trust_remote_code,
        prefix=prefix,
    )


def truncate_embeddings(
    documents: list[Document],
    output_dimension: int | None,
) -> list[Document]:
    """Truncate embeddings to specified dimension (MRL support).

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
