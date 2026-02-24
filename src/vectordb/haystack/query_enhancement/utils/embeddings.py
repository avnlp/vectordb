"""Embedder initialization for query enhancement pipelines."""

from typing import Any

from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)


# Model alias mapping
MODEL_ALIASES = {
    "qwen3": "Qwen/Qwen3-Embedding-0.6B",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}


def create_text_embedder(config: dict[str, Any]) -> SentenceTransformersTextEmbedder:
    """Create and warm up a text embedder.

    Args:
        config: Configuration dictionary with 'embeddings' section.

    Returns:
        Warmed-up SentenceTransformersTextEmbedder.
    """
    model = config.get("embeddings", {}).get(
        "model", "sentence-transformers/all-MiniLM-L6-v2"
    )
    model = MODEL_ALIASES.get(model.lower(), model)

    embedder = SentenceTransformersTextEmbedder(model=model)
    embedder.warm_up()
    return embedder


def create_document_embedder(
    config: dict[str, Any],
) -> SentenceTransformersDocumentEmbedder:
    """Create and warm up a document embedder.

    Args:
        config: Configuration dictionary with 'embeddings' section.

    Returns:
        Warmed-up SentenceTransformersDocumentEmbedder.
    """
    model = config.get("embeddings", {}).get(
        "model", "sentence-transformers/all-MiniLM-L6-v2"
    )
    model = MODEL_ALIASES.get(model.lower(), model)

    embedder = SentenceTransformersDocumentEmbedder(model=model)
    embedder.warm_up()
    return embedder
