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


def _create_embedder(config: dict[str, Any], embedder_class: type) -> Any:
    model = config.get("embeddings", {}).get(
        "model", "sentence-transformers/all-MiniLM-L6-v2"
    )
    model = MODEL_ALIASES.get(model.lower(), model)
    embedder = embedder_class(model=model)
    embedder.warm_up()
    return embedder


def create_text_embedder(config: dict[str, Any]) -> SentenceTransformersTextEmbedder:
    """Create a text embedder from config."""
    return _create_embedder(config, SentenceTransformersTextEmbedder)


def create_document_embedder(
    config: dict[str, Any],
) -> SentenceTransformersDocumentEmbedder:
    """Create a document embedder from config."""
    return _create_embedder(config, SentenceTransformersDocumentEmbedder)
