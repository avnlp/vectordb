"""Embedder initialization helpers for JSON indexing pipelines."""

from typing import Any

from haystack import Document
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)


def create_document_embedder(
    config: dict[str, Any],
) -> SentenceTransformersDocumentEmbedder:
    """Create a document embedder from configuration.

    Args:
        config: Configuration dictionary containing embeddings settings.

    Returns:
        Initialized SentenceTransformersDocumentEmbedder.
    """
    embeddings_config = config.get("embeddings", {})
    model_name = embeddings_config.get(
        "model", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Support model aliases
    model_aliases = {
        "qwen3": "Qwen/Qwen3-Embedding-0.6B",
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    }
    model_to_use = model_aliases.get(model_name.lower(), model_name)

    embedder = SentenceTransformersDocumentEmbedder(model=model_to_use)
    embedder.warm_up()
    return embedder


def create_text_embedder(
    config: dict[str, Any],
) -> SentenceTransformersTextEmbedder:
    """Create a text embedder from configuration.

    Args:
        config: Configuration dictionary containing embeddings settings.

    Returns:
        Initialized SentenceTransformersTextEmbedder.
    """
    embeddings_config = config.get("embeddings", {})
    model_name = embeddings_config.get(
        "model", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Support model aliases
    model_aliases = {
        "qwen3": "Qwen/Qwen3-Embedding-0.6B",
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    }
    model_to_use = model_aliases.get(model_name.lower(), model_name)

    embedder = SentenceTransformersTextEmbedder(model=model_to_use)
    embedder.warm_up()
    return embedder


def get_embedding_dimension(
    embedder: SentenceTransformersDocumentEmbedder | SentenceTransformersTextEmbedder,
) -> int:
    """Get embedding dimension from embedder.

    Args:
        embedder: The embedder to get dimension from.

    Returns:
        The dimension of embeddings produced by the embedder.
    """
    if isinstance(embedder, SentenceTransformersDocumentEmbedder):
        sample_doc = Document(content="sample text for dimension check")
        embedded = embedder.run(documents=[sample_doc])["documents"]
        return len(embedded[0].embedding)

    # For text embedder
    result = embedder.run(text="sample text for dimension check")
    return len(result["embedding"])


def embed_documents(
    documents: list[Document],
    embedder: SentenceTransformersDocumentEmbedder,
) -> list[Document]:
    """Generate embeddings for documents.

    Args:
        documents: List of Haystack Documents to embed.
        embedder: The document embedder to use.

    Returns:
        List of Documents with embeddings attached.
    """
    return embedder.run(documents=documents)["documents"]
