"""Utilities for namespace pipelines."""

from .config import load_config, resolve_env_vars
from .data import get_namespace_configs, load_documents_from_config
from .embeddings import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    get_document_embedder,
    get_text_embedder,
    truncate_embeddings,
)
from .timing import Timer


__all__ = [
    "load_config",
    "resolve_env_vars",
    "Timer",
    "get_document_embedder",
    "get_text_embedder",
    "truncate_embeddings",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_EMBEDDING_DIMENSION",
    "load_documents_from_config",
    "get_namespace_configs",
]
