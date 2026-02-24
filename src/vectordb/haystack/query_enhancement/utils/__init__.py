"""Shared utilities for query enhancement pipelines.

Provides common components used across query enhancement indexing and search
pipelines: configuration management, embedding utilities, LLM integration,
result fusion, and data loading.

Core Components:
    - Config: YAML configuration loading and validation with TypedDict schemas
    - Embeddings: Document and query embedder factory with batch processing
    - LLM: Groq API integration for query enhancement generation
    - Fusion: Reciprocal Rank Fusion (RRF) and weighted result combination
    - DataLoader: Dataset loaders for TriviaQA, ARC, PopQA, FactScore

Query Enhancement Support:
    Utilities support three enhancement strategies through prompt templates:
    - Multi-Query: Generates query variations using MULTI_QUERY_PROMPT template
    - HyDE: Creates hypothetical documents using HYDE_PROMPT template
    - Step-Back: Abstracts questions using STEP_BACK_PROMPT template

Lazy Loading Pattern:
    Utilities use lazy loading to avoid import errors for optional dependencies.
    Components are imported only when explicitly accessed.

Example:
    >>> from vectordb.haystack.query_enhancement.utils import load_config
    >>> config = load_config("config.yaml")
    >>> from vectordb.haystack.query_enhancement.utils import rrf_fusion_many
    >>> fused = rrf_fusion_many([results1, results2, results3], k=60)

Dependencies:
    - haystack: Core pipeline components and Document type
    - groq: LLM inference for query enhancement (optional)
    - PyYAML: Configuration file parsing
"""

# Mapping of export names to (module_path, attribute_name) for lazy loading
_IMPORT_MAP: dict[str, tuple[str, str]] = {
    "load_config": ("vectordb.haystack.query_enhancement.utils.config", "load_config"),
    "validate_config": (
        "vectordb.haystack.query_enhancement.utils.config",
        "validate_config",
    ),
    "create_document_embedder": (
        "vectordb.haystack.query_enhancement.utils.embeddings",
        "create_document_embedder",
    ),
    "create_query_embedder": (
        "vectordb.haystack.query_enhancement.utils.embeddings",
        "create_query_embedder",
    ),
    "create_groq_generator": (
        "vectordb.haystack.query_enhancement.utils.llm",
        "create_groq_generator",
    ),
    "rrf_fusion_many": (
        "vectordb.haystack.query_enhancement.utils.fusion",
        "rrf_fusion_many",
    ),
    "deduplicate_by_content": (
        "vectordb.haystack.query_enhancement.utils.fusion",
        "deduplicate_by_content",
    ),
    "stable_doc_id": (
        "vectordb.haystack.query_enhancement.utils.fusion",
        "stable_doc_id",
    ),
}


def __getattr__(name: str) -> object:
    """Lazy load utilities to avoid import errors for optional dependencies.

    Args:
        name: Name of the utility to import.

    Returns:
        The requested utility function or class.

    Raises:
        AttributeError: If the requested name is not a valid export.
    """
    if name not in _IMPORT_MAP:
        msg = f"module '{__name__}' has no attribute {name!r}"
        raise AttributeError(msg)

    module_path, attr_name = _IMPORT_MAP[name]
    module = __import__(module_path, fromlist=[attr_name])
    return getattr(module, attr_name)


__all__ = [
    "create_document_embedder",
    "create_groq_generator",
    "create_query_embedder",
    "deduplicate_by_content",
    "load_config",
    "rrf_fusion_many",
    "stable_doc_id",
    "validate_config",
]
