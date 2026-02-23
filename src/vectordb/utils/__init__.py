"""Utility modules for vector database operations.

This module provides shared utilities used across all vector database integrations,
ensuring consistent behavior for document conversion, evaluation, logging, and metadata
handling. These utilities abstract away database-specific implementation details.

Utilities Provided:
    - Document Converters: Transform documents between Haystack/LangChain formats
    - Configuration: YAML config loading, environment variable resolution
    - Evaluation: Retrieval metrics (Precision, Recall, MRR, NDCG, Hit Rate)
    - ID Management: Consistent document ID generation and extraction
    - Logging: Structured logger factory with environment-based configuration
    - Output Types: Structured containers for retrieval results
    - Scope/Filter: Multi-tenancy and namespace isolation utilities
    - Sparse Embeddings: Format normalization for hybrid search

Usage:
    >>> from vectordb.utils import ChromaDocumentConverter, compute_mrr
    >>> from vectordb.utils.config import load_config
"""

from vectordb.utils.chroma_document_converter import ChromaDocumentConverter
from vectordb.utils.config import (
    DATASET_LIMITS,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODEL_ALIASES,
    get_dataset_limits,
    load_config,
    resolve_embedding_model,
    resolve_env_vars,
    setup_logger,
)
from vectordb.utils.evaluation import (
    EvaluationResult,
    QueryResult,
    RetrievalMetrics,
    compute_hit_rate,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    evaluate_retrieval,
)
from vectordb.utils.ids import coerce_id, get_doc_id, set_doc_id
from vectordb.utils.logging import LoggerFactory
from vectordb.utils.output import (
    PipelineOutput,
    RetrievalOutput,
    RetrievedDocument,
)
from vectordb.utils.pinecone_document_converter import PineconeDocumentConverter
from vectordb.utils.scope import (
    build_scope_filter_expr,
    inject_scope_to_filter,
    inject_scope_to_metadata,
)
from vectordb.utils.sparse import (
    get_doc_sparse_embedding,
    normalize_sparse,
    to_milvus_sparse,
    to_pinecone_sparse,
    to_qdrant_sparse,
)
from vectordb.utils.weaviate_document_converter import WeaviateDocumentConverter


__all__ = [
    # Document Converters
    "ChromaDocumentConverter",
    "PineconeDocumentConverter",
    "WeaviateDocumentConverter",
    # Config
    "DATASET_LIMITS",
    "DEFAULT_EMBEDDING_MODEL",
    "EMBEDDING_MODEL_ALIASES",
    "get_dataset_limits",
    "load_config",
    "resolve_embedding_model",
    "resolve_env_vars",
    "setup_logger",
    # Evaluation
    "EvaluationResult",
    "QueryResult",
    "RetrievalMetrics",
    "compute_hit_rate",
    "compute_mrr",
    "compute_ndcg_at_k",
    "compute_precision_at_k",
    "compute_recall_at_k",
    "evaluate_retrieval",
    # IDs
    "coerce_id",
    "get_doc_id",
    "set_doc_id",
    # Logging
    "LoggerFactory",
    # Output
    "PipelineOutput",
    "RetrievalOutput",
    "RetrievedDocument",
    # Scope
    "build_scope_filter_expr",
    "inject_scope_to_filter",
    "inject_scope_to_metadata",
    # Sparse
    "get_doc_sparse_embedding",
    "normalize_sparse",
    "to_milvus_sparse",
    "to_pinecone_sparse",
    "to_qdrant_sparse",
]
