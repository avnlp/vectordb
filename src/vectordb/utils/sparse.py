"""Sparse embedding format normalization and conversion utilities.

This module provides bidirectional conversion between sparse embedding formats used
by different vector databases and the Haystack SparseEmbedding standard. Sparse
embeddings enable hybrid search by combining lexical (BM25-style) and semantic
(dense) retrieval signals.

Supported Formats:
    - Haystack: SparseEmbedding(indices=[...], values=[...])
    - Milvus: Dict[int, float] mapping indices to values
    - Pinecone: {"indices": [...], "values": [...]} dictionary
    - Qdrant: SparseVector object from qdrant_client

Key Functions:
    - normalize_sparse: Convert any format to Haystack SparseEmbedding
    - to_milvus_sparse: Convert to Milvus {index: value} format
    - to_pinecone_sparse: Convert to Pinecone sparse_values format
    - to_qdrant_sparse: Convert to Qdrant SparseVector object
    - get_doc_sparse_embedding: Extract sparse embedding from Document

Design Notes:
    The normalize_sparse function serves as a universal ingress point, accepting
    any supported format and producing a consistent Haystack SparseEmbedding. The
    to_* functions handle egress to specific backends.

Usage:
    >>> from vectordb.utils.sparse import normalize_sparse, to_pinecone_sparse
    >>> sparse = normalize_sparse({1: 0.5, 5: 0.8})  # From Milvus format
    >>> pinecone_fmt = to_pinecone_sparse(sparse)  # To Pinecone format
"""

from typing import Any, Dict, List, Optional, Union, cast

from haystack.dataclasses import SparseEmbedding


"""Type alias for sparse embedding inputs that normalize_sparse can handle.

Supports Haystack SparseEmbedding objects, Milvus dict format {int: float},
Pinecone dict format {"indices": [...], "values": [...]}, or None.
"""
SparseInput = Union[SparseEmbedding, Dict[int, float], Dict[str, List], None]


def normalize_sparse(sparse: SparseInput) -> Optional[SparseEmbedding]:
    """Normalize any sparse format to Haystack SparseEmbedding.

    Accepts:
        - SparseEmbedding object (passthrough)
        - Dict[int, float] (Milvus format: {index: value})
        - Dict[str, List] (Pinecone format: {"indices": [...], "values": [...]})
        - None (passthrough)

    Returns:
        SparseEmbedding or None.

    Example:
        >>> normalize_sparse({1: 0.5, 5: 0.8})
        SparseEmbedding(indices=[1, 5], values=[0.5, 0.8])
    """
    if sparse is None:
        return None

    if isinstance(sparse, SparseEmbedding):
        return sparse

    if isinstance(sparse, dict):
        if "indices" in sparse and "values" in sparse:
            # Pinecone format
            pinecone_sparse = cast(Dict[str, List], sparse)
            return SparseEmbedding(
                indices=list(pinecone_sparse["indices"]),
                values=list(pinecone_sparse["values"]),
            )
        # Milvus format: {index: value}
        indices = list(sparse.keys())
        values = list(sparse.values())
        return SparseEmbedding(indices=indices, values=values)

    raise TypeError(f"Unsupported sparse embedding format: {type(sparse)}")


def to_milvus_sparse(sparse: SparseEmbedding) -> Dict[int, float]:
    """Convert SparseEmbedding to Milvus format: {index: value}.

    Args:
        sparse: Haystack SparseEmbedding.

    Returns:
        Dict mapping indices to values.
    """
    return dict(zip(sparse.indices, sparse.values))


def to_pinecone_sparse(sparse: SparseEmbedding) -> Dict[str, List]:
    """Convert SparseEmbedding to Pinecone sparse_values format.

    Args:
        sparse: Haystack SparseEmbedding.

    Returns:
        Dict with 'indices' and 'values' keys.
    """
    return {
        "indices": list(sparse.indices),
        "values": list(sparse.values),
    }


def to_qdrant_sparse(sparse: SparseEmbedding) -> Any:
    """Convert SparseEmbedding to Qdrant SparseVector format.

    Args:
        sparse: Haystack SparseEmbedding.

    Returns:
        Qdrant SparseVector object.

    Note:
        Imports qdrant_client lazily to avoid hard dependency.
    """
    from qdrant_client.http.models import SparseVector

    return SparseVector(
        indices=list(sparse.indices),
        values=list(sparse.values),
    )


def get_doc_sparse_embedding(
    doc: Any,
    fallback_meta_key: str = "sparse_embedding",
) -> Optional[SparseEmbedding]:
    """Extract sparse embedding from Document, checking standard and legacy locations.

    Args:
        doc: Haystack Document.
        fallback_meta_key: Legacy meta key to check if doc.sparse_embedding is None.

    Returns:
        SparseEmbedding or None.
    """
    # Prefer standard location
    if hasattr(doc, "sparse_embedding") and doc.sparse_embedding is not None:
        return doc.sparse_embedding

    # Fallback to meta for backward compatibility
    if hasattr(doc, "meta") and doc.meta and fallback_meta_key in doc.meta:
        return normalize_sparse(doc.meta[fallback_meta_key])

    return None
