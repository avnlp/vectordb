"""Utility functions for parent document retrieval operations.

This module provides essential utilities for implementing parent document retrieval
with consistent ID generation, metadata schemas, and hierarchy management across
different vector database backends.

Key Components:
    - ids: Deterministic ID generation for parent-child document relationships
    - metadata: Canonical metadata schema for hierarchical documents
    - hierarchy: Document hierarchy processing and traversal utilities
    - config: Configuration loading and validation for parent document pipelines

ID Generation:
    All IDs are deterministically generated from content, ensuring:
    - Idempotent indexing (same content = same ID)
    - No external ID mapping storage required
    - Reproducible pipeline runs

Metadata Schema:
    Parent Documents (level=1):
        - level: Hierarchy level (1 for parent, 2 for child)
        - doc_idx: Original document index
        - parent_idx: Parent chunk index within document
        - source_id: Original source identifier

    Child Documents (level=2):
        - level: Hierarchy level (1 for parent, 2 for child)
        - parent_id: Reference to parent document (critical for retrieval)
        - doc_idx: Original document index
        - parent_idx: Parent chunk index
        - child_idx: Child chunk index within parent
        - source_id: Original source identifier

The parent_id field is essential: it links child matches back to their parent
documents during retrieval, enabling the system to return comprehensive context.

Usage:
    >>> from vectordb.haystack.parent_document_retrieval.utils import (
    ...     generate_document_id,
    ...     generate_chunk_id,
    ...     create_parent_metadata,
    ...     create_child_metadata,
    ... )
    >>> doc_id = generate_document_id("Document content", "source.txt")
    >>> parent_meta = create_parent_metadata(doc_idx=0, parent_idx=0)
    >>> child_meta = create_child_metadata(
    ...     parent_id=doc_id, doc_idx=0, parent_idx=0, child_idx=0
    ... )

Integration Points:
    - vectordb.haystack.parent_document_retrieval.indexing: Indexing pipelines
    - vectordb.haystack.parent_document_retrieval.search: Search pipelines
"""
