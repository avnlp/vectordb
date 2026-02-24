"""Deterministic ID generation for parent document retrieval.

This module provides functions for generating deterministic, unique identifiers
for documents in a parent-child hierarchy. Using deterministic IDs ensures
consistency across indexing and retrieval operations.

ID Generation Strategy:
    - Document IDs: Based on content hash + optional source identifier
    - Parent IDs: Based on content hash + document index
    - Chunk IDs: Based on parent_id + chunk index + level

Benefits of deterministic IDs:
    - Same content always produces same ID (idempotent indexing)
    - No need to store ID mappings externally
    - Reproducible across pipeline runs
"""

import hashlib


def generate_document_id(content: str, source_id: str | None = None) -> str:
    """Generate deterministic document ID from content.

    Creates a unique identifier by hashing the document content combined
    with an optional source identifier. This ensures the same content from
    the same source always generates the same ID.

    Args:
        content: Document content to hash
        source_id: Optional source identifier (e.g., filename, URL) to scope the ID

    Returns:
        20-character hexadecimal hash string

    Example:
        >>> generate_document_id("Hello world", "doc1.txt")
        'a8f5f167f44f4964e6c9'
        >>> generate_document_id("Hello world", "doc1.txt")  # Same inputs = same ID
        'a8f5f167f44f4964e6c9'
    """
    hash_input = f"{source_id or ''}{content}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:20]


def generate_chunk_id(
    parent_id: str,
    chunk_idx: int,
    level: int,
) -> str:
    """Generate deterministic chunk ID.

    Creates a unique identifier for a chunk based on its parent document ID,
    chunk index, and hierarchy level. This allows chunks to be uniquely
    identified while maintaining relationship information.

    Args:
        parent_id: Parent document ID this chunk belongs to
        chunk_idx: Index of this chunk within its parent
        level: Hierarchy level (1=parent, 2=child, etc.)

    Returns:
        16-character hexadecimal hash string

    Example:
        >>> generate_chunk_id("parent_0_abc123", chunk_idx=0, level=2)
        '7d8f9e2b4c5a6d8f'
    """
    hash_input = f"{parent_id}{chunk_idx}{level}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


def generate_parent_id(content: str, doc_idx: int) -> str:
    """Generate deterministic parent document ID.

    Creates a unique identifier for a parent document by combining the document
    index with a hash of its content. The prefix 'parent_' clearly identifies
    the document type.

    Args:
        content: Parent document content to hash
        doc_idx: Index of the document in the dataset

    Returns:
        Parent ID string in format 'parent_{doc_idx}_{content_hash}'

    Example:
        >>> generate_parent_id("Parent document content", doc_idx=0)
        'parent_0_a1b2c3d4e5f67890'
    """
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"parent_{doc_idx}_{content_hash}"
