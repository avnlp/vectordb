"""Canonical metadata schema for parent document retrieval.

This module defines the standard metadata schema used throughout the parent
document retrieval system. The schema ensures consistent metadata handling
across indexing and search operations.

Metadata Schema:
    Parent Documents (level=1):
        - level: Always 1 for parent documents
        - doc_idx: Index of the original document
        - parent_idx: Index of this parent chunk within the document
        - source_id: Identifier of the original source (optional)

    Child Documents (level=2):
        - level: Always 2 for child documents
        - parent_id: ID of the parent document (crucial for retrieval mapping)
        - doc_idx: Index of the original document
        - parent_idx: Index of the parent chunk this child belongs to
        - child_idx: Index of this child within its parent
        - source_id: Identifier of the original source (optional)

The parent_id field is essential for parent document retrieval. When a child
chunk matches a query, the parent_id allows the system to fetch the broader
parent document instead of the small child chunk.
"""


def create_parent_metadata(
    doc_idx: int,
    parent_idx: int,
    source_id: str | None = None,
    extra_metadata: dict | None = None,
) -> dict:
    """Create canonical parent document metadata.

    Creates a standardized metadata dictionary for parent documents (level=1).
    Parent documents are the larger chunks that contain multiple child chunks
    and provide broader context during retrieval.

    Args:
        doc_idx: Document index in the original dataset
        parent_idx: Index of this parent chunk within the document
        source_id: Original source identifier (e.g., filename, URL)
        extra_metadata: Additional custom metadata to include

    Returns:
        Dictionary with canonical parent metadata fields

    Example:
        >>> meta = create_parent_metadata(doc_idx=0, parent_idx=0, source_id="doc1.txt")
        >>> meta["level"]
        1
    """
    metadata = {
        "level": 1,  # Parent level in the hierarchy
        "doc_idx": doc_idx,
        "parent_idx": parent_idx,
        "source_id": source_id,
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    return metadata


def create_child_metadata(
    parent_id: str,
    doc_idx: int,
    parent_idx: int,
    child_idx: int,
    source_id: str | None = None,
    extra_metadata: dict | None = None,
) -> dict:
    """Create canonical child document metadata.

    Creates a standardized metadata dictionary for child documents (level=2).
    Child documents are the smaller chunks indexed for retrieval. The parent_id
    field enables mapping from matched children back to their parent documents.

    Args:
        parent_id: ID of the parent document (required for parent retrieval)
        doc_idx: Document index in the original dataset
        parent_idx: Index of the parent chunk this child belongs to
        child_idx: Index of this child within its parent chunk
        source_id: Original source identifier (e.g., filename, URL)
        extra_metadata: Additional custom metadata to include

    Returns:
        Dictionary with canonical child metadata fields including parent_id

    Example:
        >>> meta = create_child_metadata(
        ...     parent_id="parent_0_abc123", doc_idx=0, parent_idx=0, child_idx=0
        ... )
        >>> meta["parent_id"]
        'parent_0_abc123'
    """
    metadata = {
        "level": 2,  # Child level in the hierarchy
        "parent_id": parent_id,  # Maps child to parent for retrieval
        "doc_idx": doc_idx,
        "parent_idx": parent_idx,
        "child_idx": child_idx,
        "source_id": source_id,
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    return metadata


def get_level_from_metadata(metadata: dict | None) -> int | None:
    """Extract hierarchy level from metadata.

    Retrieves the hierarchy level from document metadata. Returns None if
    metadata is None or level is not present.

    Args:
        metadata: Document metadata dictionary

    Returns:
        Hierarchy level (1=parent, 2=child) or None if not found

    Example:
        >>> meta = {"level": 1, "doc_idx": 0}
        >>> get_level_from_metadata(meta)
        1
        >>> get_level_from_metadata(None)
        None
    """
    if metadata is None:
        return None
    return metadata.get("level")


def get_parent_id_from_metadata(metadata: dict | None) -> str | None:
    """Extract parent ID from metadata.

    Retrieves the parent document ID from child metadata. This ID is used
    during retrieval to fetch the parent document when a child matches.

    Args:
        metadata: Document metadata dictionary (typically from a child document)

    Returns:
        Parent document ID or None if not found

    Example:
        >>> meta = {"parent_id": "parent_0_abc123", "level": 2}
        >>> get_parent_id_from_metadata(meta)
        'parent_0_abc123'
        >>> get_parent_id_from_metadata({"level": 1})  # Parent has no parent_id
        None
    """
    if metadata is None:
        return None
    return metadata.get("parent_id")
