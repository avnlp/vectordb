"""Hierarchical document processing utilities using Haystack components.

This module provides functions for creating and managing document hierarchies
using Haystack's HierarchicalDocumentSplitter. It handles the two-level
hierarchy required for parent document retrieval:

Hierarchy Structure:
    - Level 1 (Parent): Larger chunks containing multiple concepts
    - Level 2+ (Children/Leaves): Smaller chunks for precise retrieval

The hierarchy enables parent document retrieval by:
    1. Indexing leaf nodes (children) for fine-grained search
    2. Storing parent nodes for context-rich results
    3. Mapping between levels via parent_id relationships in metadata
"""

from haystack import Document
from haystack.components.preprocessors import HierarchicalDocumentSplitter


def create_hierarchy(
    documents: list[Document],
    parent_size_words: int = 100,
    child_size_words: int = 25,
    split_overlap: int = 5,
) -> dict[str, list[Document]]:
    """Create parent-child document hierarchy using Haystack splitter.

    Uses HierarchicalDocumentSplitter to split documents into parent and child
    chunks. The splitter creates a two-level hierarchy where:
    - Parents (level=1) contain multiple child chunks
    - Children (leaves) have no children_ids in metadata

    Args:
        documents: Input documents to split
        parent_size_words: Size of parent chunks in words. Larger chunks
            provide broader context for retrieval.
        child_size_words: Size of child chunks in words. Smaller chunks
            enable precise matching during search.
        split_overlap: Number of words to overlap between chunks to
            prevent context loss at chunk boundaries.

    Returns:
        Dict with 'parents' and 'leaves' lists. Parents are the level=1
        documents, leaves are documents with no children (the searchable
        child chunks).

    Example:
        >>> docs = [Document(content="Long document text...")]
        >>> hierarchy = create_hierarchy(
        ...     docs, parent_size_words=100, child_size_words=25
        ... )
        >>> len(hierarchy["parents"])  # Number of parent chunks
        >>> len(hierarchy["leaves"])  # Number of child chunks
    """
    splitter = HierarchicalDocumentSplitter(
        block_sizes={parent_size_words, child_size_words},
        split_overlap=split_overlap,
        split_by="word",
    )

    all_docs = splitter.run(documents)["documents"]

    # Parents have level=1 in their metadata (assigned by Haystack splitter)
    parents = [d for d in all_docs if d.meta.get("level") == 1]

    # Leaves have no children_ids, meaning they are the leaf nodes in the hierarchy
    leaves = [d for d in all_docs if not d.meta.get("children_ids")]

    return {"parents": parents, "leaves": leaves}


def filter_by_level(documents: list[Document], level: int) -> list[Document]:
    """Filter documents by hierarchy level.

    Filters a list of documents to return only those matching the specified
    hierarchy level. Used to separate parents from children after splitting.

    Args:
        documents: List of documents with level metadata
            (from HierarchicalDocumentSplitter)
        level: Desired level (1=parent, 2=child, etc.)

    Returns:
        Filtered list of documents at the specified level

    Example:
        >>> all_docs = splitter.run(docs)["documents"]
        >>> parents = filter_by_level(all_docs, level=1)
        >>> children = filter_by_level(all_docs, level=2)
    """
    return [doc for doc in documents if doc.meta.get("level") == level]
