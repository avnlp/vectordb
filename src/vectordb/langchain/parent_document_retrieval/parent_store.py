"""Parent document store for managing parent-child document relationships.

This module provides an in-memory store for managing the relationship between
parent documents and their chunks. It maintains bidirectional mappings that
enable:
    - Storing parent documents with their complete text and metadata
    - Mapping chunk IDs to their parent document IDs
    - Retrieving full parent documents from chunk IDs
    - Persisting and loading mappings for reuse across sessions

The ParentDocumentStore uses two primary data structures:
    - parent_map: Dict mapping parent_id -> parent document data
    - chunk_to_parent: Dict mapping chunk_id -> parent_id

Parent Document Structure:
    Each parent document is stored as a dictionary containing:
        - text: Full document content
        - metadata: Document metadata (source, author, date, etc.)
        - source_index: Original index in the document collection

Usage Example:
    >>> from vectordb.langchain.parent_document_retrieval.parent_store import (
    ...     ParentDocumentStore,
    ... )
    >>>
    >>> # Initialize store with persistence
    >>> store = ParentDocumentStore(cache_dir="./cache")
    >>>
    >>> # Add parent document
    >>> store.add_parent(
    ...     "parent_1",
    ...     {
    ...         "text": "Full document text...",
    ...         "metadata": {"source": "doc1.txt"},
    ...         "source_index": 0,
    ...     },
    ... )
    >>>
    >>> # Map chunks to parent
    >>> store.add_chunk_mapping("chunk_1", "parent_1")
    >>> store.add_chunk_mapping("chunk_2", "parent_1")
    >>>
    >>> # Retrieve parent from chunk
    >>> parent = store.get_parent("chunk_1")
    >>>
    >>> # Save to disk
    >>> store.save("parent_store.pkl")
    >>>
    >>> # Load from disk
    >>> loaded_store = ParentDocumentStore.load("./cache/parent_store.pkl")

Note:
    The store is designed for in-memory operations with optional persistence.
    For large-scale production use, consider implementing a database-backed
    store or caching layer.
"""

import pickle
from pathlib import Path
from typing import Any


class ParentDocumentStore:
    """In-memory store for managing parent documents with optional persistence.

    Maintains bidirectional mappings between parent documents and their chunks,
    enabling retrieval of full parent documents from chunk IDs found during
    vector search.

    Attributes:
        parent_map: Dictionary mapping parent_id -> parent document data.
            Parent document data includes full text, metadata, and source index.
        chunk_to_parent: Dictionary mapping chunk_id -> parent_id.
            Enables reverse lookup from search results to parent documents.
        cache_dir: Optional directory path for persisting the store.

    Example:
        >>> store = ParentDocumentStore(cache_dir="./cache")
        >>>
        >>> # Store a parent document
        >>> store.add_parent(
        ...     "doc_1",
        ...     {
        ...         "text": "Complete document content...",
        ...         "metadata": {"author": "John"},
        ...         "source_index": 0,
        ...     },
        ... )
        >>>
        >>> # Map chunks to this parent
        >>> for i in range(5):
        ...     store.add_chunk_mapping(f"chunk_{i}", "doc_1")
        >>>
        >>> # Retrieve parent from any chunk
        >>> parent = store.get_parent("chunk_2")
        >>> print(parent["text"])  # Full document text
    """

    def __init__(self, cache_dir: str | None = None) -> None:
        """Initialize the parent document store.

        Args:
            cache_dir: Optional directory to persist parent documents.
                If provided, the directory will be created if it doesn't exist.

        Example:
            >>> store = ParentDocumentStore()  # In-memory only
            >>> store = ParentDocumentStore(cache_dir="./cache")  # With persistence
        """
        # Mapping: parent_id -> parent document data (text, metadata, source_index)
        self.parent_map: dict[str, dict[str, Any]] = {}

        # Mapping: chunk_id -> parent_id (enables chunk-to-parent lookup)
        self.chunk_to_parent: dict[str, str] = {}

        self.cache_dir = cache_dir

        # Create cache directory if persistence is enabled
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

    def add_parent(self, parent_id: str, parent_doc: dict[str, Any]) -> None:
        """Add a parent document to the store.

        Stores the complete parent document including full text and metadata.
        This enables retrieval of complete documents during search, rather
        than just individual chunks.

        Args:
            parent_id: Unique identifier for the parent document.
                Typically a UUID generated during indexing.
            parent_doc: Parent document data dictionary containing:
                - text: Full document content
                - metadata: Document metadata dict
                - source_index: Index in original document collection

        Example:
            >>> store.add_parent(
            ...     "parent_1",
            ...     {
            ...         "text": "Full document content here...",
            ...         "metadata": {"source": "article.txt"},
            ...         "source_index": 0,
            ...     },
            ... )
        """
        self.parent_map[parent_id] = parent_doc

    def add_chunk_mapping(self, chunk_id: str, parent_id: str) -> None:
        """Add a chunk-to-parent mapping.

        Creates the link between a chunk ID (stored in the vector database)
        and its parent document ID. This mapping is crucial for the parent
        document retrieval pattern: search finds chunks, but we return parents.

        Args:
            chunk_id: Unique identifier for the chunk.
                This ID is stored as metadata in the vector database.
            parent_id: Unique identifier for the parent document.
                Must reference a parent already added via add_parent().

        Note:
            Each chunk can belong to exactly one parent. Multiple chunks
            from the same parent will all map to that parent's ID.

        Example:
            >>> store.add_parent("parent_1", {...})
            >>> store.add_chunk_mapping("chunk_a", "parent_1")
            >>> store.add_chunk_mapping("chunk_b", "parent_1")
        """
        self.chunk_to_parent[chunk_id] = parent_id

    def get_parent(self, chunk_id: str) -> dict[str, Any] | None:
        """Retrieve a parent document by chunk ID.

        Performs a two-step lookup:
            1. Find parent_id from chunk_id via chunk_to_parent mapping
            2. Retrieve parent document from parent_map using parent_id

        Args:
            chunk_id: Unique identifier for the chunk.
                This is typically retrieved from vector search results.

        Returns:
            Parent document dictionary with keys: text, metadata, source_index.
            Returns None if chunk_id or parent_id is not found.

        Example:
            >>> parent = store.get_parent("chunk_123")
            >>> if parent:
            ...     print(f"Found parent: {parent['text'][:100]}...")
        """
        # First lookup: chunk_id -> parent_id
        parent_id = self.chunk_to_parent.get(chunk_id)
        if parent_id is None:
            return None

        # Second lookup: parent_id -> parent document
        return self.parent_map.get(parent_id)

    def get_parent_by_id(self, parent_id: str) -> dict[str, Any] | None:
        """Retrieve a parent document by its ID.

        Direct lookup in parent_map without going through chunk mapping.
        Useful when you already know the parent_id and need the full document.

        Args:
            parent_id: Unique identifier for the parent document.

        Returns:
            Parent document dictionary or None if not found.

        Example:
            >>> parent = store.get_parent_by_id("parent_123")
            >>> if parent:
            ...     print(f"Document text: {parent['text']}")
        """
        return self.parent_map.get(parent_id)

    def get_parents_for_chunks(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        """Retrieve unique parent documents for multiple chunks.

        Efficiently retrieves parent documents for a batch of chunk IDs,
        deduplicating parents to avoid returning the same parent multiple times.

        Args:
            chunk_ids: List of chunk identifiers from vector search results.
                These are typically retrieved from top-k search hits.

        Returns:
            List of unique parent documents in order of first appearance.
            Parents are deduplicated using a set to track seen parent_ids.

        Example:
            >>> chunk_ids = ["chunk_1", "chunk_2", "chunk_3"]
            >>> parents = store.get_parents_for_chunks(chunk_ids)
            >>> print(f"Retrieved {len(parents)} unique parents")
        """
        seen_parents: set[str] = set()
        parents: list[dict[str, Any]] = []

        for chunk_id in chunk_ids:
            parent_id = self.chunk_to_parent.get(chunk_id)
            if parent_id and parent_id not in seen_parents:
                parent = self.parent_map.get(parent_id)
                if parent is not None:
                    seen_parents.add(parent_id)
                    parents.append(parent)

        return parents

    def save(self, filename: str) -> str:
        """Persist parent store to disk using pickle.

        Saves both parent_map and chunk_to_parent to a pickle file for
        later restoration. Requires cache_dir to be set during initialization.

        Args:
            filename: Filename to save to (relative to cache_dir).
                Example: "parent_store.pkl"

        Returns:
            Absolute path to the saved file.

        Raises:
            ValueError: If cache_dir was not set during initialization.

        Example:
            >>> store = ParentDocumentStore(cache_dir="./cache")
            >>> # ... add parents and mappings ...
            >>> path = store.save("my_store.pkl")
            >>> print(f"Saved to: {path}")
        """
        if self.cache_dir is None:
            msg = "cache_dir not set"
            raise ValueError(msg)

        filepath = Path(self.cache_dir) / filename
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "parent_map": self.parent_map,
                    "chunk_to_parent": self.chunk_to_parent,
                },
                f,
            )

        return str(filepath)

    @classmethod
    def load(cls, filepath: str) -> "ParentDocumentStore":
        """Load parent store from disk.

        Restores a previously saved ParentDocumentStore from a pickle file.
        The cache_dir is set to the directory containing the pickle file.

        Args:
            filepath: Path to saved parent store pickle file.

        Returns:
            Loaded ParentDocumentStore instance with restored mappings.

        Example:
            >>> store = ParentDocumentStore.load("./cache/parent_store.pkl")
            >>> print(f"Loaded {len(store)} parents")
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        store = cls(cache_dir=str(Path(filepath).parent))
        store.parent_map = data["parent_map"]
        store.chunk_to_parent = data["chunk_to_parent"]

        return store

    def clear(self) -> None:
        """Clear all parent documents and chunk mappings.

        Removes all data from the store without deleting the cache directory.
        Use this when you need to reset the store, e.g., for reindexing.

        Example:
            >>> store.clear()
            >>> print(len(store))  # 0
        """
        self.parent_map.clear()
        self.chunk_to_parent.clear()

    def __len__(self) -> int:
        """Return total number of parent documents.

        Returns:
            Integer count of unique parent documents in the store.

        Example:
            >>> store = ParentDocumentStore()
            >>> store.add_parent("p1", {...})
            >>> store.add_parent("p2", {...})
            >>> len(store)  # 2
        """
        return len(self.parent_map)

    def __contains__(self, chunk_id: str) -> bool:
        """Check if a chunk ID has a parent mapping.

        Args:
            chunk_id: Chunk identifier to check.

        Returns:
            True if chunk_id exists in chunk_to_parent mapping, False otherwise.

        Example:
            >>> store.add_chunk_mapping("chunk_1", "parent_1")
            >>> "chunk_1" in store  # True
            >>> "chunk_2" in store  # False
        """
        return chunk_id in self.chunk_to_parent
