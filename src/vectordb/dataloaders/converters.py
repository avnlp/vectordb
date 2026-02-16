"""Document format converters for framework integration.

This module provides conversion utilities to transform standardized dataloader
output into framework-specific Document objects. The standardized format
(dict with "text" and "metadata" keys) is decoupled from framework types
to avoid tight coupling between dataloaders and specific frameworks.

Design Rationale:
    The converters maintain separation between data loading and framework
    integration. This allows:
    1. Dataloaders to remain framework-agnostic
    2. Easy addition of new framework support without modifying dataloaders
    3. Consistent document structure across different vector DB backends

Supported Frameworks:
    - Haystack: Converts to haystack.Document with content/meta fields
    - LangChain: Converts to langchain_core.documents.Document with
      page_content/metadata fields

Usage Patterns:
    Typical usage involves loading data, then converting for the target framework:

        >>> from vectordb.dataloaders import DatasetRegistry, DocumentConverter
        >>> data = DatasetRegistry.load("triviaqa", limit=100)
        >>>
        >>> # For Haystack pipelines
        >>> haystack_docs = DocumentConverter.to_haystack(data)
        >>>
        >>> # For LangChain chains
        >>> langchain_docs = DocumentConverter.to_langchain(data)

Integration Points:
    - Used by indexing pipelines before vector store insertion
    - Used by evaluation scripts to prepare documents for retrieval
    - Framework-specific dataloaders in haystack/ and langchain/ submodules
      use these converters internally
"""

from __future__ import annotations

from typing import Any

from haystack import Document as HaystackDocument
from langchain_core.documents import Document as LangChainDocument


class DocumentConverter:
    """Converts standardized dicts to framework-specific Document objects.

    This utility class provides static methods for converting the standardized
    dataloader output format (list of dicts with "text" and "metadata" keys)
    into Haystack or LangChain Document objects.

    The conversion preserves all metadata and maintains document content
    integrity during the transformation.
    """

    @staticmethod
    def to_haystack(items: list[dict[str, Any]]) -> list[HaystackDocument]:
        """Convert standardized dicts to Haystack Documents.

        Transforms the dataloader output format into Haystack Document objects
        suitable for use with Haystack pipelines and components.

        Args:
            items: List of dicts with structure {"text": str, "metadata": dict}

        Returns:
            List of Haystack Document objects with content and meta fields

        Example:
            >>> data = [{"text": "Content", "metadata": {"key": "value"}}]
            >>> docs = DocumentConverter.to_haystack(data)
            >>> docs[0].content
            'Content'
            >>> docs[0].meta
            {'key': 'value'}
        """
        return [
            HaystackDocument(content=item["text"], meta=item["metadata"])
            for item in items
        ]

    @staticmethod
    def to_langchain(items: list[dict[str, Any]]) -> list[LangChainDocument]:
        """Convert standardized dicts to LangChain Documents.

        Transforms the dataloader output format into LangChain Document objects
        suitable for use with LangChain chains and vector stores.

        Args:
            items: List of dicts with structure {"text": str, "metadata": dict}

        Returns:
            List of LangChain Document objects with page_content and metadata

        Example:
            >>> data = [{"text": "Content", "metadata": {"key": "value"}}]
            >>> docs = DocumentConverter.to_langchain(data)
            >>> docs[0].page_content
            'Content'
            >>> docs[0].metadata
            {'key': 'value'}
        """
        return [
            LangChainDocument(page_content=item["text"], metadata=item["metadata"])
            for item in items
        ]
