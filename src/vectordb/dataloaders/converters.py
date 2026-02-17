"""Document converters for framework integrations."""

from __future__ import annotations

from typing import Any

from haystack import Document as HaystackDocument
from langchain_core.documents import Document as LangChainDocument

from vectordb.dataloaders.types import DatasetRecord


class DocumentConverter:
    """Convert normalized records into framework document objects."""

    @staticmethod
    def _ensure_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for item in items:
            if "text" not in item or "metadata" not in item:
                raise KeyError("Missing required document keys.")
        return items

    @classmethod
    def to_haystack(cls, items: list[dict[str, Any]]) -> list[HaystackDocument]:
        """Convert normalized items to Haystack documents.

        Args:
            items: List of dicts with "text" and "metadata" keys.

        Returns:
            List of Haystack Document instances.
        """
        cls._ensure_items(items)
        return [
            HaystackDocument(content=item["text"], meta=item["metadata"])
            for item in items
        ]

    @classmethod
    def to_langchain(cls, items: list[dict[str, Any]]) -> list[LangChainDocument]:
        """Convert normalized items to LangChain documents.

        Args:
            items: List of dicts with "text" and "metadata" keys.

        Returns:
            List of LangChain Document instances.
        """
        cls._ensure_items(items)
        return [
            LangChainDocument(page_content=item["text"], metadata=item["metadata"])
            for item in items
        ]


def records_to_items(records: list[DatasetRecord]) -> list[dict[str, Any]]:
    """Convert DatasetRecord objects into converter input items.

    Args:
        records: List of normalized dataset records.

    Returns:
        List of dicts with "text" and "metadata" keys.
    """
    return [{"text": record.text, "metadata": record.metadata} for record in records]
