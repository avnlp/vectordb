"""Data loading utilities for LangChain namespace pipelines."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from vectordb.dataloaders import DataloaderCatalog
from vectordb.dataloaders.types import DatasetType


def load_documents_from_config(
    config: dict[str, Any],
    split_override: str | None = None,
    limit_override: int | None = None,
) -> list[Document]:
    """Load documents from a dataset based on configuration.

    Args:
        config: Configuration dictionary with dataset settings.
        split_override: Override the split specified in config.
        limit_override: Override the limit specified in config.

    Returns:
        List of LangChain Documents.

    Raises:
        ValueError: If dataset type is not specified or supported.
    """
    dataset_config = config.get("dataset", config.get("dataloader", {}))
    dataset_type: DatasetType = dataset_config.get("type", "").lower()
    dataset_name = dataset_config.get("dataset_name")
    split = split_override or dataset_config.get("split", "test")
    limit = limit_override or dataset_config.get("limit")

    if not dataset_type:
        raise ValueError("Dataset type must be specified in config")

    loader = DataloaderCatalog.create(
        dataset_type,
        split=split,
        limit=limit,
        dataset_id=dataset_name,
    )

    return loader.load().to_langchain()


def get_namespace_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract namespace definitions from configuration.

    Args:
        config: Configuration dictionary with namespaces section.

    Returns:
        List of namespace definition dictionaries.
    """
    namespaces_config = config.get("namespaces", {})
    return namespaces_config.get("definitions", [])
