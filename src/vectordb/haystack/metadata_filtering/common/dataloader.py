"""Data loading utilities for metadata filtering pipelines.

Integrates DatasetRegistry to load documents from configured datasets.
"""

from typing import Any

from haystack import Document

from vectordb.dataloaders import DataloaderCatalog


__all__ = ["load_documents_from_config"]


def load_documents_from_config(config: dict[str, Any]) -> list[Document]:
    """Load documents from configured dataset using DatasetRegistry.

    Configuration should include a 'dataloader' section with:
    - type: Dataset type (triviaqa, arc, popqa, factscore, earnings_calls)
    - dataset_name: Optional HuggingFace dataset ID
    - split: Dataset split (default: test)
    - limit: Max items to load (optional)

    Args:
        config: Configuration dictionary.

    Returns:
        List of Haystack Documents with text and metadata.

    Raises:
        KeyError: If 'dataloader' section missing or invalid.
        ValueError: If dataset type not supported.
    """
    dataloader_config = config.get("dataloader", {})

    if not dataloader_config:
        raise ValueError("No 'dataloader' configuration found")

    dataset_type = dataloader_config.get("type")
    if not dataset_type:
        raise ValueError("'dataloader.type' is required")

    loader = DataloaderCatalog.create(
        dataset_type,
        split=dataloader_config.get("split", "test"),
        limit=dataloader_config.get("limit"),
        dataset_id=dataloader_config.get("dataset_name"),
    )

    return loader.load().to_haystack()
