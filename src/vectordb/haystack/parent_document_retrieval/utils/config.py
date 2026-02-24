"""Configuration utilities for parent document retrieval pipelines.

This module provides configuration loading and validation specifically for
parent document retrieval pipelines. It extends the base configuration
utilities with validation for required parent document retrieval settings.

Required Configuration Sections:
    - database: Vector database connection settings
    - embeddings: Embedding model configuration
    - dataloader: Dataset loading configuration

Optional Configuration:
    - chunking: Parent/child chunk size settings
    - retrieval: Auto-merging threshold and other retrieval params
"""

from pathlib import Path
from typing import Any

from vectordb.utils.config import load_config as base_load_config


def load_parent_doc_config(
    config_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Load and validate parent document retrieval configuration.

    Loads configuration from a YAML file or dictionary and validates that
    all required sections for parent document retrieval are present.

    Required Configuration Sections:
        - database: Vector database settings (chroma, pinecone, qdrant, etc.)
        - embeddings: Model name and embedding settings
        - dataloader: Dataset type, name, split, and limit settings

    Args:
        config_or_path: Configuration dictionary or path to YAML config file

    Returns:
        Validated configuration dictionary with all required sections

    Raises:
        ValueError: If any required configuration section is missing

    Example:
        >>> config = load_parent_doc_config("config.yaml")
        >>> config["database"]["chroma"]["collection_name"]
        'parent_doc_leaves'
    """
    if isinstance(config_or_path, dict):
        config = config_or_path
    else:
        config = base_load_config(str(config_or_path))

    required = ["database", "embeddings", "dataloader"]
    for key in required:
        if key not in config:
            raise ValueError(f"Config missing required section: {key}")

    return config
