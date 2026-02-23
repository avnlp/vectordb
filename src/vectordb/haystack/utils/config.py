"""Configuration loader for Haystack pipelines.

Re-exports ConfigLoader from vectordb.utils.config_loader to provide a consistent
import path for Haystack pipelines.

Usage:
    >>> from vectordb.haystack.utils import ConfigLoader
    >>> config = ConfigLoader.load("pipeline_config.yaml")
    >>> ConfigLoader.validate(config, "pinecone")
"""

from vectordb.utils.config_loader import ConfigLoader


__all__ = ["ConfigLoader"]
