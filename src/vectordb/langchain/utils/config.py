"""Configuration loader for LangChain pipelines.

Re-exports ConfigLoader from vectordb.utils.config_loader to provide a consistent
import path for LangChain pipelines. Both Haystack and LangChain pipelines
share the same configuration format and loading logic.

Usage:
    >>> from vectordb.langchain.utils import ConfigLoader
    >>> config = ConfigLoader.load("pipeline_config.yaml")
"""

from vectordb.utils.config_loader import ConfigLoader


__all__ = ["ConfigLoader"]
