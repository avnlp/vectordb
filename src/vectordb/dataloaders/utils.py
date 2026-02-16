"""Dataloader utilities for logging and document conversion.

This module provides common utilities used across dataloader implementations,
including logging configuration and bidirectional conversion between
standardized dicts and framework-specific Document objects.

Design Principles:
    - Centralized logging configuration for consistent output format
    - Validation utilities to ensure data integrity before conversion
    - Bidirectional conversion between dicts and framework Documents

Logging Configuration:
    Logging is configured once at module import time using environment
    variable LOG_LEVEL (default: INFO). This ensures consistent logging
    across all dataloader modules without requiring explicit setup.

Validation:
    The validate_documents function ensures that data conforms to the
    expected structure (dicts with "text" and "metadata" keys) before
    conversion operations, catching errors early in the pipeline.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any


# TYPE_CHECKING block prevents circular imports at runtime
# while enabling type checking for framework-specific types
if TYPE_CHECKING:
    from haystack import Document as HaystackDocument
    from langchain_core.documents import Document as LangchainDocument


def _configure_logging() -> None:
    """Configure logging once at module load time.

    Sets up basic logging configuration with format:
    "%(asctime)s - %(levelname)s - %(message)s"

    Log level is controlled by LOG_LEVEL environment variable
    (default: INFO). This function is called automatically on
    module import to ensure consistent logging across dataloaders.
    """
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


# Execute logging configuration on module import
_configure_logging()


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance.

    Returns a logger with the module-level configuration already applied.
    Use this instead of logging.getLogger() directly to ensure consistent
    formatting and log levels.

    Args:
        name: The name for the logger (typically __name__).

    Returns:
        A configured logging.Logger instance.

    Example:
        >>> from vectordb.dataloaders.utils import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Loading dataset...")
    """
    return logging.getLogger(name)


def validate_documents(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate a list of dictionaries for document conversion.

    Ensures that each dictionary contains the required keys ("text" and
    "metadata") and that metadata is a dictionary. This validation
    catches structural errors before conversion to framework Documents.

    Args:
        data: A list of dictionaries, each containing "text" and "metadata" keys.

    Returns:
        The validated list of document dictionaries (unchanged if valid).

    Raises:
        KeyError: If required keys are missing from any document.
        TypeError: If metadata is not a dictionary in any document.

    Example:
        >>> data = [{"text": "Content", "metadata": {"key": "value"}}]
        >>> validate_documents(data)  # Returns data if valid
        [{"text": "Content", "metadata": {"key": "value"}}]
    """
    for doc in data:
        missing = [k for k in ("text", "metadata") if k not in doc]
        if missing:
            raise KeyError(f"Missing keys in document: {missing}")
        if not isinstance(doc["metadata"], dict):
            raise TypeError("Metadata must be a dictionary.")
    return data


# Backward compatibility alias
# Some existing code may use dict_to_documents instead of validate_documents
dict_to_documents = validate_documents


def langchain_docs_to_dict(docs: list[LangchainDocument]) -> list[dict[str, Any]]:
    """Convert LangChain Document objects to standardized dictionaries.

    Performs the inverse of DocumentConverter.to_langchain(), extracting
    page_content and metadata from LangChain Document objects.

    Args:
        docs: A list of LangChain Document objects.

    Returns:
        A list of dictionaries with "text" and "metadata" keys.

    Example:
        >>> from langchain_core.documents import Document
        >>> lc_docs = [Document(page_content="Hello", metadata={"id": 1})]
        >>> langchain_docs_to_dict(lc_docs)
        [{"text": "Hello", "metadata": {"id": 1}}]
    """
    return [{"text": doc.page_content, "metadata": doc.metadata} for doc in docs]


def haystack_docs_to_dict(docs: list[HaystackDocument]) -> list[dict[str, Any]]:
    """Convert Haystack Document objects to standardized dictionaries.

    Performs the inverse of DocumentConverter.to_haystack(), extracting
    content and meta from Haystack Document objects.

    Args:
        docs: A list of Haystack Document objects.

    Returns:
        A list of dictionaries with "text" and "metadata" keys.

    Example:
        >>> from haystack import Document
        >>> hs_docs = [Document(content="Hello", meta={"id": 1})]
        >>> haystack_docs_to_dict(hs_docs)
        [{"text": "Hello", "metadata": {"id": 1}}]
    """
    return [{"text": doc.content, "metadata": doc.meta} for doc in docs]
