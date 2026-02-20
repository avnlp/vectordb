"""Base utilities for metadata filtering pipelines.

Provides Timer context manager and abstract BaseMetadataFilteringPipeline class
with shared initialization and orchestration logic.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from haystack.components.embedders import SentenceTransformersDocumentEmbedder

from vectordb.haystack.metadata_filtering.vectordb_pipeline_type import (
    FilteredQueryResult,
    FilterField,
)
from vectordb.utils.logging import LoggerFactory


__all__ = ["Timer", "BaseMetadataFilteringPipeline"]


class Timer:
    """Context manager for measuring elapsed time with perf_counter precision.

    Captures start and end times using time.perf_counter() for lightweight
    timing without instrumentation overhead.

    Attributes:
        start_time: Timestamp when context entered (0.0 if not started).
        end_time: Timestamp when context exited (0.0 if not exited).
    """

    def __init__(self) -> None:
        """Initialize Timer with zero start/end times."""
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    def __enter__(self) -> "Timer":
        """Enter context manager and start timer.

        Returns:
            Self for use in 'with' statement.
        """
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and stop timer.

        Args:
            exc_type: Exception type if raised.
            exc_val: Exception value if raised.
            exc_tb: Exception traceback if raised.
        """
        self.end_time = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds.

        Returns:
            Elapsed time in milliseconds (end_time - start_time) * 1000.
            Returns 0.0 if timer not started/stopped.
        """
        if self.start_time == 0.0 or self.end_time == 0.0:
            return 0.0
        return (self.end_time - self.start_time) * 1000.0


class BaseMetadataFilteringPipeline(ABC):
    """Abstract base class for metadata filtering pipelines.

    Provides shared initialization, embedder setup, and orchestration.
    Each vector database implements concrete subclass.

    Attributes:
        config: Configuration dictionary.
        logger: Logger instance.
        embedder: Document embedder (initialized lazily).
    """

    def __init__(self, config_path: str) -> None:
        """Initialize pipeline from YAML configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        self.embedder: SentenceTransformersDocumentEmbedder | None = None
        self._connect()

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file.

        Returns:
            Configuration dictionary.
        """
        import yaml

        with open(config_path) as f:
            return yaml.safe_load(f)

    def _setup_logger(self) -> logging.Logger:
        """Set up logger from configuration.

        Returns:
            Configured logger instance.
        """
        logging_config = self.config.get("logging", {})
        logger_name = logging_config.get("name", "metadata_filtering")
        log_level_str = logging_config.get("level", "INFO")
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)

        factory = LoggerFactory(logger_name, log_level=log_level)
        return factory.get_logger()

    def _init_embedder(self) -> None:
        """Initialize document embedder if not already done.

        Sets self.embedder from configuration.
        """
        if self.embedder is not None:
            return

        embeddings_config = self.config.get("embeddings", {})
        model_name = embeddings_config.get(
            "model", "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.embedder = SentenceTransformersDocumentEmbedder(model=model_name)
        self.embedder.warm_up()
        self.logger.info("Initialized embedder with model: %s", model_name)

    @abstractmethod
    def _connect(self) -> None:
        """Establish connection to vector database.

        Implemented by subclasses for specific DB connections.
        """

    def _get_metadata_schema(self) -> dict[str, FilterField]:
        """Extract metadata schema from configuration.

        Returns:
            Dict mapping field names to FilterField definitions.
        """
        metadata_filtering = self.config.get("metadata_filtering", {})
        schema_list = metadata_filtering.get("schema", [])

        schema: dict[str, FilterField] = {}
        for field_def in schema_list:
            name = field_def["field"]
            field_type = field_def.get("type", "string")
            operators = field_def.get("operators", [])
            description = field_def.get("description", "")

            schema[name] = FilterField(
                name=name,
                type=field_type,
                operators=operators,
                description=description,
            )

        return schema

    @abstractmethod
    def run(self) -> list[FilteredQueryResult]:
        """Execute complete metadata filtering pipeline.

        Implemented by subclasses to orchestrate:
        1. Load data
        2. Embed documents
        3. Index into database
        4. Apply pre-filter
        5. Run vector search on candidates
        6. Rank results

        Returns:
            List of FilteredQueryResult objects with timing metrics.
        """
