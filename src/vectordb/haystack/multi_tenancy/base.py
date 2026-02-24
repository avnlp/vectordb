"""Base class for multi-tenancy pipelines.

Provides common functionality for all vector database multi-tenancy
implementations including config loading, tenant context management,
and abstract methods for subclasses.
"""

from __future__ import annotations

import logging
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import yaml
from haystack import Document

from vectordb.haystack.multi_tenancy.common.tenant_context import TenantContext
from vectordb.haystack.multi_tenancy.common.types import (
    MultitenancyTimingMetrics,
    TenantIsolationConfig,
    TenantIsolationStrategy,
    TenantStats,
)
from vectordb.utils.logging import LoggerFactory


__all__ = [
    "Timer",
    "BaseMultitenancyPipeline",
]


class Timer:
    """Context manager for timing operations.

    Attributes:
        elapsed_ms: Elapsed time in milliseconds after exiting context.

    Example:
        >>> with Timer() as t:
        ...     # do some work
        >>> print(f"Took {t.elapsed_ms:.2f}ms")
    """

    def __init__(self) -> None:
        """Initialize timer."""
        self._start: float = 0.0
        self._end: float = 0.0

    def __enter__(self) -> "Timer":
        """Start the timer."""
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Stop the timer."""
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        """Return elapsed time in milliseconds."""
        return (self._end - self._start) * 1000


def _resolve_env_vars(value: Any) -> Any:
    """Resolve environment variables in configuration values.

    Supports both simple ${VAR} and ${VAR:-default} syntax.

    Args:
        value: The value to resolve, can be a string, dict, or list.

    Returns:
        The resolved value with environment variables expanded.
    """
    if isinstance(value, str):
        pattern = r"\$\{([^}:]+)(?::-([^}]*))?\}"
        match = re.match(pattern, value)
        if match:
            env_var = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(env_var, default)
        return value
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


class BaseMultitenancyPipeline(ABC):
    """Abstract base class for multi-tenant pipelines.

    Provides common functionality for all vector database multi-tenancy
    implementations including config loading, tenant context management,
    and abstract methods for subclasses.

    Attributes:
        config_path: Path to YAML configuration file.
        config: Loaded configuration dictionary.
        tenant_context: Current tenant context for operations.
    """

    DEFAULT_EMBEDDING_MODEL: ClassVar[str] = "Qwen/Qwen3-Embedding-0.6B"
    DEFAULT_EMBEDDING_DIMENSION: ClassVar[int] = 1024

    config_path: str
    config: dict[str, Any]
    tenant_context: TenantContext

    _logger: logging.Logger | None = None

    def __init__(
        self,
        config_path: str,
        tenant_context: TenantContext | None = None,
    ) -> None:
        """Initialize the multi-tenancy pipeline.

        Args:
            config_path: Path to YAML configuration file.
            tenant_context: Explicit tenant context. If None, resolves from
                environment (TENANT_ID) or config (tenant.id).

        Raises:
            FileNotFoundError: If config file does not exist.
            TenantContextError: If tenant cannot be resolved.
        """
        config = self._load_config_static(config_path)
        resolved_tenant = TenantContext.resolve(tenant_context, config)

        self.config_path = config_path
        self.config = config
        self.tenant_context = resolved_tenant
        self._post_init()

    def _post_init(self) -> None:
        """Post-initialization hook for subclasses."""
        self._connect()

    @staticmethod
    def _load_config_static(config_path: str) -> dict[str, Any]:
        """Load configuration from YAML file with env var resolution.

        Args:
            config_path: Path to YAML configuration file.

        Returns:
            Configuration dictionary with environment variables resolved.

        Raises:
            FileNotFoundError: If config file does not exist.
            yaml.YAMLError: If config file is invalid YAML.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path) as f:
            config = yaml.safe_load(f)

        return _resolve_env_vars(config)

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Load configuration from YAML file (traced version).

        Args:
            config_path: Path to YAML configuration file.

        Returns:
            Configuration dictionary.
        """
        return self._load_config_static(config_path)

    @property
    def logger(self) -> logging.Logger:
        """Get logger instance (lazy initialized)."""
        if self._logger is None:
            pipeline_name = self.config.get("pipeline", {}).get(
                "name", self.__class__.__name__
            )
            factory = LoggerFactory.configure_from_env(pipeline_name)
            self._logger = factory.get_logger()
        return self._logger

    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension from config or default.

        Returns:
            Embedding dimension.
        """
        embedding_config = self.config.get("embedding", {})
        output_dim = embedding_config.get("output_dimension")
        if output_dim is not None:
            return output_dim
        return embedding_config.get("dimension", self.DEFAULT_EMBEDDING_DIMENSION)

    def _get_tenant_isolation_config(self) -> TenantIsolationConfig:
        """Extract TenantIsolationConfig from YAML config.

        Returns:
            TenantIsolationConfig from config or defaults.
        """
        mt_config = self.config.get("multitenancy", {})

        return TenantIsolationConfig(
            strategy=mt_config.get("strategy", "partition_key"),
            field_name=mt_config.get("field_name", "tenant_id"),
            auto_create_tenant=mt_config.get("auto_create_tenant", True),
            partition_key_isolation=mt_config.get("partition_key_isolation", False),
            num_partitions=mt_config.get("num_partitions", 64),
        )

    def _get_collection_name(self) -> str:
        """Get collection name from config.

        Returns:
            Collection name with tenant prefix if configured.
        """
        collection_config = self.config.get("collection", {})
        return collection_config.get("name", "multitenancy")

    def _create_timing_metrics(
        self,
        tenant_resolution_ms: float = 0.0,
        index_operation_ms: float = 0.0,
        retrieval_ms: float = 0.0,
        total_ms: float = 0.0,
        num_documents: int = 0,
    ) -> MultitenancyTimingMetrics:
        """Create timing metrics for current tenant.

        Args:
            tenant_resolution_ms: Time to resolve tenant.
            index_operation_ms: Time for indexing.
            retrieval_ms: Time for retrieval.
            total_ms: Total operation time.
            num_documents: Number of documents processed.

        Returns:
            MultitenancyTimingMetrics instance.
        """
        return MultitenancyTimingMetrics(
            tenant_resolution_ms=tenant_resolution_ms,
            index_operation_ms=index_operation_ms,
            retrieval_ms=retrieval_ms,
            total_ms=total_ms,
            tenant_id=self.tenant_context.tenant_id,
            num_documents=num_documents,
        )

    @property
    @abstractmethod
    def isolation_strategy(self) -> TenantIsolationStrategy:
        """Return the isolation strategy for this pipeline.

        Returns:
            TenantIsolationStrategy enum value.
        """

    @abstractmethod
    def _connect(self) -> None:
        """Establish connection to the vector database."""

    @abstractmethod
    def close(self) -> None:
        """Close connection to the vector database."""

    @abstractmethod
    def _get_document_store(self) -> Any:
        """Return configured document store for the specific database.

        Returns:
            Database-specific document store instance.
        """

    @abstractmethod
    def create_tenant(self, tenant_id: str) -> bool:
        """Create a new tenant in the database.

        Args:
            tenant_id: Tenant identifier to create.

        Returns:
            True if tenant was created, False if already exists.
        """

    @abstractmethod
    def tenant_exists(self, tenant_id: str) -> bool:
        """Check if a tenant exists.

        Args:
            tenant_id: Tenant identifier to check.

        Returns:
            True if tenant exists, False otherwise.
        """

    @abstractmethod
    def get_tenant_stats(self, tenant_id: str) -> TenantStats:
        """Get statistics for a tenant.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            TenantStats with tenant statistics.
        """

    @abstractmethod
    def list_tenants(self) -> list[str]:
        """List all tenants.

        Returns:
            List of tenant identifiers.
        """

    @abstractmethod
    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant and all its data.

        Args:
            tenant_id: Tenant identifier to delete.

        Returns:
            True if tenant was deleted, False if not found.
        """

    @abstractmethod
    def index_documents(
        self,
        documents: list[Document],
        tenant_id: str | None = None,
    ) -> int:
        """Index documents for a tenant.

        Args:
            documents: List of Haystack Documents to index.
            tenant_id: Target tenant. Uses current context if None.

        Returns:
            Number of documents indexed.
        """

    @abstractmethod
    def query(
        self,
        query: str,
        top_k: int = 10,
        tenant_id: str | None = None,
    ) -> list[Document]:
        """Query documents within tenant scope.

        Args:
            query: Query string.
            top_k: Number of results to return.
            tenant_id: Target tenant. Uses current context if None.

        Returns:
            List of retrieved Documents.
        """

    def __enter__(self) -> "BaseMultitenancyPipeline":
        """Enter context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context manager and close connection."""
        self.close()
