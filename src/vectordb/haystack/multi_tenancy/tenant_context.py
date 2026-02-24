"""Tenant context manager for multi-tenancy pipelines.

Provides immutable tenant context for pipeline operations with support for
environment variable and configuration-based tenant resolution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


__all__ = [
    "TenantContext",
    "TenantContextError",
]


class TenantContextError(ValueError):
    """Raised when tenant context cannot be resolved."""


@dataclass(frozen=True)
class TenantContext:
    """Immutable tenant context for pipeline operations.

    Provides a consistent way to identify and manage tenant context across
    all multi-tenancy pipeline operations. Supports creation from environment
    variables or YAML configuration.

    Attributes:
        tenant_id: Unique tenant identifier (required).
        tenant_name: Human-readable tenant name (optional).
        metadata: Additional tenant metadata for auditing/filtering.

    Example:
        >>> # From environment variable
        >>> os.environ["TENANT_ID"] = "company_abc"
        >>> tenant = TenantContext.from_environment()
        >>> tenant.tenant_id
        'company_abc'

        >>> # From config
        >>> config = {"tenant": {"id": "org_123", "name": "Acme Corp"}}
        >>> tenant = TenantContext.from_config(config)
        >>> tenant.tenant_id
        'org_123'

        >>> # Direct creation
        >>> tenant = TenantContext(tenant_id="user_456", tenant_name="John Doe")
    """

    tenant_id: str
    tenant_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate tenant_id is not empty."""
        if not self.tenant_id:
            raise TenantContextError("tenant_id cannot be empty")
        if not isinstance(self.tenant_id, str):
            raise TenantContextError("tenant_id must be a string")

    @classmethod
    def from_environment(
        cls,
        env_var: str = "TENANT_ID",
        name_env_var: str | None = "TENANT_NAME",
    ) -> "TenantContext":
        """Create TenantContext from environment variables.

        Resolution order for tenant_id:
        1. TENANT_ID environment variable (or custom env_var)

        Args:
            env_var: Environment variable name for tenant ID.
            name_env_var: Environment variable name for tenant name (optional).

        Returns:
            TenantContext with values from environment.

        Raises:
            TenantContextError: If TENANT_ID environment variable is not set.

        Example:
            >>> os.environ["TENANT_ID"] = "company_abc"
            >>> os.environ["TENANT_NAME"] = "Acme Corporation"
            >>> tenant = TenantContext.from_environment()
            >>> tenant.tenant_id
            'company_abc'
            >>> tenant.tenant_name
            'Acme Corporation'
        """
        tenant_id = os.environ.get(env_var)
        if not tenant_id:
            raise TenantContextError(
                f"Environment variable '{env_var}' is not set. "
                "Set TENANT_ID or pass tenant_context to pipeline."
            )

        tenant_name = None
        if name_env_var:
            tenant_name = os.environ.get(name_env_var)

        return cls(tenant_id=tenant_id, tenant_name=tenant_name)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TenantContext":
        """Create TenantContext from YAML config section.

        Expects config structure:
        ```yaml
        tenant:
          id: "company_abc"
          name: "Acme Corporation"  # optional
          metadata:                  # optional
            department: "engineering"
        ```

        Args:
            config: Configuration dictionary containing tenant section.

        Returns:
            TenantContext with values from config.

        Raises:
            TenantContextError: If tenant.id is not found in config.

        Example:
            >>> config = {
            ...     "tenant": {
            ...         "id": "org_123",
            ...         "name": "Acme Corp",
            ...         "metadata": {"tier": "enterprise"}
            ...     }
            ... }
            >>> tenant = TenantContext.from_config(config)
            >>> tenant.tenant_id
            'org_123'
            >>> tenant.metadata
            {'tier': 'enterprise'}
        """
        tenant_config = config.get("tenant", {})
        tenant_id = tenant_config.get("id")

        if not tenant_id:
            raise TenantContextError(
                "Configuration missing 'tenant.id'. "
                "Add tenant.id to config or set TENANT_ID environment variable."
            )

        return cls(
            tenant_id=tenant_id,
            tenant_name=tenant_config.get("name"),
            metadata=tenant_config.get("metadata", {}),
        )

    @classmethod
    def resolve(
        cls,
        tenant_context: "TenantContext | None" = None,
        config: dict[str, Any] | None = None,
    ) -> "TenantContext":
        """Resolve tenant context from multiple sources.

        Resolution order (first match wins):
        1. Explicit tenant_context parameter
        2. TENANT_ID environment variable
        3. tenant.id in config

        Args:
            tenant_context: Explicit tenant context (highest priority).
            config: Configuration dictionary with tenant section.

        Returns:
            Resolved TenantContext.

        Raises:
            TenantContextError: If tenant cannot be resolved from any source.

        Example:
            >>> # With explicit context
            >>> explicit = TenantContext(tenant_id="explicit")
            >>> resolved = TenantContext.resolve(tenant_context=explicit)
            >>> resolved.tenant_id
            'explicit'

            >>> # From environment
            >>> os.environ["TENANT_ID"] = "env_tenant"
            >>> resolved = TenantContext.resolve()
            >>> resolved.tenant_id
            'env_tenant'
        """
        if tenant_context is not None:
            return tenant_context

        tenant_id_env = os.environ.get("TENANT_ID")
        if tenant_id_env:
            return cls.from_environment()

        if config is not None and "tenant" in config:
            return cls.from_config(config)

        raise TenantContextError(
            "Cannot resolve tenant context. Provide one of:\n"
            "  1. tenant_context parameter\n"
            "  2. TENANT_ID environment variable\n"
            "  3. tenant.id in YAML config"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert tenant context to dictionary.

        Returns:
            Dictionary representation of tenant context.
        """
        return {
            "tenant_id": self.tenant_id,
            "tenant_name": self.tenant_name,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        """Return string representation."""
        if self.tenant_name:
            return f"TenantContext({self.tenant_id}, name={self.tenant_name})"
        return f"TenantContext({self.tenant_id})"
