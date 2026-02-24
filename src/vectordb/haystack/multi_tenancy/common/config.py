"""Configuration loading and validation for multi-tenancy pipelines."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml


def resolve_env_vars(value: Any) -> Any:
    """Resolve environment variables in configuration values.

    Supports both simple ${VAR} and ${VAR:-default} syntax.
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
        return {k: resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_env_vars(item) for item in value]
    return value


def load_config(config_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Load and validate multi-tenancy configuration.

    Args:
        config_or_path: Either a config dict or path to YAML file.

    Returns:
        Validated configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If config is invalid.
    """
    if isinstance(config_or_path, dict):
        return resolve_env_vars(config_or_path)

    path = Path(config_or_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_or_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    return resolve_env_vars(config)


def get_database_type(config: dict[str, Any]) -> str:
    """Extract database type from config.

    Args:
        config: Configuration dictionary.

    Returns:
        Database type string (milvus, weaviate, pinecone, qdrant, chroma).
    """
    return config.get("database", {}).get("type", "milvus").lower()
