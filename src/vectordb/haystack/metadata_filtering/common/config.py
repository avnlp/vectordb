"""Configuration loading and validation for metadata filtering pipelines.

Handles YAML configuration loading with environment variable resolution.
"""

import os
from typing import Any

import yaml


__all__ = ["load_metadata_filtering_config", "resolve_env_vars"]


def resolve_env_vars(value: Any) -> Any:
    """Recursively resolve environment variables in config values.

    Supports ${VAR_NAME} and ${VAR_NAME:-default} syntax.

    Args:
        value: Configuration value (may contain env var references).

    Returns:
        Value with environment variables resolved.

    Raises:
        ValueError: If required environment variable is not set.
    """
    if isinstance(value, str):
        # Handle ${VAR_NAME:-default} pattern
        import re

        pattern = r"\$\{([^}:]+)(?::-(.*?))?\}"

        def replacer(match: Any) -> str:
            var_name = match.group(1)
            default = match.group(2)
            return os.environ.get(var_name, default or "")

        resolved = re.sub(pattern, replacer, value)

        if "${" in resolved:
            raise ValueError(f"Unresolved environment variables in: {resolved}")

        return resolved

    if isinstance(value, dict):
        return {k: resolve_env_vars(v) for k, v in value.items()}

    if isinstance(value, list):
        return [resolve_env_vars(v) for v in value]

    return value


def load_metadata_filtering_config(config_or_path: str | dict) -> dict[str, Any]:
    """Load and parse metadata filtering configuration.

    Args:
        config_or_path: Path to YAML file or dict with config.

    Returns:
        Parsed configuration dictionary with env vars resolved.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If YAML is invalid.
    """
    if isinstance(config_or_path, dict):
        config = config_or_path
    else:
        if not os.path.exists(config_or_path):
            raise FileNotFoundError(f"Config file not found: {config_or_path}")

        with open(config_or_path) as f:
            config = yaml.safe_load(f)

    if config is None:
        config = {}

    # Resolve environment variables in all values
    return resolve_env_vars(config)
