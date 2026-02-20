"""Configuration loading and validation for JSON indexing pipelines."""

import os
import re
from typing import Any

import yaml


def _resolve_env_vars(value: Any) -> Any:
    """Resolve environment variables in configuration values.

    Supports both simple ${VAR} and ${VAR:-default} syntax.

    Args:
        value: The value to resolve, can be a string, dict, or list.

    Returns:
        The resolved value with environment variables expanded.
    """
    if isinstance(value, str):
        # Match ${VAR} or ${VAR:-default}
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


def load_config(config_or_path: dict[str, Any] | str) -> dict[str, Any]:
    """Load configuration from YAML file or return dict as-is.

    Args:
        config_or_path: Path to YAML file or configuration dictionary.

    Returns:
        Configuration dictionary with environment variables resolved.

    Raises:
        FileNotFoundError: If config_path does not exist.
    """
    if isinstance(config_or_path, dict):
        return _resolve_env_vars(config_or_path)

    with open(config_or_path) as f:
        config = yaml.safe_load(f)
    return _resolve_env_vars(config)
