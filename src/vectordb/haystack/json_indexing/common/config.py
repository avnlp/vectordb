"""Configuration loading and validation for JSON indexing pipelines."""

import os
import re
from typing import Any

import yaml


def _resolve_env_vars(value: Any) -> Any:
    """Resolve environment variables in configuration values.

    Supports both simple ${VAR} and ${VAR:-default} syntax, including
    multiple substitutions within a single string (e.g., "http://${HOST}:${PORT}").

    Args:
        value: The value to resolve, can be a string, dict, or list.

    Returns:
        The resolved value with environment variables expanded.
    """
    if isinstance(value, str):
        pattern = r"\$\{([^}]+)\}"

        def replacer(match: re.Match[str]) -> str:
            expr = match.group(1)
            if ":-" in expr:
                var, default = expr.split(":-", 1)
                return os.environ.get(var, default)
            return os.environ.get(expr, "")

        return re.sub(pattern, replacer, value)
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
