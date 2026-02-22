"""Configuration loading utilities for namespace pipelines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file with environment variable resolution.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Configuration dictionary with env vars resolved.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    return resolve_env_vars(config)


def resolve_env_vars(config: dict[str, Any]) -> dict[str, Any]:
    """Recursively resolve ${ENV_VAR} patterns in config values.

    Args:
        config: Configuration dictionary.

    Returns:
        Configuration with environment variables resolved.
    """
    resolved = {}
    for key, value in config.items():
        if isinstance(value, dict):
            resolved[key] = resolve_env_vars(value)
        elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            resolved[key] = os.environ.get(env_var, "")
        else:
            resolved[key] = value
    return resolved
