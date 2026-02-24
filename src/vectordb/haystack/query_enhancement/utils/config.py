"""Configuration loading and validation for query enhancement pipelines."""

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Load and validate query enhancement configuration.

    Args:
        config_or_path: Either a config dict or path to YAML file.

    Returns:
        Validated configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If config is invalid.
    """
    if isinstance(config_or_path, dict):
        config = config_or_path
    else:
        path = Path(config_or_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            config = yaml.safe_load(f)

    return resolve_env_vars(config)


def resolve_env_vars(config: Any) -> Any:
    """Recursively resolve ${ENV_VAR} patterns in configuration."""
    if isinstance(config, dict):
        return {k: resolve_env_vars(v) for k, v in config.items()}
    if isinstance(config, list):
        return [resolve_env_vars(item) for item in config]
    if isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        env_var = config[2:-1]
        return os.getenv(env_var, "")
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate required configuration fields.

    Raises:
        ValueError: If required fields are missing.
    """
    required_sections = ["dataloader", "embeddings", "query_enhancement"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
