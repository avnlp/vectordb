"""Configuration utilities for vectordb pipelines.

This module provides centralized configuration management for all vectordb pipelines,
including YAML file loading, environment variable resolution, logging setup, and
embedding model alias resolution.

Key Features:
    - YAML Configuration: Load and parse YAML config files with automatic env var
      substitution using ${VAR} or ${VAR:-default} syntax
    - Logging Setup: Initialize loggers from configuration dictionaries
    - Model Aliases: Resolve short embedding model names to full HuggingFace paths
    - Dataset Limits: Predefined indexing and evaluation limits per dataset

Environment Variable Syntax:
    - ${VAR}: Substitute with environment variable VAR, empty string if unset
    - ${VAR:-default}: Substitute with VAR if set, otherwise use 'default'

Usage:
    >>> from vectordb.utils.config import load_config, resolve_embedding_model
    >>> config = load_config("pipeline_config.yaml")
    >>> model = resolve_embedding_model("qwen3")  # Returns full HF path
"""

import logging
import os
import re
from typing import Any

import yaml

from vectordb.utils.logging import LoggerFactory


def resolve_env_vars(value: Any) -> Any:
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
        return {k: resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_env_vars(item) for item in value]
    return value


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from a YAML file with environment variable resolution.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary with environment variables resolved.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the YAML file is malformed.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return resolve_env_vars(config)


def setup_logger(config: dict[str, Any]) -> logging.Logger:
    """Set up a logger based on configuration.

    Args:
        config: Configuration dictionary containing logging settings.

    Returns:
        Configured logger instance.
    """
    logging_config = config.get("logging", {})
    logger_name = logging_config.get("name", "vectordb_pipeline")
    log_level_str = logging_config.get("level", "INFO")
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    factory = LoggerFactory(logger_name, log_level=log_level)
    return factory.get_logger()


# Model name aliases for embedding models
EMBEDDING_MODEL_ALIASES: dict[str, str] = {
    "qwen3": "Qwen/Qwen3-Embedding-0.6B",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}

# Default embedding model used across all pipelines
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def resolve_embedding_model(model_name: str) -> str:
    """Resolve embedding model name from alias or return as-is.

    Args:
        model_name: Model name or alias (e.g., "qwen3", "minilm").

    Returns:
        Full model path suitable for loading.
    """
    return EMBEDDING_MODEL_ALIASES.get(model_name.lower(), model_name)


# Dataset limits for different scenarios
# These define reasonable defaults for indexing and evaluation
DATASET_LIMITS: dict[str, dict[str, int]] = {
    "trivia_qa": {
        "index_limit": 500,  # Large dataset, limit for manageable indexing
        "eval_limit": 100,  # Queries to evaluate
    },
    "ai2_arc": {
        "index_limit": 1000,  # Smaller dataset, can index more
        "eval_limit": 200,
    },
    "akariasai/PopQA": {
        "index_limit": 500,
        "eval_limit": 100,
    },
    "dskar/FActScore": {
        "index_limit": 500,
        "eval_limit": 100,
    },
    "lamini/earnings-calls-qa": {
        "index_limit": 300,  # Financial transcripts are longer
        "eval_limit": 50,
    },
}


def get_dataset_limits(dataset_name: str) -> dict[str, int]:
    """Get default indexing and evaluation limits for a dataset.

    Args:
        dataset_name: Name of the dataset (lowercase).

    Returns:
        Dictionary with 'index_limit' and 'eval_limit' keys.
    """
    return DATASET_LIMITS.get(
        dataset_name.lower(),
        {"index_limit": 500, "eval_limit": 100},
    )
