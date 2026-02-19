"""Configuration management for vector database pipelines.

This module provides utilities for loading, validating, and resolving pipeline
configurations from YAML files or dictionaries. It handles environment variable
substitution for secrets and deployment-specific values.

Key Features:
    - YAML Loading: Parse configuration files with proper error handling
    - Environment Variables: Substitute ${VAR} and ${VAR:-default} patterns
    - Validation: Ensure required sections exist for each database type
    - Recursive Resolution: Handle nested dicts and lists with env vars

Environment Variable Syntax:
    - ${VAR}: Substitute with environment variable, empty string if unset
    - ${VAR:-default}: Use VAR if set, otherwise use the default value

Required Config Sections by Database:
    - pinecone: api_key, index_name
    - weaviate: cluster_url, api_key
    - chroma: path or host/port
    - milvus: uri, collection_name
    - qdrant: url, collection_name

Usage:
    >>> from vectordb.utils.config_loader import ConfigLoader
    >>> config = ConfigLoader.load("pipeline_config.yaml")
    >>> ConfigLoader.validate(config, "pinecone")
"""

import os
import re
from pathlib import Path
from typing import Any


class ConfigLoader:
    """Handles loading and validating pipeline configurations.

    Supports environment variable substitution: ${VAR} or ${VAR:-default}
    """

    @classmethod
    def load(cls, config_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
        """Load and resolve configuration from dict or YAML file.

        Args:
            config_or_path: Configuration dict or path to YAML file.

        Returns:
            Resolved configuration dictionary.
        """
        if isinstance(config_or_path, dict):
            return cls._resolve_env_vars(config_or_path)

        path = Path(config_or_path)
        import yaml

        with open(path) as f:
            config = yaml.safe_load(f)
        return cls._resolve_env_vars(config)

    @classmethod
    def validate(cls, config: dict[str, Any], db_type: str) -> None:
        """Validate required config sections exist.

        Args:
            config: Configuration dictionary.
            db_type: Database type (pinecone, weaviate, chroma, milvus, qdrant).

        Raises:
            ValueError: If required sections are missing.
        """
        required = ["dataloader", "embeddings", db_type]
        missing = [k for k in required if k not in config]
        if missing:
            msg = f"Missing required config sections: {missing}"
            raise ValueError(msg)

    @classmethod
    def _resolve_env_vars(cls, value: Any) -> Any:
        """Recursively resolve ${VAR} and ${VAR:-default} patterns."""
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
            return {k: cls._resolve_env_vars(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._resolve_env_vars(item) for item in value]
        return value
