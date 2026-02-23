"""Shared fixtures for namespace and collection management tests.

This module provides pytest fixtures specifically designed for testing
namespace operations across vector databases. These fixtures support
configuration file creation for testing namespace CRUD operations
and document scoping.

Fixtures:
    temp_config_file: Creates a temporary YAML configuration file path
        for testing configuration loading and validation.

Note:
    Namespace tests often require dynamic configuration generation to
    test various namespace scenarios without persisting test data.
"""

from pathlib import Path

import pytest


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Create a temporary config file path for testing.

    Args:
        tmp_path: Pytest built-in fixture providing a temporary directory.

    Returns:
        Path object pointing to a test configuration file in the
        temporary directory.
    """
    return tmp_path / "test_config.yaml"
