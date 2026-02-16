"""Pytest configuration for tests."""

import sys
from unittest.mock import MagicMock

import pytest


# Setup mocks for chroma tests BEFORE any imports
def setup_chroma_mocks():
    """Setup chromadb mocks that work across xdist workers."""
    # Mock pysqlite3 and sqlite3
    sys.modules["pysqlite3"] = MagicMock()
    sys.modules["sqlite3"] = MagicMock()

    # Mock weave
    class MockWeaveModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class MockWeave:
        class Model(MockWeaveModel):
            pass

        @staticmethod
        def op():
            def decorator(func):
                return func

            return decorator

        @staticmethod
        def init(project_name, **kwargs):
            pass

    sys.modules["weave"] = MockWeave()

    # Mock chromadb and submodules
    sys.modules["chromadb"] = MagicMock()
    sys.modules["chromadb.api"] = MagicMock()
    sys.modules["chromadb.api.configuration"] = MagicMock()
    sys.modules["chromadb.api.types"] = MagicMock()
    sys.modules["chromadb.utils"] = MagicMock()
    sys.modules["chromadb.utils.embedding_functions"] = MagicMock()


def pytest_configure(config):
    """Register custom markers and setup mocks."""
    config.addinivalue_line(
        "markers", "chroma: Mark test as chroma test (not parallelizable)"
    )

    # Setup chromadb mocks early
    setup_chroma_mocks()


def pytest_collection_modifyitems(config, items):
    """Mark chroma tests to run in a single worker."""
    for item in items:
        if "test_chroma" in str(item.fspath):
            # Force chroma tests into a single xdist group
            item.add_marker(pytest.mark.xdist_group(name="chroma"))
