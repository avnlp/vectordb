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

    # Mock chromadb and submodules using a single parent mock so that
    # sys.modules entries and attribute navigation resolve to the same objects.
    # This ensures patch("vectordb.databases.chroma.chromadb.X.Y") targets
    # the same mock that "from chromadb.X import Y" resolves to.
    chromadb_mock = MagicMock()
    sys.modules["chromadb"] = chromadb_mock
    sys.modules["chromadb.api"] = chromadb_mock.api
    sys.modules["chromadb.api.configuration"] = chromadb_mock.api.configuration
    sys.modules["chromadb.api.types"] = chromadb_mock.api.types
    sys.modules["chromadb.config"] = chromadb_mock.config
    sys.modules["chromadb.execution"] = chromadb_mock.execution
    sys.modules["chromadb.execution.expression"] = chromadb_mock.execution.expression
    sys.modules["chromadb.utils"] = chromadb_mock.utils
    sys.modules["chromadb.utils.embedding_functions"] = (
        chromadb_mock.utils.embedding_functions
    )


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
