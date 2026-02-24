"""Test suite for Haystack Qdrant MMR (Maximal Marginal Relevance) pipelines.

This module provides test coverage for the Qdrant MMR indexing and search
functionality using the Haystack 2.0 framework.

Environment Variables:
    QDRANT_URL: Qdrant server URL for integration tests
    QDRANT_API_KEY: API key for Qdrant authentication (optional)
    GROQ_API_KEY: API key for Groq LLM provider (RAG tests only)
"""

import os
from typing import Any

from vectordb.haystack.mmr.indexing import QdrantMmrIndexingPipeline
from vectordb.haystack.mmr.search import QdrantMmrSearchPipeline

from .base import MmrTestBase


class TestQdrantMMR(MmrTestBase):
    """Tests for Qdrant MMR indexing and search pipelines."""

    db_module = "qdrant"
    db_class_name = "QdrantVectorDB"
    integration_env_var = "QDRANT_URL"
    indexing_pipeline_cls = QdrantMmrIndexingPipeline
    search_pipeline_cls = QdrantMmrSearchPipeline

    @property
    def unit_indexing_db_config(self) -> dict[str, Any]:
        """Return Qdrant configuration for unit testing indexing pipelines."""
        return {
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "test-key",
                "collection_name": "test-collection",
                "dimension": 384,
            }
        }

    @property
    def unit_search_db_config(self) -> dict[str, Any]:
        """Return Qdrant configuration for unit testing search pipelines."""
        return {
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "test-key",
                "collection_name": "test-collection",
            }
        }

    @property
    def integration_indexing_db_config(self) -> dict[str, Any]:
        """Return Qdrant configuration for integration testing indexing pipelines."""
        return {
            "qdrant": {
                "url": os.getenv("QDRANT_URL"),
                "api_key": os.getenv("QDRANT_API_KEY"),
                "collection_name": "mmr_integration_test",
                "dimension": 384,
                "recreate": True,
            }
        }

    @property
    def integration_search_db_config(self) -> dict[str, Any]:
        """Return Qdrant configuration for integration testing search pipelines."""
        return {
            "qdrant": {
                "url": os.getenv("QDRANT_URL"),
                "api_key": os.getenv("QDRANT_API_KEY"),
                "collection_name": "mmr_integration_test",
            }
        }
