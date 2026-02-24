"""Unit and integration tests for Weaviate MMR pipelines.

This module tests the Maximal Marginal Relevance (MMR) functionality within
the Haystack Weaviate integration.

Environment Variables:
    - WEAVIATE_URL: URL of the Weaviate instance (required for integration tests)
    - GROQ_API_KEY: API key for Groq LLM service (required for RAG integration tests)
"""

import os
from typing import Any

from vectordb.haystack.mmr.indexing import WeaviateMmrIndexingPipeline
from vectordb.haystack.mmr.search import WeaviateMmrSearchPipeline

from .base import MmrTestBase


class TestWeaviateMMR(MmrTestBase):
    """Tests for Weaviate MMR indexing and search pipelines."""

    db_module = "weaviate"
    db_class_name = "WeaviateVectorDB"
    integration_env_var = "WEAVIATE_URL"
    indexing_pipeline_cls = WeaviateMmrIndexingPipeline
    search_pipeline_cls = WeaviateMmrSearchPipeline

    @property
    def unit_indexing_db_config(self) -> dict[str, Any]:
        """Return Weaviate configuration for unit testing indexing pipelines."""
        return {
            "weaviate": {
                "url": "http://localhost:8080",
                "collection_name": "test-collection",
                "dimension": 384,
            }
        }

    @property
    def unit_search_db_config(self) -> dict[str, Any]:
        """Return Weaviate configuration for unit testing search pipelines."""
        return {
            "weaviate": {
                "url": "http://localhost:8080",
                "collection_name": "test-collection",
            }
        }

    @property
    def integration_indexing_db_config(self) -> dict[str, Any]:
        """Return Weaviate configuration for integration testing indexing pipelines."""
        return {
            "weaviate": {
                "url": os.getenv("WEAVIATE_URL"),
                "collection_name": "mmr_integration_test",
                "dimension": 384,
                "recreate": True,
            }
        }

    @property
    def integration_search_db_config(self) -> dict[str, Any]:
        """Return Weaviate configuration for integration testing search pipelines."""
        return {
            "weaviate": {
                "url": os.getenv("WEAVIATE_URL"),
                "collection_name": "mmr_integration_test",
            }
        }
