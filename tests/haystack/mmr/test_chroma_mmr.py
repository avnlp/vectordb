"""Unit and integration tests for Chroma MMR (Maximal Marginal Relevance) pipelines.

This module provides test coverage for the Haystack-based Chroma integration
with MMR reranking capabilities.

Integration Tests:
    Tests marked with @pytest.mark.integration require:
    - CHROMA_HOST environment variable set
    - GROQ_API_KEY for RAG integration tests
"""

from typing import Any

from vectordb.haystack.mmr.indexing import ChromaMmrIndexingPipeline
from vectordb.haystack.mmr.search import ChromaMmrSearchPipeline

from .base import MmrTestBase


class TestChromaMMR(MmrTestBase):
    """Tests for Chroma MMR indexing and search pipelines."""

    db_module = "chroma"
    db_class_name = "ChromaVectorDB"
    integration_env_var = "CHROMA_HOST"
    indexing_pipeline_cls = ChromaMmrIndexingPipeline
    search_pipeline_cls = ChromaMmrSearchPipeline

    @property
    def unit_indexing_db_config(self) -> dict[str, Any]:
        """Return Chroma configuration for unit testing indexing pipelines."""
        return {
            "chroma": {
                "persist_directory": "/tmp/chroma_test",
                "collection_name": "test-collection",
                "dimension": 384,
            }
        }

    @property
    def unit_search_db_config(self) -> dict[str, Any]:
        """Return Chroma configuration for unit testing search pipelines."""
        return {
            "chroma": {
                "persist_directory": "/tmp/chroma_test",
                "collection_name": "test-collection",
            }
        }

    @property
    def integration_indexing_db_config(self) -> dict[str, Any]:
        """Return Chroma configuration for integration testing indexing pipelines."""
        return {
            "chroma": {
                "persist_directory": "/tmp/chroma_integration_test",
                "collection_name": "mmr-integration-test",
                "dimension": 384,
                "recreate": True,
            }
        }

    @property
    def integration_search_db_config(self) -> dict[str, Any]:
        """Return Chroma configuration for integration testing search pipelines."""
        return {
            "chroma": {
                "persist_directory": "/tmp/chroma_integration_test",
                "collection_name": "mmr-integration-test",
            }
        }
