"""Unit and integration tests for Pinecone MMR (Maximal Marginal Relevance) pipelines.

This module tests the Haystack integration with Pinecone vector database for
MMR-based retrieval and RAG (Retrieval-Augmented Generation) pipelines.

Configuration:
    Tests require PINECONE_API_KEY environment variable for integration tests.
    Optional GROQ_API_KEY for RAG integration tests.

Example:
    Run unit tests:
        pytest tests/haystack/mmr/test_pinecone_mmr.py -v -m "not integration"

    Run integration tests:
        PINECONE_API_KEY=xxx pytest \\
            tests/haystack/mmr/test_pinecone_mmr.py -v -m integration
"""

import os
from typing import Any

from vectordb.haystack.mmr.indexing import PineconeMmrIndexingPipeline
from vectordb.haystack.mmr.search import PineconeMmrSearchPipeline

from .base import MmrTestBase


class TestPineconeMMR(MmrTestBase):
    """Tests for Pinecone MMR indexing and search pipelines."""

    db_module = "pinecone"
    db_class_name = "PineconeVectorDB"
    integration_env_var = "PINECONE_API_KEY"
    indexing_pipeline_cls = PineconeMmrIndexingPipeline
    search_pipeline_cls = PineconeMmrSearchPipeline

    @property
    def unit_indexing_db_config(self) -> dict[str, Any]:
        """Return Pinecone configuration for unit testing indexing pipelines."""
        return {
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-index",
                "namespace": "test",
                "dimension": 384,
            }
        }

    @property
    def unit_search_db_config(self) -> dict[str, Any]:
        """Return Pinecone configuration for unit testing search pipelines."""
        return {
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-index",
                "namespace": "test",
            }
        }

    @property
    def integration_indexing_db_config(self) -> dict[str, Any]:
        """Return Pinecone configuration for integration testing indexing pipelines."""
        return {
            "pinecone": {
                "api_key": os.getenv("PINECONE_API_KEY"),
                "index_name": "mmr-integration-test",
                "namespace": "integration_test",
                "dimension": 384,
                "recreate": True,
            }
        }

    @property
    def integration_search_db_config(self) -> dict[str, Any]:
        """Return Pinecone configuration for integration testing search pipelines."""
        return {
            "pinecone": {
                "api_key": os.getenv("PINECONE_API_KEY"),
                "index_name": "mmr-integration-test",
                "namespace": "integration_test",
            }
        }
