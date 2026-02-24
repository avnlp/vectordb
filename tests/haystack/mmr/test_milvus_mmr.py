"""Unit and integration tests for Haystack Milvus MMR pipelines.

This module provides test coverage for the Maximal Marginal Relevance (MMR)
pipeline implementations with Milvus vector database integration.

Environment Requirements:
    - MILVUS_URI: URI for Milvus server connection (e.g., http://localhost:19530)
    - GROQ_API_KEY: API key for Groq LLM provider (for RAG integration tests)
"""

import os
from typing import Any

from vectordb.haystack.mmr.indexing import MilvusMmrIndexingPipeline
from vectordb.haystack.mmr.search import MilvusMmrSearchPipeline

from .base import MmrTestBase


class TestMilvusMMR(MmrTestBase):
    """Tests for Milvus MMR indexing and search pipelines."""

    db_module = "milvus"
    db_class_name = "MilvusVectorDB"
    integration_env_var = "MILVUS_URI"
    indexing_pipeline_cls = MilvusMmrIndexingPipeline
    search_pipeline_cls = MilvusMmrSearchPipeline

    @property
    def unit_indexing_db_config(self) -> dict[str, Any]:
        """Return Milvus configuration for unit testing indexing pipelines."""
        return {
            "milvus": {
                "uri": "http://localhost:19530",
                "collection_name": "test-collection",
                "dimension": 384,
            }
        }

    @property
    def unit_search_db_config(self) -> dict[str, Any]:
        """Return Milvus configuration for unit testing search pipelines."""
        return {
            "milvus": {
                "uri": "http://localhost:19530",
                "collection_name": "test-collection",
            }
        }

    @property
    def integration_indexing_db_config(self) -> dict[str, Any]:
        """Return Milvus configuration for integration testing indexing pipelines."""
        return {
            "milvus": {
                "uri": os.getenv("MILVUS_URI"),
                "collection_name": "mmr_integration_test",
                "dimension": 384,
                "recreate": True,
            }
        }

    @property
    def integration_search_db_config(self) -> dict[str, Any]:
        """Return Milvus configuration for integration testing search pipelines."""
        return {
            "milvus": {
                "uri": os.getenv("MILVUS_URI"),
                "collection_name": "mmr_integration_test",
            }
        }
