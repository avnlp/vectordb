"""Milvus-specific tests for query enhancement pipelines.

Common tests are in test_query_enhancement_base.py.
This file contains only Milvus-specific tests.
"""

import os

import pytest


class TestMilvusQueryEnhancementIntegration:
    """Integration tests for Milvus query enhancement."""

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not os.getenv("MILVUS_URI"), reason="MILVUS_URI not set for integration test"
    )
    def test_indexing_integration(self) -> None:
        """Integration test for Milvus indexing pipeline."""
        pytest.skip("Integration test placeholder.")

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not os.getenv("MILVUS_URI"), reason="MILVUS_URI not set for integration test"
    )
    def test_search_integration(self) -> None:
        """Integration test for Milvus search pipeline."""
        pytest.skip("Integration test placeholder.")
