"""Qdrant-specific tests for query enhancement pipelines.

Common tests are in test_query_enhancement_base.py.
This file contains only Qdrant-specific tests.
"""

import os

import pytest


class TestQdrantQueryEnhancementIntegration:
    """Integration tests for Qdrant query enhancement."""

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not os.getenv("QDRANT_URL"), reason="QDRANT_URL not set for integration test"
    )
    def test_indexing_integration(self) -> None:
        """Integration test for Qdrant indexing pipeline."""
        pytest.skip("Integration test placeholder.")

    @pytest.mark.integration
    @pytest.mark.enable_socket
    @pytest.mark.skipif(
        not os.getenv("QDRANT_URL"), reason="QDRANT_URL not set for integration test"
    )
    def test_search_integration(self) -> None:
        """Integration test for Qdrant search pipeline."""
        pytest.skip("Integration test placeholder.")
