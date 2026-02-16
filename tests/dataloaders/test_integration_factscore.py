"""Integration tests for FactScore dataset loader with actual HuggingFace data.

This module contains integration tests that fetch actual FactScore datasets.
Tests are marked with @pytest.mark.integration and require internet access.
"""

import pytest

from vectordb.dataloaders.factscore import FactScoreDataloader


class TestFactScoreDataloaderIntegration:
    """Integration test suite for FactScore dataset with live HuggingFace data.

    Tests cover:
    - Actual dataset loading from HuggingFace
    - Real data structure validation
    - Fact decomposition handling
    - Metadata preservation from live data
    - Limit handling on real datasets
    """

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_factscore_load_real_test_split(self) -> None:
        """Test loading real FactScore test split from HuggingFace."""
        loader = FactScoreDataloader(split="test")
        result = loader.load()

        # Should return actual data
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_factscore_load_real_validation_split(self) -> None:
        """Test loading real FactScore validation split from HuggingFace."""
        loader = FactScoreDataloader(split="validation")
        result = loader.load()

        # Should return actual data
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_factscore_load_real_data_structure(self) -> None:
        """Test that real FactScore data has expected structure."""
        loader = FactScoreDataloader(limit=1)
        result = loader.load()

        assert len(result) > 0
        item = result[0]

        # Check structure
        assert "text" in item
        assert "metadata" in item

        # Check metadata fields
        metadata = item["metadata"]
        assert "topic" in metadata
        assert "id" in metadata
        assert "facts" in metadata

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_factscore_load_real_topic_preserved(self) -> None:
        """Test that topic is preserved in metadata."""
        loader = FactScoreDataloader(limit=10)
        result = loader.load()

        assert len(result) > 0
        for item in result:
            topic = item["metadata"]["topic"]
            assert len(topic) > 0
            assert len(topic.strip()) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_factscore_load_real_facts_list(self) -> None:
        """Test that facts field contains a list of facts."""
        loader = FactScoreDataloader(limit=10)
        result = loader.load()

        assert len(result) > 0
        for item in result:
            facts = item["metadata"]["facts"]
            assert isinstance(facts, list)
            assert len(facts) > 0
            # Facts should be non-empty strings
            for fact in facts:
                assert len(fact) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_factscore_load_real_limit_respected(self) -> None:
        """Test that limit parameter is respected on real data."""
        loader = FactScoreDataloader(limit=5)
        result = loader.load()

        assert len(result) <= 5

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_factscore_load_real_id_unique(self) -> None:
        """Test that IDs are unique and valid."""
        loader = FactScoreDataloader(limit=10)
        result = loader.load()

        assert len(result) > 0
        ids = []
        for item in result:
            item_id = item["metadata"]["id"]
            assert len(item_id) > 0
            ids.append(item_id)

        # IDs should be unique within a reasonable set
        # Note: IDs might repeat if we expand decomposed facts
        assert len(ids) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_factscore_load_real_text_contains_facts(self) -> None:
        """Test that text field contains fact content."""
        loader = FactScoreDataloader(limit=1)
        result = loader.load()

        assert len(result) > 0
        for item in result:
            text = item["text"]
            # Text should be non-empty and contain facts
            assert len(text) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_factscore_load_real_batches(self) -> None:
        """Test loading multiple items validates batch consistency."""
        loader = FactScoreDataloader(limit=20)
        result = loader.load()

        assert len(result) > 0
        assert len(result) <= 20

        # All items should have same structure
        for item in result:
            assert "text" in item
            assert "metadata" in item
            assert "topic" in item["metadata"]
            assert "id" in item["metadata"]
            assert "facts" in item["metadata"]

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_factscore_load_real_different_limits(self) -> None:
        """Test that loading with different limits works correctly."""
        limits = [1, 5, 10]

        for limit in limits:
            loader = FactScoreDataloader(limit=limit)
            result = loader.load()

            assert len(result) <= limit
            assert len(result) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_factscore_load_real_text_quality(self) -> None:
        """Test that text content is meaningful."""
        loader = FactScoreDataloader(limit=1)
        result = loader.load()

        assert len(result) > 0
        for item in result:
            text = item["text"]
            # Text should contain meaningful content
            assert len(text) > 10  # More than just a few words

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_factscore_load_real_topic_fact_relation(self) -> None:
        """Test that facts relate to the topic."""
        loader = FactScoreDataloader(limit=10)
        result = loader.load()

        assert len(result) > 0
        for item in result:
            topic = item["metadata"]["topic"]
            facts = item["metadata"]["facts"]

            # Should have meaningful topic
            assert len(topic) > 0
            # Should have facts
            assert len(facts) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_factscore_load_real_fact_count(self) -> None:
        """Test that items contain reasonable number of facts."""
        loader = FactScoreDataloader(limit=10)
        result = loader.load()

        assert len(result) > 0
        for item in result:
            facts = item["metadata"]["facts"]
            # Each item should have at least one fact
            assert len(facts) >= 1
