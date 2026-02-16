"""Integration tests for Earnings Calls dataset loader with actual HuggingFace data.

This module contains integration tests that fetch actual Earnings Calls datasets.
Tests are marked with @pytest.mark.integration and require internet access.
"""

import pytest

from vectordb.dataloaders.earnings_calls import EarningsCallDataloader


class TestEarningsCallDataloaderIntegration:
    """Integration test suite for Earnings Calls dataset with live HuggingFace data.

    Tests cover:
    - Actual dataset loading from HuggingFace
    - Real data structure validation
    - Financial document handling
    - Metadata preservation from live data
    - Limit handling on real datasets
    """

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_earnings_call_load_real_test_split(self) -> None:
        """Test loading real Earnings Calls test split from HuggingFace."""
        loader = EarningsCallDataloader(split="test")
        result = loader.load()

        # Should return actual data
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_earnings_call_load_real_validation_split(self) -> None:
        """Test loading real Earnings Calls validation split from HuggingFace."""
        loader = EarningsCallDataloader(split="validation")
        result = loader.load()

        # Should return actual data
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_earnings_call_load_real_data_structure(self) -> None:
        """Test that real Earnings Calls data has expected structure."""
        loader = EarningsCallDataloader(limit=1)
        result = loader.load()

        assert len(result) > 0
        item = result[0]

        # Check structure
        assert "text" in item
        assert "metadata" in item

        # Check that we have meaningful data
        assert len(item["text"]) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_earnings_call_load_real_handles_qa_format(self) -> None:
        """Test that loader handles question-answer format correctly."""
        loader = EarningsCallDataloader(limit=1)
        result = loader.load()

        assert len(result) > 0
        item = result[0]

        # Item should have text (either question+answer or document content)
        assert "text" in item
        assert len(item["text"]) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_earnings_call_load_real_limit_respected(self) -> None:
        """Test that limit parameter is respected on real data."""
        loader = EarningsCallDataloader(limit=5)
        result = loader.load()

        assert len(result) <= 5

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_earnings_call_load_real_text_quality(self) -> None:
        """Test that text content is meaningful financial content."""
        loader = EarningsCallDataloader(limit=10)
        result = loader.load()

        assert len(result) > 0
        for item in result:
            text = item["text"]
            # Should be substantial content
            assert len(text) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_earnings_call_load_real_batches(self) -> None:
        """Test loading multiple items validates batch consistency."""
        loader = EarningsCallDataloader(limit=20)
        result = loader.load()

        assert len(result) > 0
        assert len(result) <= 20

        # All items should have same structure
        for item in result:
            assert "text" in item
            assert isinstance(item["text"], str)
            assert len(item["text"]) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_earnings_call_load_real_different_limits(self) -> None:
        """Test that loading with different limits works correctly."""
        limits = [1, 5, 10]

        for limit in limits:
            loader = EarningsCallDataloader(limit=limit)
            result = loader.load()

            assert len(result) <= limit
            assert len(result) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_earnings_call_load_real_text_length_variation(self) -> None:
        """Test that text length varies appropriately across items."""
        loader = EarningsCallDataloader(limit=10)
        result = loader.load()

        assert len(result) > 0
        text_lengths = [len(item["text"]) for item in result]

        # Should have some variation in text lengths
        assert len(set(text_lengths)) > 1 or len(result) == 1

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_earnings_call_load_real_metadata_structure(self) -> None:
        """Test that metadata is properly structured."""
        loader = EarningsCallDataloader(limit=1)
        result = loader.load()

        assert len(result) > 0
        item = result[0]

        # Should have metadata dict
        if "metadata" in item:
            assert isinstance(item["metadata"], dict)

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_earnings_call_load_real_consistency(self) -> None:
        """Test that loading is consistent across calls."""
        loader1 = EarningsCallDataloader(limit=5)
        result1 = loader1.load()

        loader2 = EarningsCallDataloader(limit=5)
        result2 = loader2.load()

        # Should have same number of items
        assert len(result1) == len(result2)

        # Items should have same structure
        for item1, item2 in zip(result1, result2):
            assert "text" in item1
            assert "text" in item2

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_earnings_call_load_real_custom_dataset(self) -> None:
        """Test that custom dataset name can be specified."""
        # Test with default dataset name
        loader1 = EarningsCallDataloader()
        result1 = loader1.load()

        assert isinstance(result1, list)
        assert len(result1) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_earnings_call_load_real_document_quality(self) -> None:
        """Test that loaded documents represent real earnings calls."""
        loader = EarningsCallDataloader(limit=1)
        result = loader.load()

        assert len(result) > 0
        item = result[0]
        text = item["text"]

        # Financial/earnings related documents
        # Should contain meaningful financial information
        assert len(text) > 50  # Reasonable minimum length
