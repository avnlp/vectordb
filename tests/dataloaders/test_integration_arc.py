"""Integration tests for ARC dataset loader with actual HuggingFace data.

This module contains integration tests that fetch actual datasets from HuggingFace.
Tests are marked with @pytest.mark.integration and require internet access.
"""

import pytest

from vectordb.dataloaders.arc import ARCDataloader


class TestARCDataloaderIntegration:
    """Integration test suite for ARC dataset with live HuggingFace data.

    Tests cover:
    - Actual dataset loading from HuggingFace
    - Real data structure validation
    - Question and choice formatting
    - Metadata preservation from live data
    - Limit handling on real datasets
    """

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_arc_load_real_validation_split(self) -> None:
        """Test loading real ARC validation split from HuggingFace."""
        loader = ARCDataloader(split="validation")
        result = loader.load()

        # Should return actual data
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_arc_load_real_test_split(self) -> None:
        """Test loading real ARC test split from HuggingFace."""
        loader = ARCDataloader(split="test")
        result = loader.load()

        # Should return actual data
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_arc_load_real_data_structure(self) -> None:
        """Test that real ARC data has expected structure."""
        loader = ARCDataloader(limit=1)
        result = loader.load()

        assert len(result) > 0
        item = result[0]

        # Check structure
        assert "text" in item
        assert "metadata" in item

        # Check metadata fields
        metadata = item["metadata"]
        assert "question" in metadata
        assert "id" in metadata
        assert "answer_key" in metadata

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_arc_load_real_text_contains_question(self) -> None:
        """Test that loaded text contains the question."""
        loader = ARCDataloader(limit=1)
        result = loader.load()

        assert len(result) > 0
        item = result[0]
        question = item["metadata"]["question"]

        # Text should contain the question
        assert question in item["text"]

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_arc_load_real_text_contains_choices(self) -> None:
        """Test that loaded text includes choice options."""
        loader = ARCDataloader(limit=1)
        result = loader.load()

        assert len(result) > 0
        item = result[0]

        # Text should contain choices section
        assert "Choices:" in item["text"]
        # Should have answer options (A, B, C, D)
        text = item["text"]
        assert any(f"{opt})" in text for opt in ["A", "B", "C", "D"])

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_arc_load_real_answer_key_valid(self) -> None:
        """Test that answer keys are valid (A, B, C, or D)."""
        loader = ARCDataloader(limit=10)
        result = loader.load()

        assert len(result) > 0
        for item in result:
            answer_key = item["metadata"]["answer_key"]
            assert answer_key in ["A", "B", "C", "D"]

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_arc_load_real_limit_respected(self) -> None:
        """Test that limit parameter is respected on real data."""
        loader = ARCDataloader(limit=5)
        result = loader.load()

        assert len(result) <= 5

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_arc_load_real_no_empty_questions(self) -> None:
        """Test that loaded questions are not empty."""
        loader = ARCDataloader(limit=10)
        result = loader.load()

        assert len(result) > 0
        for item in result:
            question = item["metadata"]["question"]
            assert len(question) > 0
            assert len(question.strip()) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_arc_load_real_challenge_config(self) -> None:
        """Test loading ARC-Challenge configuration."""
        loader = ARCDataloader(config="ARC-Challenge")
        result = loader.load()

        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_arc_load_real_easy_config(self) -> None:
        """Test loading ARC-Easy configuration."""
        loader = ARCDataloader(config="ARC-Easy")
        result = loader.load()

        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_arc_load_real_question_formats_correctly(self) -> None:
        """Test that questions are formatted correctly with options."""
        loader = ARCDataloader(limit=1)
        result = loader.load()

        assert len(result) > 0
        item = result[0]
        text = item["text"]

        # Should have question, choices header, and formatted options
        assert item["metadata"]["question"] in text
        assert "Choices:" in text

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_arc_load_real_batches(self) -> None:
        """Test loading multiple items validates batch consistency."""
        loader = ARCDataloader(limit=20)
        result = loader.load()

        assert len(result) > 0
        assert len(result) <= 20

        # All items should have same structure
        for item in result:
            assert "text" in item
            assert "metadata" in item
            assert "question" in item["metadata"]
            assert "answer_key" in item["metadata"]

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_arc_load_real_different_limits(self) -> None:
        """Test that loading with different limits works correctly."""
        limits = [1, 5, 10]

        for limit in limits:
            loader = ARCDataloader(limit=limit)
            result = loader.load()

            assert len(result) <= limit
            assert len(result) > 0
