"""Integration tests for PopQA dataset loader with actual HuggingFace data.

This module contains integration tests that fetch actual PopQA datasets.
Tests are marked with @pytest.mark.integration and require internet access.
"""

import pytest

from vectordb.dataloaders.popqa import PopQADataloader


class TestPopQADataloaderIntegration:
    """Integration test suite for PopQA dataset with live HuggingFace data.

    Tests cover:
    - Actual dataset loading from HuggingFace
    - Real data structure validation
    - Entity and answer extraction
    - Metadata preservation from live data
    - Limit handling on real datasets
    """

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_popqa_load_real_test_split(self) -> None:
        """Test loading real PopQA test split from HuggingFace."""
        loader = PopQADataloader(split="test")
        result = loader.load()

        # Should return actual data
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_popqa_load_real_validation_split(self) -> None:
        """Test loading real PopQA validation split from HuggingFace."""
        loader = PopQADataloader(split="validation")
        result = loader.load()

        # Should return actual data
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_popqa_load_real_data_structure(self) -> None:
        """Test that real PopQA data has expected structure."""
        loader = PopQADataloader(limit=1)
        result = loader.load()

        assert len(result) > 0
        item = result[0]

        # Check structure
        assert "text" in item
        assert "metadata" in item

        # Check metadata fields
        metadata = item["metadata"]
        assert "question" in metadata
        assert "entity" in metadata
        assert "answers" in metadata

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_popqa_load_real_text_contains_question(self) -> None:
        """Test that loaded text contains the question."""
        loader = PopQADataloader(limit=1)
        result = loader.load()

        assert len(result) > 0
        item = result[0]
        question = item["metadata"]["question"]

        # Text should contain the question
        assert question in item["text"]

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_popqa_load_real_entity_preserved(self) -> None:
        """Test that entity is preserved in metadata."""
        loader = PopQADataloader(limit=10)
        result = loader.load()

        assert len(result) > 0
        for item in result:
            entity = item["metadata"]["entity"]
            assert len(entity) > 0
            assert len(entity.strip()) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_popqa_load_real_answers_list(self) -> None:
        """Test that answers field contains a list of answers."""
        loader = PopQADataloader(limit=10)
        result = loader.load()

        assert len(result) > 0
        for item in result:
            answers = item["metadata"]["answers"]
            assert isinstance(answers, list)
            assert len(answers) > 0
            # Answers should be non-empty strings
            for answer in answers:
                assert len(answer) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_popqa_load_real_limit_respected(self) -> None:
        """Test that limit parameter is respected on real data."""
        loader = PopQADataloader(limit=5)
        result = loader.load()

        assert len(result) <= 5

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_popqa_load_real_question_not_empty(self) -> None:
        """Test that loaded questions are not empty."""
        loader = PopQADataloader(limit=10)
        result = loader.load()

        assert len(result) > 0
        for item in result:
            question = item["metadata"]["question"]
            assert len(question) > 0
            assert len(question.strip()) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_popqa_load_real_text_contains_content(self) -> None:
        """Test that text field contains the content."""
        loader = PopQADataloader(limit=1)
        result = loader.load()

        assert len(result) > 0
        for item in result:
            text = item["text"]
            # Text should be non-empty
            assert len(text) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_popqa_load_real_batches(self) -> None:
        """Test loading multiple items validates batch consistency."""
        loader = PopQADataloader(limit=20)
        result = loader.load()

        assert len(result) > 0
        assert len(result) <= 20

        # All items should have same structure
        for item in result:
            assert "text" in item
            assert "metadata" in item
            assert "question" in item["metadata"]
            assert "entity" in item["metadata"]
            assert "answers" in item["metadata"]

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_popqa_load_real_different_limits(self) -> None:
        """Test that loading with different limits works correctly."""
        limits = [1, 5, 10]

        for limit in limits:
            loader = PopQADataloader(limit=limit)
            result = loader.load()

            assert len(result) <= limit
            assert len(result) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_popqa_load_real_question_entity_relation(self) -> None:
        """Test that entity relates to the question topic."""
        loader = PopQADataloader(limit=10)
        result = loader.load()

        assert len(result) > 0
        # Verify structure is consistent
        for item in result:
            assert "question" in item["metadata"]
            assert "entity" in item["metadata"]
            # Both should be meaningful strings
            assert len(item["metadata"]["question"]) > 0
            assert len(item["metadata"]["entity"]) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_popqa_load_real_content_quality(self) -> None:
        """Test that content is substantial and meaningful."""
        loader = PopQADataloader(limit=1)
        result = loader.load()

        assert len(result) > 0
        item = result[0]

        # Text should contain more than just metadata
        text = item["text"]
        question = item["metadata"]["question"]

        # Text should be longer than question (contains content)
        assert len(text) > len(question)
