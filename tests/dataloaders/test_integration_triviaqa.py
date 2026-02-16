"""Integration tests for TriviaQA dataset loader with actual HuggingFace data.

This module contains integration tests that fetch actual TriviaQA datasets.
Tests are marked with @pytest.mark.integration and require internet access.
"""

import pytest

from vectordb.dataloaders.triviaqa import TriviaQADataloader


class TestTriviaQADataloaderIntegration:
    """Integration test suite for TriviaQA dataset with live HuggingFace data.

    Tests cover:
    - Actual dataset loading from HuggingFace
    - Real data structure validation
    - Search result handling
    - Metadata preservation from live data
    - Limit handling on real datasets
    """

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_triviaqa_load_real_test_split(self) -> None:
        """Test loading real TriviaQA test split from HuggingFace."""
        loader = TriviaQADataloader(split="test")
        result = loader.load()

        # Should return actual data
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_triviaqa_load_real_validation_split(self) -> None:
        """Test loading real TriviaQA validation split from HuggingFace."""
        loader = TriviaQADataloader(split="validation")
        result = loader.load()

        # Should return actual data
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_triviaqa_load_real_data_structure(self) -> None:
        """Test that real TriviaQA data has expected structure."""
        loader = TriviaQADataloader(limit=1)
        result = loader.load()

        assert len(result) > 0
        item = result[0]

        # Check structure
        assert "text" in item
        assert "metadata" in item

        # Check metadata fields
        metadata = item["metadata"]
        assert "question" in metadata
        assert "answer" in metadata

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_triviaqa_load_real_text_contains_question(self) -> None:
        """Test that loaded text contains the question."""
        loader = TriviaQADataloader(limit=1)
        result = loader.load()

        assert len(result) > 0
        item = result[0]
        question = item["metadata"]["question"]

        # Text should contain the question
        assert question in item["text"]

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_triviaqa_load_real_text_contains_search_context(self) -> None:
        """Test that loaded text includes search context."""
        loader = TriviaQADataloader(limit=1)
        result = loader.load()

        assert len(result) > 0
        item = result[0]

        # Should have search context or title in text
        assert "text" in item
        assert len(item["text"]) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_triviaqa_load_real_answer_not_empty(self) -> None:
        """Test that answers are not empty strings."""
        loader = TriviaQADataloader(limit=10)
        result = loader.load()

        assert len(result) > 0
        for item in result:
            answer = item["metadata"]["answer"]
            assert len(answer) > 0
            assert len(answer.strip()) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_triviaqa_load_real_limit_respected(self) -> None:
        """Test that limit parameter is respected on real data."""
        loader = TriviaQADataloader(limit=5)
        result = loader.load()

        assert len(result) <= 5

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_triviaqa_load_real_has_search_results(self) -> None:
        """Test that items have search result metadata."""
        loader = TriviaQADataloader(limit=1)
        result = loader.load()

        assert len(result) > 0
        item = result[0]
        metadata = item["metadata"]

        # Should have search-related metadata
        assert "rank" in metadata or "answer" in metadata

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_triviaqa_load_real_multiple_search_results(self) -> None:
        """Test that TriviaQA expands multiple search results into items."""
        loader = TriviaQADataloader(limit=10)
        result = loader.load()

        # Should have multiple items due to search result expansion
        assert len(result) > 0

        # Verify each item is properly formatted
        for item in result:
            assert "text" in item
            assert "metadata" in item
            assert "question" in item["metadata"]

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_triviaqa_load_real_question_not_empty(self) -> None:
        """Test that loaded questions are not empty."""
        loader = TriviaQADataloader(limit=10)
        result = loader.load()

        assert len(result) > 0
        for item in result:
            question = item["metadata"]["question"]
            assert len(question) > 0
            assert len(question.strip()) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_triviaqa_load_real_rc_config(self) -> None:
        """Test loading TriviaQA with rc (reading comprehension) config."""
        loader = TriviaQADataloader(config="rc")
        result = loader.load()

        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_triviaqa_load_real_batches(self) -> None:
        """Test loading multiple items validates batch consistency."""
        loader = TriviaQADataloader(limit=20)
        result = loader.load()

        assert len(result) > 0
        assert len(result) <= 20

        # All items should have same structure
        for item in result:
            assert "text" in item
            assert "metadata" in item
            assert "question" in item["metadata"]
            assert "answer" in item["metadata"]

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_triviaqa_load_real_different_limits(self) -> None:
        """Test that loading with different limits works correctly."""
        limits = [1, 5, 10]

        for limit in limits:
            loader = TriviaQADataloader(limit=limit)
            result = loader.load()

            assert len(result) <= limit
            assert len(result) > 0

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_triviaqa_load_real_text_format(self) -> None:
        """Test that text field contains meaningful content."""
        loader = TriviaQADataloader(limit=1)
        result = loader.load()

        assert len(result) > 0
        for item in result:
            text = item["text"]
            # Should be non-empty and contain more than just the question
            assert len(text) > len(item["metadata"]["question"])
