"""Tests for the EvaluationExtractor class.

This module tests the EvaluationExtractor which converts dataset items
into evaluation-ready format for RAG system assessment. The extractor:
1. Loads datasets via DatasetRegistry
2. Extracts queries from question/entity fields
3. Normalizes answers to list format
4. Filters duplicate questions
5. Applies limits
6. Strips internal metadata fields from output

Evaluation output format:
    {
        "query": str,           # The question or entity
        "answers": list[str],   # Normalized answer(s)
        "metadata": dict        # Filtered metadata (no question/answer fields)
    }
"""

from unittest.mock import Mock, patch

from vectordb.dataloaders.evaluation import EvaluationExtractor


class TestEvaluationExtractor:
    """Tests for EvaluationExtractor class."""

    def test_extract_basic_functionality(self) -> None:
        """Test basic extraction functionality."""
        # Mock the DatasetRegistry.load method to return sample data
        mock_data = [
            {
                "text": "Some context",
                "metadata": {
                    "question": "What is the capital of France?",
                    "answer": "Paris",
                    "source": "wikipedia",
                },
            },
            {
                "text": "Another context",
                "metadata": {
                    "question": "Who wrote Romeo and Juliet?",
                    "answer": "William Shakespeare",
                    "year": 1595,
                },
            },
        ]

        with patch(
            "vectordb.dataloaders.evaluation.DatasetRegistry.load",
            return_value=mock_data,
        ):
            result = EvaluationExtractor.extract(dataset_type="triviaqa")

        assert len(result) == 2
        assert result[0]["query"] == "What is the capital of France?"
        assert result[0]["answers"] == ["Paris"]
        assert result[1]["query"] == "Who wrote Romeo and Juliet?"
        assert result[1]["answers"] == ["William Shakespeare"]

    def test_extract_with_entity_field(self) -> None:
        """Test extraction when question comes from entity field."""
        mock_data = [
            {
                "text": "Some context",
                "metadata": {
                    "entity": "Albert Einstein",
                    "answer": "Physicist",
                    "birth_year": 1879,
                },
            },
        ]

        with patch(
            "vectordb.dataloaders.evaluation.DatasetRegistry.load",
            return_value=mock_data,
        ):
            result = EvaluationExtractor.extract(dataset_type="triviaqa")

        assert len(result) == 1
        assert result[0]["query"] == "Albert Einstein"
        assert result[0]["answers"] == ["Physicist"]

    def test_extract_with_answers_list(self) -> None:
        """Test extraction when answers is a list."""
        mock_data = [
            {
                "text": "Some context",
                "metadata": {
                    "question": "What are primary colors?",
                    "answers": ["Red", "Blue", "Yellow"],
                    "category": "Art",
                },
            },
        ]

        with patch(
            "vectordb.dataloaders.evaluation.DatasetRegistry.load",
            return_value=mock_data,
        ):
            result = EvaluationExtractor.extract(dataset_type="triviaqa")

        assert len(result) == 1
        assert result[0]["query"] == "What are primary colors?"
        assert result[0]["answers"] == ["Red", "Blue", "Yellow"]

    def test_extract_with_single_answer_string(self) -> None:
        """Test extraction when answer is a single string."""
        mock_data = [
            {
                "text": "Some context",
                "metadata": {
                    "question": "What is 2+2?",
                    "answer": "4",
                    "subject": "Math",
                },
            },
        ]

        with patch(
            "vectordb.dataloaders.evaluation.DatasetRegistry.load",
            return_value=mock_data,
        ):
            result = EvaluationExtractor.extract(dataset_type="triviaqa")

        assert len(result) == 1
        assert result[0]["query"] == "What is 2+2?"
        assert result[0]["answers"] == ["4"]

    def test_extract_with_no_answers(self) -> None:
        """Test extraction when no answers are provided."""
        mock_data = [
            {
                "text": "Some context",
                "metadata": {
                    "question": "What is the meaning of life?",
                    "source": "unknown",
                },
            },
        ]

        with patch(
            "vectordb.dataloaders.evaluation.DatasetRegistry.load",
            return_value=mock_data,
        ):
            result = EvaluationExtractor.extract(dataset_type="triviaqa")

        assert len(result) == 1
        assert result[0]["query"] == "What is the meaning of life?"
        assert result[0]["answers"] == []

    def test_extract_duplicate_questions(self) -> None:
        """Test that duplicate questions are filtered out."""
        mock_data = [
            {
                "text": "Context 1",
                "metadata": {
                    "question": "What is the capital of France?",
                    "answer": "Paris",
                    "source": "wikipedia",
                },
            },
            {
                "text": "Context 2",
                "metadata": {
                    "question": "What is the capital of France?",  # Duplicate
                    "answer": "Paris",
                    "source": "another_source",
                },
            },
        ]

        with patch(
            "vectordb.dataloaders.evaluation.DatasetRegistry.load",
            return_value=mock_data,
        ):
            result = EvaluationExtractor.extract(dataset_type="triviaqa")

        # Only one result should remain due to deduplication
        assert len(result) == 1
        assert result[0]["query"] == "What is the capital of France?"

    def test_extract_limit_functionality(self) -> None:
        """Test that the limit parameter works correctly."""
        mock_data = [
            {
                "text": "Context 1",
                "metadata": {
                    "question": "Question 1",
                    "answer": "Answer 1",
                    "source": "source1",
                },
            },
            {
                "text": "Context 2",
                "metadata": {
                    "question": "Question 2",
                    "answer": "Answer 2",
                    "source": "source2",
                },
            },
            {
                "text": "Context 3",
                "metadata": {
                    "question": "Question 3",
                    "answer": "Answer 3",
                    "source": "source3",
                },
            },
        ]

        with patch(
            "vectordb.dataloaders.evaluation.DatasetRegistry.load",
            return_value=mock_data,
        ):
            # Limit to 2 results
            result = EvaluationExtractor.extract(dataset_type="triviaqa", limit=2)

        assert len(result) == 2
        assert result[0]["query"] == "Question 1"
        assert result[1]["query"] == "Question 2"

    def test_extract_metadata_filtering(self) -> None:
        """Test that metadata is properly filtered to exclude question/answer fields."""
        mock_data = [
            {
                "text": "Some context",
                "metadata": {
                    "question": "What is the capital of France?",
                    "answer": "Paris",
                    "source": "wikipedia",
                    "year": 2023,
                    "author": "John Doe",
                    "extra_field": "extra_value",
                },
            },
        ]

        with patch(
            "vectordb.dataloaders.evaluation.DatasetRegistry.load",
            return_value=mock_data,
        ):
            result = EvaluationExtractor.extract(dataset_type="triviaqa")

        assert len(result) == 1
        metadata = result[0]["metadata"]

        # Check that question and answer fields are not in metadata
        assert "question" not in metadata
        assert "answer" not in metadata
        assert "answers" not in metadata
        assert "entity" not in metadata

        # Check that other fields are preserved
        assert "source" in metadata
        assert "year" in metadata
        assert "author" in metadata
        assert "extra_field" in metadata

    def test_extract_with_none_question(self) -> None:
        """Test that items with None question are skipped."""
        mock_data = [
            {
                "text": "Context 1",
                "metadata": {
                    "question": "Valid question",
                    "answer": "Valid answer",
                    "source": "source1",
                },
            },
            {
                "text": "Context 2",
                "metadata": {
                    "question": None,  # This should be skipped
                    "answer": "Answer without question",
                    "source": "source2",
                },
            },
        ]

        with patch(
            "vectordb.dataloaders.evaluation.DatasetRegistry.load",
            return_value=mock_data,
        ):
            result = EvaluationExtractor.extract(dataset_type="triviaqa")

        # Only the item with a valid question should be included
        assert len(result) == 1
        assert result[0]["query"] == "Valid question"

    def test_extract_with_entity_as_fallback(self) -> None:
        """Test that entity field is used when question field is missing."""
        mock_data = [
            {
                "text": "Context 1",
                "metadata": {
                    "entity": "Albert Einstein",
                    "answer": "Physicist",
                    "field": "Physics",
                },
            },
        ]

        with patch(
            "vectordb.dataloaders.evaluation.DatasetRegistry.load",
            return_value=mock_data,
        ):
            result = EvaluationExtractor.extract(dataset_type="triviaqa")

        assert len(result) == 1
        assert result[0]["query"] == "Albert Einstein"
        assert result[0]["answers"] == ["Physicist"]

    def test_extract_passes_correct_parameters(self) -> None:
        """Test that correct parameters are passed to DatasetRegistry.load."""
        mock_data = []
        mock_registry = Mock()
        mock_registry.load.return_value = mock_data

        with patch("vectordb.dataloaders.evaluation.DatasetRegistry", mock_registry):
            EvaluationExtractor.extract(
                dataset_type="triviaqa",
                dataset_name="custom_dataset",
                split="train",
                limit=10,
            )

        # Verify that DatasetRegistry.load was called with correct parameters
        mock_registry.load.assert_called_once_with(
            "triviaqa", dataset_name="custom_dataset", split="train"
        )
