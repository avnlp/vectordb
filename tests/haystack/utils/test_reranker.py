"""Tests for RerankerFactory utility class."""

from unittest.mock import MagicMock, patch

import pytest

from vectordb.haystack.utils.reranker import RerankerFactory


class TestRerankerFactory:
    """Tests for RerankerFactory class."""

    def test_create_missing_model(self) -> None:
        """Test error when model not specified."""
        with pytest.raises(KeyError):
            RerankerFactory.create({})

    @patch("vectordb.haystack.utils.reranker.SentenceTransformersSimilarityRanker")
    def test_create_reranker(self, mock_ranker_class: MagicMock) -> None:
        """Test reranker creation."""
        mock_instance = MagicMock()
        mock_ranker_class.return_value = mock_instance

        config = {
            "reranker": {
                "model": "BAAI/bge-reranker-v2-m3",
                "top_k": 10,
            }
        }
        result = RerankerFactory.create(config)

        mock_ranker_class.assert_called_once_with(
            model="BAAI/bge-reranker-v2-m3",
            top_k=10,
        )
        mock_instance.warm_up.assert_called_once()
        assert result == mock_instance

    @patch("vectordb.haystack.utils.reranker.SentenceTransformersSimilarityRanker")
    def test_create_reranker_default_top_k(self, mock_ranker_class: MagicMock) -> None:
        """Test reranker with default top_k."""
        mock_instance = MagicMock()
        mock_ranker_class.return_value = mock_instance

        config = {"reranker": {"model": "test-model"}}
        RerankerFactory.create(config)

        call_kwargs = mock_ranker_class.call_args[1]
        assert call_kwargs["top_k"] == 5

    def test_create_diversity_ranker_missing_model(self) -> None:
        """Test error when MMR model not specified."""
        with pytest.raises(KeyError):
            RerankerFactory.create_diversity_ranker({})

    @patch("haystack.components.rankers.SentenceTransformersDiversityRanker")
    def test_create_diversity_ranker(self, mock_ranker_class: MagicMock) -> None:
        """Test diversity ranker creation."""
        mock_instance = MagicMock()
        mock_ranker_class.return_value = mock_instance

        config = {
            "mmr": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "top_k": 15,
            }
        }
        result = RerankerFactory.create_diversity_ranker(config)

        mock_ranker_class.assert_called_once_with(
            model="sentence-transformers/all-MiniLM-L6-v2",
            top_k=15,
            strategy="maximum_margin_relevance",
        )
        mock_instance.warm_up.assert_called_once()
        assert result == mock_instance

    @patch("haystack.components.rankers.SentenceTransformersDiversityRanker")
    def test_create_diversity_ranker_default_top_k(
        self, mock_ranker_class: MagicMock
    ) -> None:
        """Test diversity ranker with default top_k."""
        mock_instance = MagicMock()
        mock_ranker_class.return_value = mock_instance

        config = {"mmr": {"model": "test-model"}}
        RerankerFactory.create_diversity_ranker(config)

        call_kwargs = mock_ranker_class.call_args[1]
        assert call_kwargs["top_k"] == 10
