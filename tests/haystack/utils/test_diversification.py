"""Tests for DiversificationHelper utility class."""

import pytest
from haystack import Document

from vectordb.haystack.utils import DiversificationHelper


class TestDiversificationHelper:
    """Tests for semantic diversification helper."""

    @pytest.fixture
    def docs_with_similar_embeddings(self) -> list[Document]:
        """Create documents with similar embeddings."""
        return [
            Document(content="Doc 1", embedding=[1.0, 0.0, 0.0]),
            Document(content="Doc 2", embedding=[0.99, 0.1, 0.0]),
            Document(content="Doc 3", embedding=[0.98, 0.15, 0.0]),
            Document(content="Doc 4", embedding=[0.0, 1.0, 0.0]),
        ]

    @pytest.fixture
    def docs_with_diverse_embeddings(self) -> list[Document]:
        """Create documents with diverse embeddings."""
        return [
            Document(content="Doc 1", embedding=[1.0, 0.0, 0.0]),
            Document(content="Doc 2", embedding=[0.0, 1.0, 0.0]),
            Document(content="Doc 3", embedding=[0.0, 0.0, 1.0]),
        ]

    def test_apply_disabled(self, docs_with_similar_embeddings: list[Document]) -> None:
        """Test diversification returns all docs when disabled."""
        config = {"semantic_diversification": {"enabled": False}}
        result = DiversificationHelper.apply(docs_with_similar_embeddings, config)
        assert len(result) == 4

    def test_apply_no_config(
        self, docs_with_similar_embeddings: list[Document]
    ) -> None:
        """Test diversification returns all docs with no config."""
        result = DiversificationHelper.apply(docs_with_similar_embeddings, {})
        assert len(result) == 4

    def test_apply_filters_similar_docs(
        self, docs_with_similar_embeddings: list[Document]
    ) -> None:
        """Test diversification filters out similar documents."""
        config = {
            "semantic_diversification": {
                "enabled": True,
                "diversity_threshold": 0.95,
                "max_similar_docs": 1,
            }
        }
        result = DiversificationHelper.apply(docs_with_similar_embeddings, config)
        # First doc is kept, 2nd and 3rd are too similar, 4th is diverse
        assert len(result) == 2
        assert result[0].content == "Doc 1"
        assert result[1].content == "Doc 4"

    def test_apply_keeps_diverse_docs(
        self, docs_with_diverse_embeddings: list[Document]
    ) -> None:
        """Test diversification keeps all diverse documents."""
        config = {
            "semantic_diversification": {
                "enabled": True,
                "diversity_threshold": 0.9,
                "max_similar_docs": 1,
            }
        }
        result = DiversificationHelper.apply(docs_with_diverse_embeddings, config)
        assert len(result) == 3

    def test_apply_empty_list(self) -> None:
        """Test diversification with empty list."""
        config = {"semantic_diversification": {"enabled": True}}
        result = DiversificationHelper.apply([], config)
        assert result == []

    def test_apply_no_embeddings(self) -> None:
        """Test diversification with docs without embeddings."""
        docs = [
            Document(content="Doc 1"),
            Document(content="Doc 2"),
        ]
        config = {"semantic_diversification": {"enabled": True}}
        result = DiversificationHelper.apply(docs, config)
        assert len(result) == 2

    def test_apply_with_max_similar(
        self, docs_with_similar_embeddings: list[Document]
    ) -> None:
        """Test max_similar_docs parameter."""
        config = {
            "semantic_diversification": {
                "enabled": True,
                "diversity_threshold": 0.95,
                "max_similar_docs": 2,
            }
        }
        result = DiversificationHelper.apply(docs_with_similar_embeddings, config)
        # With max_similar_docs=2, allows 2 similar docs
        assert len(result) == 3

    def test_cosine_similarity_identical(self) -> None:
        """Test cosine similarity for identical vectors."""
        vec = [1.0, 0.0, 0.0]
        result = DiversificationHelper._cosine_similarity(vec, vec)
        assert result == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self) -> None:
        """Test cosine similarity for orthogonal vectors."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        result = DiversificationHelper._cosine_similarity(vec_a, vec_b)
        assert result == pytest.approx(0.0)

    def test_cosine_similarity_zero_vector(self) -> None:
        """Test cosine similarity with zero vector."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 0.0, 0.0]
        result = DiversificationHelper._cosine_similarity(vec_a, vec_b)
        assert result == 0.0
