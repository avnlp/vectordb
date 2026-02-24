"""Tests for ClusteringDiversityRanker."""

import pytest
from haystack import Document


class TestClusteringDiversityRanker:
    """Tests for ClusteringDiversityRanker class."""

    @pytest.fixture
    def sample_documents(self) -> list[Document]:
        """Create sample documents for testing."""
        return [
            Document(
                content=f"Document about topic {i}",
                meta={"source": f"doc{i}"},
                score=0.9 - i * 0.05,
            )
            for i in range(20)
        ]

    def test_init_default_values(self) -> None:
        """Test initialization with default values."""
        from vectordb.haystack.diversity_filtering.rankers import (
            ClusteringDiversityRanker,
        )

        ranker = ClusteringDiversityRanker()

        assert ranker.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert ranker.top_k == 10
        assert ranker.similarity == "cosine"
        assert ranker._embedding_model is None

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values."""
        from vectordb.haystack.diversity_filtering.rankers import (
            ClusteringDiversityRanker,
        )

        ranker = ClusteringDiversityRanker(
            model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            top_k=5,
            similarity="dot_product",
        )

        assert (
            ranker.model_name
            == "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        assert ranker.top_k == 5
        assert ranker.similarity == "dot_product"

    def test_component_has_run_method(self) -> None:
        """Test that ranker has proper Haystack component methods."""
        from vectordb.haystack.diversity_filtering.rankers import (
            ClusteringDiversityRanker,
        )

        ranker = ClusteringDiversityRanker()

        assert hasattr(ranker, "run")
        assert hasattr(ranker, "warm_up")
