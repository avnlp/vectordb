"""Tests for diversification utilities (LangChain)."""

import pytest
from langchain_core.documents import Document

from vectordb.langchain.utils.diversification import DiversificationHelper


class TestCosineSimilarity:
    """Unit tests for DiversificationHelper.cosine_similarity method."""

    def test_identical_vectors(self):
        """Test cosine similarity of identical vectors is 1.0."""
        embedding = [0.1, 0.2, 0.3, 0.4]
        similarity = DiversificationHelper.cosine_similarity(embedding, embedding)
        assert similarity == pytest.approx(1.0, rel=1e-5)

    def test_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors is 0.0."""
        embedding1 = [1.0, 0.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0, 0.0]
        similarity = DiversificationHelper.cosine_similarity(embedding1, embedding2)
        assert similarity == pytest.approx(0.0, abs=1e-5)

    def test_opposite_vectors(self):
        """Test cosine similarity of opposite vectors is -1.0."""
        embedding1 = [1.0, 0.0, 0.0, 0.0]
        embedding2 = [-1.0, 0.0, 0.0, 0.0]
        similarity = DiversificationHelper.cosine_similarity(embedding1, embedding2)
        assert similarity == pytest.approx(-1.0, rel=1e-5)

    def test_zero_vector_first(self):
        """Test cosine similarity with first zero vector returns 0.0."""
        embedding1 = [0.0, 0.0, 0.0, 0.0]
        embedding2 = [1.0, 2.0, 3.0, 4.0]
        similarity = DiversificationHelper.cosine_similarity(embedding1, embedding2)
        assert similarity == 0.0

    def test_zero_vector_second(self):
        """Test cosine similarity with second zero vector returns 0.0."""
        embedding1 = [1.0, 2.0, 3.0, 4.0]
        embedding2 = [0.0, 0.0, 0.0, 0.0]
        similarity = DiversificationHelper.cosine_similarity(embedding1, embedding2)
        assert similarity == 0.0

    def test_both_zero_vectors(self):
        """Test cosine similarity with both zero vectors returns 0.0."""
        embedding1 = [0.0, 0.0, 0.0, 0.0]
        embedding2 = [0.0, 0.0, 0.0, 0.0]
        similarity = DiversificationHelper.cosine_similarity(embedding1, embedding2)
        assert similarity == 0.0

    def test_highly_similar_vectors(self):
        """Test cosine similarity of highly similar vectors is close to 1.0."""
        embedding1 = [1.0, 2.0, 3.0]
        embedding2 = [1.1, 2.1, 3.1]
        similarity = DiversificationHelper.cosine_similarity(embedding1, embedding2)
        assert similarity > 0.99

    def test_dissimilar_vectors(self):
        """Test cosine similarity of dissimilar vectors is low."""
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.0, 0.5, 0.5]
        similarity = DiversificationHelper.cosine_similarity(embedding1, embedding2)
        assert similarity < 0.5

    def test_multi_dimensional_vectors(self):
        """Test cosine similarity with higher dimensional vectors."""
        embedding1 = [0.1] * 100
        embedding2 = [0.1] * 100
        similarity = DiversificationHelper.cosine_similarity(embedding1, embedding2)
        assert similarity == pytest.approx(1.0, rel=1e-5)

    def test_returns_float(self):
        """Test that cosine similarity returns a float type."""
        embedding1 = [1.0, 2.0, 3.0]
        embedding2 = [4.0, 5.0, 6.0]
        similarity = DiversificationHelper.cosine_similarity(embedding1, embedding2)
        assert isinstance(similarity, float)

    def test_negative_similarity(self):
        """Test cosine similarity can be negative."""
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [-0.5, -0.5, 0.0]
        similarity = DiversificationHelper.cosine_similarity(embedding1, embedding2)
        assert similarity < 0


class TestDiversify:
    """Unit tests for DiversificationHelper.diversify method."""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(page_content="Document 1 content", metadata={"id": "1"}),
            Document(page_content="Document 2 content", metadata={"id": "2"}),
            Document(page_content="Document 3 content", metadata={"id": "3"}),
            Document(page_content="Document 4 content", metadata={"id": "4"}),
            Document(page_content="Document 5 content", metadata={"id": "5"}),
        ]

    @pytest.fixture
    def similar_embeddings(self):
        """Create embeddings where all are similar to query."""
        return [
            [0.9, 0.1, 0.1],
            [0.85, 0.2, 0.2],
            [0.8, 0.3, 0.3],
            [0.75, 0.4, 0.4],
            [0.7, 0.5, 0.5],
        ]

    @pytest.fixture
    def diverse_embeddings(self):
        """Create embeddings with high diversity."""
        return [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
        ]

    def test_empty_documents(self):
        """Test diversify with empty documents list returns empty list."""
        result = DiversificationHelper.diversify(
            documents=[],
            embeddings=[],
            max_documents=5,
        )
        assert result == []

    def test_single_document(self, sample_documents):
        """Test diversify with single document returns that document."""
        embeddings = [[0.5, 0.5, 0.5]]
        result = DiversificationHelper.diversify(
            documents=sample_documents[:1],
            embeddings=embeddings,
            max_documents=5,
        )
        assert len(result) == 1
        assert result[0] == sample_documents[0]

    def test_less_than_max_documents(self, sample_documents):
        """Test diversify when documents less than max_documents returns all."""
        embeddings = [[0.5, 0.5, 0.5]] * 3
        result = DiversificationHelper.diversify(
            documents=sample_documents[:3],
            embeddings=embeddings,
            max_documents=10,
        )
        assert len(result) == 3

    def test_max_documents_less_than_documents(
        self, sample_documents, diverse_embeddings
    ):
        """Test diversify with max_documents less than available documents."""
        result = DiversificationHelper.diversify(
            documents=sample_documents,
            embeddings=diverse_embeddings,
            max_documents=3,
        )
        assert len(result) == 3

    def test_max_documents_zero(self, sample_documents):
        """Test diversify with max_documents=0 returns empty list."""
        embeddings = [[0.5, 0.5, 0.5]] * 3
        result = DiversificationHelper.diversify(
            documents=sample_documents[:3],
            embeddings=embeddings,
            max_documents=0,
        )
        assert len(result) == 0

    def test_mismatched_embeddings_count(self, sample_documents):
        """Test diversify with mismatched embeddings count raises ValueError."""
        with pytest.raises(ValueError, match="Number of embeddings must match"):
            DiversificationHelper.diversify(
                documents=sample_documents[:3],
                embeddings=[[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]],  # Only 2 embeddings
                max_documents=5,
            )

    def test_first_document_always_included(self, sample_documents):
        """Test that first document is always included in results."""
        embeddings = [
            [0.9, 0.1, 0.1],
            [0.85, 0.2, 0.2],
            [0.8, 0.3, 0.3],
        ]
        result = DiversificationHelper.diversify(
            documents=sample_documents[:3],
            embeddings=embeddings,
            max_documents=2,
        )
        assert result[0] == sample_documents[0]

    def test_diverse_embeddings_preserved(self, sample_documents, diverse_embeddings):
        """Test that diverse embeddings are properly selected."""
        result = DiversificationHelper.diversify(
            documents=sample_documents,
            embeddings=diverse_embeddings,
            max_documents=5,
            similarity_threshold=1.0,  # High threshold, select all
        )
        assert len(result) == 5

    def test_low_threshold_selects_diverse(self, sample_documents, diverse_embeddings):
        """Test low similarity threshold promotes diversity."""
        result = DiversificationHelper.diversify(
            documents=sample_documents,
            embeddings=diverse_embeddings,
            max_documents=3,
            similarity_threshold=0.1,  # Very low threshold
        )
        assert len(result) == 3
        # Should include first doc and then select most diverse ones

    def test_high_threshold_selects_more(self, sample_documents, similar_embeddings):
        """Test high similarity threshold allows more similar docs."""
        result = DiversificationHelper.diversify(
            documents=sample_documents,
            embeddings=similar_embeddings,
            max_documents=5,
            similarity_threshold=0.9,  # High threshold
        )
        assert len(result) <= 5

    def test_preserves_metadata(self, sample_documents, diverse_embeddings):
        """Test diversify preserves document metadata."""
        result = DiversificationHelper.diversify(
            documents=sample_documents,
            embeddings=diverse_embeddings,
            max_documents=3,
        )
        for doc in result:
            assert "id" in doc.metadata

    def test_preserves_page_content(self, sample_documents, diverse_embeddings):
        """Test diversify preserves document page content."""
        result = DiversificationHelper.diversify(
            documents=sample_documents,
            embeddings=diverse_embeddings,
            max_documents=3,
        )
        content_set = {doc.page_content for doc in result}
        for doc in sample_documents[:3]:
            assert doc.page_content in content_set


class TestClusteringBasedDiversity:
    """Unit tests for DiversificationHelper.clustering_based_diversity method."""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(page_content="Document 1 content", metadata={"id": "1"}),
            Document(page_content="Document 2 content", metadata={"id": "2"}),
            Document(page_content="Document 3 content", metadata={"id": "3"}),
            Document(page_content="Document 4 content", metadata={"id": "4"}),
            Document(page_content="Document 5 content", metadata={"id": "5"}),
            Document(page_content="Document 6 content", metadata={"id": "6"}),
        ]

    def test_empty_documents(self):
        """Test clustering with empty documents returns empty list."""
        result = DiversificationHelper.clustering_based_diversity(
            documents=[],
            embeddings=[],
        )
        assert result == []

    def test_mismatched_embeddings_count(self, sample_documents):
        """Test clustering raises error.

        Embeddings count doesn't match documents.
        """
        with pytest.raises(ValueError, match="Number of embeddings must match"):
            DiversificationHelper.clustering_based_diversity(
                documents=sample_documents[:3],
                embeddings=[[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]],
            )

    def test_clustering_max_documents_zero(self, sample_documents):
        """Test clustering with 0 clusters or samples returns empty list."""
        embeddings = [[0.5, 0.5, 0.5]] * 3
        result = DiversificationHelper.clustering_based_diversity(
            documents=sample_documents[:3],
            embeddings=embeddings,
            num_clusters=0,
            samples_per_cluster=2,
        )
        assert len(result) == 0

        result = DiversificationHelper.clustering_based_diversity(
            documents=sample_documents[:3],
            embeddings=embeddings,
            num_clusters=3,
            samples_per_cluster=0,
        )
        assert len(result) == 0

    def test_clusters_less_than_documents(self, sample_documents):
        """Test clustering with num_clusters less than document count."""
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 0.9],
        ]
        result = DiversificationHelper.clustering_based_diversity(
            documents=sample_documents,
            embeddings=embeddings,
            num_clusters=3,
            samples_per_cluster=1,
        )
        assert len(result) == 3

    def test_clusters_more_than_documents(self, sample_documents):
        """Test clustering with num_clusters more than document count."""
        embeddings = [[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]]
        result = DiversificationHelper.clustering_based_diversity(
            documents=sample_documents[:2],
            embeddings=embeddings,
            num_clusters=10,  # More clusters than documents
            samples_per_cluster=2,
        )
        # Should return at most number of documents
        assert len(result) <= 2

    def test_samples_per_cluster_limit(self, sample_documents):
        """Test samples_per_cluster limits documents per cluster."""
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.85, 0.15, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
            [0.0, 0.85, 0.15],
        ]
        result = DiversificationHelper.clustering_based_diversity(
            documents=sample_documents,
            embeddings=embeddings,
            num_clusters=2,
            samples_per_cluster=1,  # Only 1 per cluster
        )
        assert len(result) == 2

    def test_preserves_metadata(self, sample_documents):
        """Test clustering preserves document metadata."""
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ]
        result = DiversificationHelper.clustering_based_diversity(
            documents=sample_documents[:4],
            embeddings=embeddings,
            num_clusters=2,
            samples_per_cluster=1,
        )
        for doc in result:
            assert "id" in doc.metadata

    def test_fallback_without_sklearn(self):
        """Test fallback to diversify when sklearn is not available."""
        import sys
        from unittest.mock import patch

        # Mock sklearn to raise ImportError
        with patch.dict(sys.modules, {"sklearn.cluster": None}):
            docs = [
                Document(page_content="Doc 1", metadata={"id": "1"}),
                Document(page_content="Doc 2", metadata={"id": "2"}),
                Document(page_content="Doc 3", metadata={"id": "3"}),
            ]
            emb = [[0.5, 0.5, 0.5]] * 3
            # This should not raise, should fall back to diversify
            result = DiversificationHelper.clustering_based_diversity(
                documents=docs,
                embeddings=emb,
                num_clusters=2,
                samples_per_cluster=2,
            )
            assert len(result) > 0


class TestDiversificationHelperEdgeCases:
    """Edge case tests for DiversificationHelper."""

    def test_very_high_dimensional_embeddings(self):
        """Test with very high dimensional embeddings."""
        dim = 1000
        documents = [
            Document(page_content="Doc 1", metadata={"id": "1"}),
            Document(page_content="Doc 2", metadata={"id": "2"}),
        ]
        embeddings = [
            [0.5] * dim,
            [0.6] * dim,
        ]
        result = DiversificationHelper.diversify(
            documents=documents,
            embeddings=embeddings,
            max_documents=2,
        )
        assert len(result) == 2

    def test_very_low_dimensional_embeddings(self):
        """Test with 1-dimensional embeddings."""
        documents = [
            Document(page_content="Doc 1", metadata={"id": "1"}),
            Document(page_content="Doc 2", metadata={"id": "2"}),
        ]
        embeddings = [[0.5], [0.8]]
        result = DiversificationHelper.diversify(
            documents=documents,
            embeddings=embeddings,
            max_documents=2,
        )
        assert len(result) == 2

    def test_threshold_zero(self):
        """Test with similarity threshold of zero."""
        documents = [
            Document(page_content="Doc 1", metadata={"id": "1"}),
            Document(page_content="Doc 2", metadata={"id": "2"}),
            Document(page_content="Doc 3", metadata={"id": "3"}),
        ]
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        result = DiversificationHelper.diversify(
            documents=documents,
            embeddings=embeddings,
            max_documents=3,
            similarity_threshold=0.0,
        )
        # With threshold 0, should select all docs
        assert len(result) == 3

    def test_all_same_embeddings(self):
        """Test with all identical embeddings."""
        documents = [
            Document(page_content="Doc 1", metadata={"id": "1"}),
            Document(page_content="Doc 2", metadata={"id": "2"}),
            Document(page_content="Doc 3", metadata={"id": "3"}),
        ]
        embeddings = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
        result = DiversificationHelper.diversify(
            documents=documents,
            embeddings=embeddings,
            max_documents=3,
        )
        # When documents <= max_documents, all are returned without diversification
        assert len(result) == 3

    def test_already_diverse_documents(self):
        """Test with already diverse documents."""
        documents = [
            Document(page_content="Doc 1", metadata={"id": "1"}),
            Document(page_content="Doc 2", metadata={"id": "2"}),
            Document(page_content="Doc 3", metadata={"id": "3"}),
        ]
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        result = DiversificationHelper.diversify(
            documents=documents,
            embeddings=embeddings,
            max_documents=3,
            similarity_threshold=0.5,
        )
        # All should be included since they're orthogonal
        assert len(result) == 3
