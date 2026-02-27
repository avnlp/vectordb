"""Tests for diversity filtering helper utilities (LangChain)."""

import pytest
from langchain_core.documents import Document

from vectordb.langchain.diversity_filtering.helpers import DiversityFilteringHelper


def _doc(doc_id: str) -> Document:
    return Document(page_content=f"Document {doc_id}", metadata={"id": doc_id})


class TestCosineSimilarity:
    """Unit tests for cosine similarity."""

    def test_identical_vectors(self) -> None:
        """Verifies that two identical vectors yield a cosine similarity of 1.0."""
        similarity = DiversityFilteringHelper.cosine_similarity(
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        )
        assert similarity == pytest.approx(1.0, rel=1e-5)

    def test_orthogonal_vectors(self) -> None:
        """Verifies that two orthogonal vectors yield a cosine similarity of 0.0."""
        similarity = DiversityFilteringHelper.cosine_similarity(
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        )
        assert similarity == pytest.approx(0.0, abs=1e-5)

    def test_zero_vector(self) -> None:
        """Verifies that a zero vector yields a cosine similarity of 0.0."""
        similarity = DiversityFilteringHelper.cosine_similarity(
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
        )
        assert similarity == 0.0


class TestMMRDiversify:
    """Unit tests for MMR-based diversification."""

    @pytest.fixture
    def documents(self) -> list[Document]:
        """Returns a list of four test documents with sequential IDs."""
        return [_doc("1"), _doc("2"), _doc("3"), _doc("4")]

    @pytest.fixture
    def embeddings(self) -> list[list[float]]:
        """Returns embeddings aligned to the documents fixture for MMR testing."""
        return [
            [1.0, 0.2, 0.0],  # doc1 (chosen first for query [1, 1, 0])
            [1.0, 0.0, 0.0],  # doc2 (relevance-heavy choice)
            [0.0, 1.0, 0.0],  # doc3 (diversity-heavy choice)
            [0.2, 0.0, 1.0],  # doc4
        ]

    def test_empty_documents(self) -> None:
        """Verifies that an empty document list returns an empty result."""
        result = DiversityFilteringHelper.mmr_diversify(
            documents=[],
            embeddings=[],
            query_embedding=[1.0, 0.0, 0.0],
        )
        assert result == []

    def test_single_document(self) -> None:
        """Verifies that a single document is returned unchanged."""
        docs = [_doc("1")]
        result = DiversityFilteringHelper.mmr_diversify(
            documents=docs,
            embeddings=[[1.0, 0.0, 0.0]],
            query_embedding=[1.0, 0.0, 0.0],
            max_documents=5,
        )
        assert result == docs

    def test_fewer_docs_than_max_documents(self, documents, embeddings) -> None:
        """Verifies all documents are returned when count is below max_documents."""
        result = DiversityFilteringHelper.mmr_diversify(
            documents=documents[:2],
            embeddings=embeddings[:2],
            query_embedding=[1.0, 1.0, 0.0],
            max_documents=5,
        )
        assert len(result) == 2

    def test_high_lambda_prioritizes_relevance(self, documents, embeddings) -> None:
        """Verifies that lambda=1.0 selects the two most query-relevant documents."""
        result = DiversityFilteringHelper.mmr_diversify(
            documents=documents,
            embeddings=embeddings,
            query_embedding=[1.0, 1.0, 0.0],
            max_documents=2,
            lambda_param=1.0,
        )
        assert [doc.metadata["id"] for doc in result] == ["1", "2"]

    def test_low_lambda_prioritizes_diversity(self, documents, embeddings) -> None:
        """Verifies that lambda=0.0 selects a maximally diverse second document."""
        result = DiversityFilteringHelper.mmr_diversify(
            documents=documents,
            embeddings=embeddings,
            query_embedding=[1.0, 1.0, 0.0],
            max_documents=2,
            lambda_param=0.0,
        )
        ids = [doc.metadata["id"] for doc in result]
        assert ids[0] == "1"
        assert ids[1] != "2"

    def test_balanced_lambda_returns_mixed_selection(
        self, documents, embeddings
    ) -> None:
        """Verifies that lambda=0.5 returns a mixed relevance-diversity selection."""
        result = DiversityFilteringHelper.mmr_diversify(
            documents=documents,
            embeddings=embeddings,
            query_embedding=[1.0, 1.0, 0.0],
            max_documents=3,
            lambda_param=0.5,
        )
        ids = [doc.metadata["id"] for doc in result]
        assert ids[0] == "1"
        assert len(ids) == 3

    def test_query_embedding_changes_selection(self, documents, embeddings) -> None:
        """Verifies that changing the query embedding alters document selection."""
        relevance_query = DiversityFilteringHelper.mmr_diversify(
            documents=documents,
            embeddings=embeddings,
            query_embedding=[1.0, 1.0, 0.0],
            max_documents=2,
            lambda_param=1.0,
        )
        different_query = DiversityFilteringHelper.mmr_diversify(
            documents=documents,
            embeddings=embeddings,
            query_embedding=[0.0, 0.0, 1.0],
            max_documents=2,
            lambda_param=1.0,
        )
        assert [doc.metadata["id"] for doc in relevance_query] != [
            doc.metadata["id"] for doc in different_query
        ]

    def test_max_documents_zero_returns_empty(self, documents, embeddings) -> None:
        """Verifies that max_documents=0 returns an empty list."""
        result = DiversityFilteringHelper.mmr_diversify(
            documents=documents,
            embeddings=embeddings,
            query_embedding=[1.0, 0.0, 0.0],
            max_documents=0,
        )
        assert result == []

    @pytest.mark.parametrize("lambda_param", [-0.1, 1.1])
    def test_invalid_lambda_raises(self, documents, embeddings, lambda_param) -> None:
        """Verifies that out-of-range lambda values raise ValueError."""
        with pytest.raises(
            ValueError, match="lambda_param must be between 0.0 and 1.0"
        ):
            DiversityFilteringHelper.mmr_diversify(
                documents=documents,
                embeddings=embeddings,
                query_embedding=[1.0, 0.0, 0.0],
                lambda_param=lambda_param,
            )

    def test_query_document_dimension_mismatch_raises(
        self, documents, embeddings
    ) -> None:
        """Verifies that mismatched query/document embedding dimensions raise."""
        with pytest.raises(
            ValueError,
            match="All document embeddings must match query embedding dimensions",
        ):
            DiversityFilteringHelper.mmr_diversify(
                documents=documents,
                embeddings=embeddings,
                query_embedding=[1.0, 0.0],
            )

    def test_embedding_count_mismatch_raises(self, documents) -> None:
        """Verifies that mismatched embeddings-to-documents count raises ValueError."""
        with pytest.raises(ValueError, match="Number of embeddings must match"):
            DiversityFilteringHelper.mmr_diversify(
                documents=documents,
                embeddings=[[1.0, 0.0, 0.0]],
                query_embedding=[1.0, 0.0, 0.0],
            )


class TestClusteringDiversify:
    """Unit tests for clustering-based diversification."""

    @pytest.fixture
    def documents(self) -> list[Document]:
        """Returns a list of six test documents with sequential IDs."""
        return [_doc("1"), _doc("2"), _doc("3"), _doc("4"), _doc("5"), _doc("6")]

    def test_empty_documents(self) -> None:
        """Verifies that an empty document list returns an empty result."""
        result = DiversityFilteringHelper.clustering_diversify(
            documents=[],
            embeddings=[],
        )
        assert result == []

    def test_mismatched_embeddings_count_raises(self, documents) -> None:
        """Verifies that mismatched embedding count raises ValueError."""
        with pytest.raises(ValueError, match="Number of embeddings must match"):
            DiversityFilteringHelper.clustering_diversify(
                documents=documents,
                embeddings=[[1.0, 0.0, 0.0]],
            )

    def test_zero_clusters_or_samples_returns_empty(self, documents) -> None:
        """Verifies that zero clusters or zero samples per cluster returns empty."""
        embeddings = [[0.1, 0.2, 0.3]] * 3
        no_clusters = DiversityFilteringHelper.clustering_diversify(
            documents=documents[:3],
            embeddings=embeddings,
            num_clusters=0,
            samples_per_cluster=1,
        )
        no_samples = DiversityFilteringHelper.clustering_diversify(
            documents=documents[:3],
            embeddings=embeddings,
            num_clusters=2,
            samples_per_cluster=0,
        )
        assert no_clusters == []
        assert no_samples == []

    def test_clusters_less_than_documents(self, documents) -> None:
        """Verifies that fewer clusters than documents returns one doc per cluster."""
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 0.9],
        ]
        result = DiversityFilteringHelper.clustering_diversify(
            documents=documents,
            embeddings=embeddings,
            num_clusters=3,
            samples_per_cluster=1,
        )
        assert len(result) == 3

    def test_clusters_more_than_documents(self, documents) -> None:
        """Verifies that more clusters than documents is handled gracefully."""
        embeddings = [[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]]
        result = DiversityFilteringHelper.clustering_diversify(
            documents=documents[:2],
            embeddings=embeddings,
            num_clusters=10,
            samples_per_cluster=2,
        )
        assert len(result) <= 2

    def test_fallback_without_sklearn(self) -> None:
        """Verifies graceful fallback when sklearn is unavailable."""
        import sys
        from unittest.mock import patch

        docs = [_doc("1"), _doc("2"), _doc("3")]
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ]

        with patch.dict(sys.modules, {"sklearn.cluster": None}):
            result = DiversityFilteringHelper.clustering_diversify(
                documents=docs,
                embeddings=embeddings,
                num_clusters=2,
                samples_per_cluster=1,
            )
            assert len(result) > 0
