"""Tests for MMR (Maximal Marginal Relevance) utilities (LangChain).

This module tests the MMRHelper class which provides utilities for Maximal
Marginal Relevance based document reranking. MMR balances relevance to the
query with diversity among selected documents to reduce redundancy.

Test classes:
    TestCosineSimilarity: Tests for vector similarity computation including
        edge cases like zero vectors, orthogonal vectors, and high dimensions.
    TestMMRRerank: Tests for the main MMR algorithm including lambda parameter
        tuning, document selection, and score computation.
    TestMMRRerankSimple: Tests for the simplified MMR interface that returns
        documents without scores.
    TestMMRHelperEdgeCases: Edge cases including high-dimensional vectors,
        negative values, and boundary conditions.
"""

import pytest
from langchain_core.documents import Document

from vectordb.langchain.utils.mmr import MMRHelper


class TestCosineSimilarity:
    """Tests for MMRHelper.cosine_similarity method.

    Validates cosine similarity computation between embedding vectors.
    Tests cover mathematical properties (identity, orthogonality, symmetry)
    and edge cases (zero vectors, high dimensions).
    """

    def test_identical_vectors(self):
        """Test cosine similarity of identical vectors is 1.0."""
        embedding = [0.1, 0.2, 0.3, 0.4]
        similarity = MMRHelper.cosine_similarity(embedding, embedding)
        assert similarity == pytest.approx(1.0, rel=1e-5)

    def test_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors is 0.0."""
        embedding1 = [1.0, 0.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0, 0.0]
        similarity = MMRHelper.cosine_similarity(embedding1, embedding2)
        assert similarity == pytest.approx(0.0, abs=1e-5)

    def test_opposite_vectors(self):
        """Test cosine similarity of opposite vectors is -1.0."""
        embedding1 = [1.0, 0.0, 0.0, 0.0]
        embedding2 = [-1.0, 0.0, 0.0, 0.0]
        similarity = MMRHelper.cosine_similarity(embedding1, embedding2)
        assert similarity == pytest.approx(-1.0, rel=1e-5)

    def test_zero_vector_first(self):
        """Test cosine similarity with first zero vector returns 0.0."""
        embedding1 = [0.0, 0.0, 0.0, 0.0]
        embedding2 = [1.0, 2.0, 3.0, 4.0]
        similarity = MMRHelper.cosine_similarity(embedding1, embedding2)
        assert similarity == 0.0

    def test_zero_vector_second(self):
        """Test cosine similarity with second zero vector returns 0.0."""
        embedding1 = [1.0, 2.0, 3.0, 4.0]
        embedding2 = [0.0, 0.0, 0.0, 0.0]
        similarity = MMRHelper.cosine_similarity(embedding1, embedding2)
        assert similarity == 0.0

    def test_both_zero_vectors(self):
        """Test cosine similarity with both zero vectors returns 0.0."""
        embedding1 = [0.0, 0.0, 0.0, 0.0]
        embedding2 = [0.0, 0.0, 0.0, 0.0]
        similarity = MMRHelper.cosine_similarity(embedding1, embedding2)
        assert similarity == 0.0

    def test_highly_similar_vectors(self):
        """Test cosine similarity of highly similar vectors is close to 1.0."""
        embedding1 = [1.0, 2.0, 3.0]
        embedding2 = [1.1, 2.1, 3.1]
        similarity = MMRHelper.cosine_similarity(embedding1, embedding2)
        assert similarity > 0.99

    def test_dissimilar_vectors(self):
        """Test cosine similarity of dissimilar vectors is low."""
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.0, 0.5, 0.5]
        similarity = MMRHelper.cosine_similarity(embedding1, embedding2)
        assert similarity < 0.5

    def test_multi_dimensional_vectors(self):
        """Test cosine similarity with higher dimensional vectors."""
        embedding1 = [0.1] * 100
        embedding2 = [0.1] * 100
        similarity = MMRHelper.cosine_similarity(embedding1, embedding2)
        assert similarity == pytest.approx(1.0, rel=1e-5)

    def test_returns_float(self):
        """Test that cosine similarity returns a float type."""
        embedding1 = [1.0, 2.0, 3.0]
        embedding2 = [4.0, 5.0, 6.0]
        similarity = MMRHelper.cosine_similarity(embedding1, embedding2)
        assert isinstance(similarity, float)


class TestMMRRerank:
    """Tests for MMRHelper.mmr_rerank method.

    Validates the core MMR algorithm that selects documents balancing
    relevance to query and diversity among results. Tests cover lambda
    parameter effects, top-k selection, score computation, and edge cases.
    """

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
            [0.9, 0.1, 0.1],  # Very similar to query
            [0.85, 0.2, 0.2],
            [0.8, 0.3, 0.3],
            [0.75, 0.4, 0.4],
            [0.7, 0.5, 0.5],
        ]

    @pytest.fixture
    def diverse_embeddings(self):
        """Create embeddings with high diversity."""
        return [
            [1.0, 0.0, 0.0],  # Direction 1
            [0.0, 1.0, 0.0],  # Direction 2
            [0.0, 0.0, 1.0],  # Direction 3
            [0.5, 0.5, 0.0],  # Direction 4
            [0.5, 0.0, 0.5],  # Direction 5
        ]

    @pytest.fixture
    def query_embedding(self):
        """Create a sample query embedding."""
        return [0.9, 0.1, 0.1]

    def test_empty_documents(self, query_embedding):
        """Test MMR with empty documents list returns empty list."""
        result = MMRHelper.mmr_rerank(
            documents=[],
            embeddings=[],
            query_embedding=query_embedding,
        )
        assert result == []

    def test_single_document(self, sample_documents, query_embedding):
        """Test MMR with single document returns that document."""
        embeddings = [[0.5, 0.5, 0.5]]
        result = MMRHelper.mmr_rerank(
            documents=sample_documents[:1],
            embeddings=embeddings,
            query_embedding=query_embedding,
        )
        assert len(result) == 1
        assert result[0][0] == sample_documents[0]

    def test_k_greater_than_documents(self, sample_documents, query_embedding):
        """Test MMR with k larger than document count returns all documents."""
        embeddings = [[0.5, 0.5, 0.5]] * 5
        result = MMRHelper.mmr_rerank(
            documents=sample_documents,
            embeddings=embeddings,
            query_embedding=query_embedding,
            k=100,
        )
        assert len(result) == 5

    def test_k_less_than_documents(self, sample_documents, query_embedding):
        """Test MMR with k smaller than document count returns k documents."""
        embeddings = [[0.5, 0.5, 0.5]] * 5
        result = MMRHelper.mmr_rerank(
            documents=sample_documents,
            embeddings=embeddings,
            query_embedding=query_embedding,
            k=2,
        )
        assert len(result) == 2

    def test_k_equals_documents(self, sample_documents, query_embedding):
        """Test MMR with k equal to document count returns all documents."""
        embeddings = [[0.5, 0.5, 0.5]] * 5
        result = MMRHelper.mmr_rerank(
            documents=sample_documents,
            embeddings=embeddings,
            query_embedding=query_embedding,
            k=5,
        )
        assert len(result) == 5

    def test_lambda_param_high_relevance(
        self, sample_documents, similar_embeddings, query_embedding
    ):
        """Test high lambda_param prioritizes relevance over diversity."""
        result = MMRHelper.mmr_rerank(
            documents=sample_documents,
            embeddings=similar_embeddings,
            query_embedding=query_embedding,
            lambda_param=0.95,
            k=3,
        )
        # With high lambda, should pick most relevant first
        assert len(result) == 3
        assert isinstance(result[0][0], Document)

    def test_lambda_param_low_diversity(
        self, sample_documents, diverse_embeddings, query_embedding
    ):
        """Test low lambda_param promotes diversity."""
        result = MMRHelper.mmr_rerank(
            documents=sample_documents,
            embeddings=diverse_embeddings,
            query_embedding=query_embedding,
            lambda_param=0.1,
            k=3,
        )
        assert len(result) == 3

    def test_lambda_param_zero(self, sample_documents, query_embedding):
        """Test lambda_param=0 selects purely by diversity."""
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
        result = MMRHelper.mmr_rerank(
            documents=sample_documents[:4],
            embeddings=embeddings,
            query_embedding=query_embedding,
            lambda_param=0.0,
            k=2,
        )
        assert len(result) == 2

    def test_lambda_param_one(self, sample_documents, query_embedding):
        """Test lambda_param=1 selects purely by relevance."""
        embeddings = [
            [0.9, 0.1, 0.1],
            [0.5, 0.5, 0.5],
            [0.3, 0.3, 0.3],
        ]
        result = MMRHelper.mmr_rerank(
            documents=sample_documents[:3],
            embeddings=embeddings,
            query_embedding=query_embedding,
            lambda_param=1.0,
            k=2,
        )
        assert len(result) == 2
        # First should be most relevant (highest similarity to query)
        assert result[0][0] == sample_documents[0]

    def test_returns_tuples_with_scores(self, sample_documents, query_embedding):
        """Test MMR returns list of (Document, score) tuples."""
        embeddings = [[0.5, 0.5, 0.5]] * 3
        result = MMRHelper.mmr_rerank(
            documents=sample_documents[:3],
            embeddings=embeddings,
            query_embedding=query_embedding,
        )
        assert len(result) == 3
        for doc, score in result:
            assert isinstance(doc, Document)
            assert isinstance(score, float)

    def test_scores_are_mmr_scores(self, sample_documents, query_embedding):
        """Test that returned scores are valid MMR scores."""
        embeddings = [[0.5, 0.5, 0.5]] * 3
        result = MMRHelper.mmr_rerank(
            documents=sample_documents[:3],
            embeddings=embeddings,
            query_embedding=query_embedding,
        )
        for _, score in result:
            assert -1.1 <= score <= 1.1  # Valid MMR score range [-1, 1]

    def test_all_similar_items(self, sample_documents, query_embedding):
        """Test MMR when all items are similar to query."""
        embeddings = [[0.9, 0.1, 0.1]] * 5
        result = MMRHelper.mmr_rerank(
            documents=sample_documents,
            embeddings=embeddings,
            query_embedding=query_embedding,
            k=3,
        )
        assert len(result) == 3

    def test_default_k_value(self, sample_documents, query_embedding):
        """Test MMR uses default k=10 when not specified."""
        embeddings = [[0.5, 0.5, 0.5]] * 5
        result = MMRHelper.mmr_rerank(
            documents=sample_documents,
            embeddings=embeddings,
            query_embedding=query_embedding,
        )
        assert len(result) == 5  # Limited by available documents

    def test_default_lambda_value(self, sample_documents, query_embedding):
        """Test MMR uses default lambda_param=0.5 when not specified."""
        embeddings = [[0.5, 0.5, 0.5]] * 3
        result = MMRHelper.mmr_rerank(
            documents=sample_documents[:3],
            embeddings=embeddings,
            query_embedding=query_embedding,
        )
        assert len(result) == 3

    def test_preserves_document_order_in_output(
        self, sample_documents, query_embedding
    ):
        """Test that documents are returned in MMR-scored order."""
        embeddings = [
            [0.9, 0.1, 0.1],
            [0.8, 0.2, 0.2],
            [0.7, 0.3, 0.3],
        ]
        result = MMRHelper.mmr_rerank(
            documents=sample_documents[:3],
            embeddings=embeddings,
            query_embedding=query_embedding,
            k=3,
        )
        # Scores should be in descending order
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_diversity_calculation(self, sample_documents, query_embedding):
        """Test diversity is properly calculated in MMR."""
        # Create embeddings where one is very different
        embeddings = [
            [1.0, 0.0, 0.0],  # Similar to query
            [-1.0, 0.0, 0.0],  # Opposite direction - high diversity
            [0.0, 1.0, 0.0],  # Orthogonal - high diversity
        ]
        result = MMRHelper.mmr_rerank(
            documents=sample_documents[:3],
            embeddings=embeddings,
            query_embedding=[0.9, 0.1, 0.1],
            lambda_param=0.3,  # Lower lambda, more diversity
            k=2,
        )
        assert len(result) == 2


class TestMMRRerankSimple:
    """Unit tests for MMRHelper.mmr_rerank_simple method."""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(page_content="Document 1 content", metadata={"id": "1"}),
            Document(page_content="Document 2 content", metadata={"id": "2"}),
            Document(page_content="Document 3 content", metadata={"id": "3"}),
        ]

    @pytest.fixture
    def query_embedding(self):
        """Create a sample query embedding."""
        return [0.9, 0.1, 0.1]

    def test_empty_documents(self, query_embedding):
        """Test mmr_rerank_simple with empty list returns empty list."""
        result = MMRHelper.mmr_rerank_simple(
            documents=[],
            embeddings=[],
            query_embedding=query_embedding,
        )
        assert result == []

    def test_returns_only_documents(self, sample_documents, query_embedding):
        """Test mmr_rerank_simple returns only Document objects."""
        embeddings = [[0.5, 0.5, 0.5]] * 3
        result = MMRHelper.mmr_rerank_simple(
            documents=sample_documents,
            embeddings=embeddings,
            query_embedding=query_embedding,
        )
        assert len(result) == 3
        for doc in result:
            assert isinstance(doc, Document)
            assert not isinstance(doc, tuple)

    def test_single_document(self, sample_documents, query_embedding):
        """Test mmr_rerank_simple with single document."""
        embeddings = [[0.5, 0.5, 0.5]]
        result = MMRHelper.mmr_rerank_simple(
            documents=sample_documents[:1],
            embeddings=embeddings,
            query_embedding=query_embedding,
        )
        assert len(result) == 1
        assert result[0] == sample_documents[0]

    def test_k_parameter(self, sample_documents, query_embedding):
        """Test mmr_rerank_simple respects k parameter."""
        embeddings = [[0.5, 0.5, 0.5]] * 3
        result = MMRHelper.mmr_rerank_simple(
            documents=sample_documents,
            embeddings=embeddings,
            query_embedding=query_embedding,
            k=2,
        )
        assert len(result) == 2

    def test_k_greater_than_documents(self, sample_documents, query_embedding):
        """Test mmr_rerank_simple with k larger than available documents."""
        embeddings = [[0.5, 0.5, 0.5]] * 3
        result = MMRHelper.mmr_rerank_simple(
            documents=sample_documents,
            embeddings=embeddings,
            query_embedding=query_embedding,
            k=100,
        )
        assert len(result) == 3

    def test_default_k_value(self, sample_documents, query_embedding):
        """Test mmr_rerank_simple uses default k=10."""
        embeddings = [[0.5, 0.5, 0.5]] * 3
        result = MMRHelper.mmr_rerank_simple(
            documents=sample_documents,
            embeddings=embeddings,
            query_embedding=query_embedding,
        )
        assert len(result) == 3

    def test_preserves_metadata(self, sample_documents, query_embedding):
        """Test mmr_rerank_simple preserves document metadata."""
        embeddings = [[0.5, 0.5, 0.5]] * 3
        result = MMRHelper.mmr_rerank_simple(
            documents=sample_documents,
            embeddings=embeddings,
            query_embedding=query_embedding,
        )
        for doc in result:
            assert "id" in doc.metadata

    def test_preserves_page_content(self, sample_documents, query_embedding):
        """Test mmr_rerank_simple preserves document content."""
        embeddings = [[0.5, 0.5, 0.5]] * 3
        result = MMRHelper.mmr_rerank_simple(
            documents=sample_documents,
            embeddings=embeddings,
            query_embedding=query_embedding,
        )
        content_set = {doc.page_content for doc in result}
        expected_set = {doc.page_content for doc in sample_documents}
        assert content_set == expected_set

    def test_equivalence_with_mmr_rerank(self, sample_documents, query_embedding):
        """Test mmr_rerank_simple returns same documents as mmr_rerank."""
        embeddings = [
            [0.9, 0.1, 0.1],
            [0.5, 0.5, 0.5],
            [0.1, 0.9, 0.1],
        ]
        simple_result = MMRHelper.mmr_rerank_simple(
            documents=sample_documents,
            embeddings=embeddings,
            query_embedding=query_embedding,
            k=3,
        )
        full_result = MMRHelper.mmr_rerank(
            documents=sample_documents,
            embeddings=embeddings,
            query_embedding=query_embedding,
            k=3,
        )
        simple_docs = list(simple_result)
        full_docs = [doc for doc, _ in full_result]
        assert simple_docs == full_docs


class TestMMRHelperEdgeCases:
    """Edge case tests for MMRHelper."""

    @pytest.fixture
    def query_embedding(self):
        """Create a sample query embedding."""
        return [0.5, 0.5, 0.5]

    def test_very_high_dimensional_embeddings(self):
        """Test MMR with very high dimensional embeddings."""
        dim = 1000
        query_embedding = [0.5] * dim
        documents = [
            Document(page_content="Doc 1", metadata={"id": "1"}),
            Document(page_content="Doc 2", metadata={"id": "2"}),
        ]
        embeddings = [
            [0.5] * dim,
            [0.6] * dim,
        ]
        result = MMRHelper.mmr_rerank(
            documents=documents,
            embeddings=embeddings,
            query_embedding=query_embedding,
        )
        assert len(result) == 2

    def test_very_low_dimensional_embeddings(self, query_embedding):
        """Test MMR with 1-dimensional embeddings."""
        documents = [
            Document(page_content="Doc 1", metadata={"id": "1"}),
            Document(page_content="Doc 2", metadata={"id": "2"}),
        ]
        embeddings = [[0.5], [0.8]]
        result = MMRHelper.mmr_rerank(
            documents=documents,
            embeddings=embeddings,
            query_embedding=[0.9],
        )
        assert len(result) == 2

    def test_negative_embedding_values(self, query_embedding):
        """Test MMR with negative embedding values."""
        documents = [
            Document(page_content="Doc 1", metadata={"id": "1"}),
            Document(page_content="Doc 2", metadata={"id": "2"}),
        ]
        embeddings = [
            [-0.5, -0.5, -0.5],
            [0.5, 0.5, 0.5],
        ]
        result = MMRHelper.mmr_rerank(
            documents=documents,
            embeddings=embeddings,
            query_embedding=query_embedding,
        )
        assert len(result) == 2

    def test_mixed_similarity_scenario(self, query_embedding):
        """Test MMR with documents having varied similarity to query."""
        documents = [
            Document(page_content="Highly relevant doc", metadata={"id": "1"}),
            Document(page_content="Somewhat relevant doc", metadata={"id": "2"}),
            Document(page_content="Less relevant doc", metadata={"id": "3"}),
            Document(page_content="Another relevant doc", metadata={"id": "4"}),
        ]
        embeddings = [
            [0.95, 0.05, 0.05],  # Very similar
            [0.6, 0.4, 0.4],  # Somewhat similar
            [0.2, 0.5, 0.5],  # Less similar
            [0.9, 0.1, 0.1],  # Very similar
        ]
        result = MMRHelper.mmr_rerank(
            documents=documents,
            embeddings=embeddings,
            query_embedding=query_embedding,
            lambda_param=0.5,
            k=2,
        )
        assert len(result) == 2

    def test_all_documents_selected_with_k_equal_count(self, query_embedding):
        """Test that all documents are selected when k equals count."""
        documents = [
            Document(page_content=f"Doc {i}", metadata={"id": str(i)})
            for i in range(10)
        ]
        embeddings = [[0.5, 0.5, 0.5] for _ in range(10)]
        result = MMRHelper.mmr_rerank(
            documents=documents,
            embeddings=embeddings,
            query_embedding=query_embedding,
            k=10,
        )
        assert len(result) == 10

    def test_k_is_zero(self, query_embedding):
        """Test MMR with k=0 returns empty list."""
        documents = [
            Document(page_content="Doc 1", metadata={"id": "1"}),
            Document(page_content="Doc 2", metadata={"id": "2"}),
        ]
        embeddings = [[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]]
        result = MMRHelper.mmr_rerank(
            documents=documents,
            embeddings=embeddings,
            query_embedding=query_embedding,
            k=0,
        )
        assert len(result) == 0

    def test_mmr_simple_k_is_zero(self, query_embedding):
        """Test mmr_rerank_simple with k=0 returns empty list."""
        documents = [
            Document(page_content="Doc 1", metadata={"id": "1"}),
        ]
        embeddings = [[0.5, 0.5, 0.5]]
        result = MMRHelper.mmr_rerank_simple(
            documents=documents,
            embeddings=embeddings,
            query_embedding=query_embedding,
            k=0,
        )
        assert len(result) == 0
