"""Tests for reranking utilities using cross-encoder models.

This module tests the RerankerHelper class which provides utilities for
reranking retrieved documents using cross-encoder models. Reranking improves
precision by scoring query-document pairs directly rather than using embeddings.

RerankerHelper Methods:
    create_reranker: Factory for HuggingFaceCrossEncoder
    rerank: Rerank documents and return top-k
    rerank_with_scores: Rerank and return (document, score) tuples

Cross-Encoder Scoring:
    Cross-encoders jointly encode query and document, producing a relevance
    score. More accurate than bi-encoder similarity but slower (O(n) inference).

Test Classes:
    TestCreateReranker: Reranker factory with model configuration
    TestRerank: Document reranking with top-k selection
    TestRerankWithScores: Reranking with score preservation
    TestRerankerHelperEdgeCases: Long queries, unicode, edge conditions

Default Model: cross-encoder/ms-marco-MiniLM-L-6-v2

All tests mock HuggingFaceCrossEncoder to avoid model downloads.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from vectordb.langchain.utils.reranker import RerankerHelper


class TestCreateReranker:
    """Tests for RerankerHelper.create_reranker factory method.

    Validates creation of HuggingFaceCrossEncoder instances from config.
    Default model is cross-encoder/ms-marco-MiniLM-L-6-v2, optimized for
    passage reranking tasks.
    """

    @patch("vectordb.langchain.utils.reranker.HuggingFaceCrossEncoder")
    def test_default_model(self, mock_cross_encoder):
        """Test create_reranker uses default model when not specified."""
        config = {}
        RerankerHelper.create_reranker(config)
        mock_cross_encoder.assert_called_once_with(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    @patch("vectordb.langchain.utils.reranker.HuggingFaceCrossEncoder")
    def test_custom_model(self, mock_cross_encoder):
        """Test create_reranker uses custom model from config."""
        config = {"reranker": {"model": "custom/reranker-model"}}
        RerankerHelper.create_reranker(config)
        mock_cross_encoder.assert_called_once_with(model_name="custom/reranker-model")

    @patch("vectordb.langchain.utils.reranker.HuggingFaceCrossEncoder")
    def test_empty_reranker_config(self, mock_cross_encoder):
        """Test create_reranker with empty reranker section."""
        config = {"other_config": "value"}
        RerankerHelper.create_reranker(config)
        mock_cross_encoder.assert_called_once_with(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    @patch("vectordb.langchain.utils.reranker.HuggingFaceCrossEncoder")
    def test_returns_cross_encoder_instance(self, mock_cross_encoder):
        """Test create_reranker returns HuggingFaceCrossEncoder instance."""
        mock_instance = MagicMock()
        mock_cross_encoder.return_value = mock_instance
        config = {}
        result = RerankerHelper.create_reranker(config)
        assert result == mock_instance

    @patch("vectordb.langchain.utils.reranker.HuggingFaceCrossEncoder")
    def test_model_name_only_parameter(self, mock_cross_encoder):
        """Test create_reranker only passes model_name to cross encoder."""
        config = {"reranker": {"model": "test/model"}}
        RerankerHelper.create_reranker(config)
        mock_cross_encoder.assert_called_once()
        call_kwargs = mock_cross_encoder.call_args[1]
        assert "model_name" in call_kwargs
        assert call_kwargs["model_name"] == "test/model"


class TestRerank:
    """Tests for RerankerHelper.rerank document reranking.

    Validates reranking of documents by cross-encoder relevance scores.
    Documents are sorted by score descending and optionally limited to top-k.

    Behavior:
        - Empty list returns empty list
        - Sorts by cross-encoder score descending
        - top_k limits returned documents
        - Preserves document metadata and content
    """

    @pytest.fixture
    def mock_reranker(self):
        """Create a mock cross-encoder reranker."""
        return MagicMock()

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                page_content="Document about Python programming", metadata={"id": "1"}
            ),
            Document(
                page_content="Document about JavaScript development",
                metadata={"id": "2"},
            ),
            Document(
                page_content="Document about machine learning", metadata={"id": "3"}
            ),
        ]

    def test_empty_documents_returns_empty_list(self, mock_reranker):
        """Test rerank with empty documents returns empty list."""
        result = RerankerHelper.rerank(mock_reranker, "test query", [])
        assert result == []

    def test_single_document(self, mock_reranker, sample_documents):
        """Test rerank with single document."""
        mock_reranker.rank.return_value = [0.9]
        result = RerankerHelper.rerank(
            mock_reranker, "test query", [sample_documents[0]]
        )
        assert len(result) == 1
        assert result[0] == sample_documents[0]

    def test_sorts_by_score_descending(self, mock_reranker, sample_documents):
        """Test rerank sorts documents by score in descending order."""
        # Mock scores: doc2 highest, doc3 middle, doc1 lowest
        mock_reranker.rank.return_value = [0.3, 0.9, 0.6]
        result = RerankerHelper.rerank(mock_reranker, "test query", sample_documents)
        assert result[0].page_content == "Document about JavaScript development"
        assert result[1].page_content == "Document about machine learning"
        assert result[2].page_content == "Document about Python programming"

    def test_top_k_limits_results(self, mock_reranker, sample_documents):
        """Test rerank with top_k limits number of results."""
        mock_reranker.rank.return_value = [0.9, 0.6, 0.3]
        result = RerankerHelper.rerank(
            mock_reranker, "test query", sample_documents, top_k=2
        )
        assert len(result) == 2

    def test_top_k_greater_than_documents(self, mock_reranker, sample_documents):
        """Test rerank with top_k larger than document count."""
        mock_reranker.rank.return_value = [0.9, 0.6, 0.3]
        result = RerankerHelper.rerank(
            mock_reranker, "test query", sample_documents, top_k=10
        )
        assert len(result) == 3

    def test_top_k_equals_documents(self, mock_reranker, sample_documents):
        """Test rerank with top_k equal to document count."""
        mock_reranker.rank.return_value = [0.9, 0.6, 0.3]
        result = RerankerHelper.rerank(
            mock_reranker, "test query", sample_documents, top_k=3
        )
        assert len(result) == 3

    def test_top_k_none_returns_all(self, mock_reranker, sample_documents):
        """Test rerank with top_k=None returns all documents."""
        mock_reranker.rank.return_value = [0.9, 0.6, 0.3]
        result = RerankerHelper.rerank(
            mock_reranker, "test query", sample_documents, top_k=None
        )
        assert len(result) == 3

    def test_preserves_metadata(self, mock_reranker, sample_documents):
        """Test rerank preserves document metadata."""
        mock_reranker.rank.return_value = [0.9, 0.6, 0.3]
        result = RerankerHelper.rerank(mock_reranker, "test query", sample_documents)
        for doc in result:
            assert hasattr(doc, "metadata")
            assert "id" in doc.metadata

    def test_preserves_page_content(self, mock_reranker, sample_documents):
        """Test rerank preserves document page content."""
        mock_reranker.rank.return_value = [0.9, 0.6, 0.3]
        result = RerankerHelper.rerank(mock_reranker, "test query", sample_documents)
        original_contents = {doc.page_content for doc in sample_documents}
        result_contents = {doc.page_content for doc in result}
        assert result_contents == original_contents

    def test_calls_reranker_rank(self, mock_reranker, sample_documents):
        """Test rerank calls the cross-encoder's rank method."""
        mock_reranker.rank.return_value = [0.5] * len(sample_documents)
        query = "test query"
        RerankerHelper.rerank(mock_reranker, query, sample_documents)
        mock_reranker.rank.assert_called_once()
        call_args = mock_reranker.rank.call_args[0][0]
        # Should be list of [query, doc_content] pairs
        assert len(call_args) == len(sample_documents)
        for pair in call_args:
            assert len(pair) == 2
            assert pair[0] == query

    def test_equal_scores_preserves_order(self, mock_reranker, sample_documents):
        """Test rerank with equal scores preserves input order for ties."""
        mock_reranker.rank.return_value = [0.5, 0.5, 0.5]
        result = RerankerHelper.rerank(mock_reranker, "test query", sample_documents)
        # Original order should be preserved for equal scores
        assert result[0] == sample_documents[0]
        assert result[1] == sample_documents[1]
        assert result[2] == sample_documents[2]

    def test_returns_document_objects(self, mock_reranker, sample_documents):
        """Test rerank returns Document objects, not tuples."""
        mock_reranker.rank.return_value = [0.9, 0.6, 0.3]
        result = RerankerHelper.rerank(mock_reranker, "test query", sample_documents)
        for doc in result:
            assert isinstance(doc, Document)
            assert not isinstance(doc, tuple)


class TestRerankWithScores:
    """Tests for RerankerHelper.rerank_with_scores with score preservation.

    Validates reranking that returns (Document, score) tuples. Useful when
    downstream processing needs relevance scores for filtering or display.

    Return Value:
        List of (Document, float) tuples sorted by score descending.
    """

    @pytest.fixture
    def mock_reranker(self):
        """Create a mock cross-encoder reranker."""
        return MagicMock()

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(page_content="Doc 1 content", metadata={"id": "1"}),
            Document(page_content="Doc 2 content", metadata={"id": "2"}),
            Document(page_content="Doc 3 content", metadata={"id": "3"}),
        ]

    def test_empty_documents_returns_empty_list(self, mock_reranker):
        """Test rerank_with_scores with empty documents returns empty list."""
        result = RerankerHelper.rerank_with_scores(mock_reranker, "test query", [])
        assert result == []

    def test_returns_tuples_with_scores(self, mock_reranker, sample_documents):
        """Test rerank_with_scores returns list of (Document, score) tuples."""
        mock_reranker.rank.return_value = [0.9, 0.6, 0.3]
        result = RerankerHelper.rerank_with_scores(
            mock_reranker, "test query", sample_documents
        )
        assert len(result) == 3
        for doc, score in result:
            assert isinstance(doc, Document)
            assert isinstance(score, float)

    def test_sorts_by_score_descending(self, mock_reranker, sample_documents):
        """Test rerank_with_scores sorts by score descending."""
        mock_reranker.rank.return_value = [0.3, 0.9, 0.6]
        result = RerankerHelper.rerank_with_scores(
            mock_reranker, "test query", sample_documents
        )
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(self, mock_reranker, sample_documents):
        """Test rerank_with_scores with top_k limits results."""
        mock_reranker.rank.return_value = [0.9, 0.6, 0.3]
        result = RerankerHelper.rerank_with_scores(
            mock_reranker, "test query", sample_documents, top_k=2
        )
        assert len(result) == 2

    def test_top_k_greater_than_documents(self, mock_reranker, sample_documents):
        """Test rerank_with_scores with top_k larger than document count."""
        mock_reranker.rank.return_value = [0.9, 0.6, 0.3]
        result = RerankerHelper.rerank_with_scores(
            mock_reranker, "test query", sample_documents, top_k=10
        )
        assert len(result) == 3

    def test_top_k_none_returns_all(self, mock_reranker, sample_documents):
        """Test rerank_with_scores with top_k=None returns all."""
        mock_reranker.rank.return_value = [0.9, 0.6, 0.3]
        result = RerankerHelper.rerank_with_scores(
            mock_reranker, "test query", sample_documents, top_k=None
        )
        assert len(result) == 3

    def test_scores_are_correct_type(self, mock_reranker, sample_documents):
        """Test returned scores are floats."""
        mock_reranker.rank.return_value = [0.9, 0.6, 0.3]
        result = RerankerHelper.rerank_with_scores(
            mock_reranker, "test query", sample_documents
        )
        for _, score in result:
            assert isinstance(score, float)

    def test_scores_in_valid_range(self, mock_reranker, sample_documents):
        """Test scores are in valid range (0 to 1 typically)."""
        mock_reranker.rank.return_value = [0.9, 0.6, 0.3]
        result = RerankerHelper.rerank_with_scores(
            mock_reranker, "test query", sample_documents
        )
        for _, score in result:
            assert 0.0 <= score <= 1.0

    def test_preserves_metadata(self, mock_reranker, sample_documents):
        """Test rerank_with_scores preserves document metadata."""
        mock_reranker.rank.return_value = [0.9, 0.6, 0.3]
        result = RerankerHelper.rerank_with_scores(
            mock_reranker, "test query", sample_documents
        )
        for doc, _ in result:
            assert hasattr(doc, "metadata")
            assert "id" in doc.metadata

    def test_document_matches_original(self, mock_reranker, sample_documents):
        """Test that documents in result match original documents."""
        mock_reranker.rank.return_value = [0.9, 0.6, 0.3]
        result = RerankerHelper.rerank_with_scores(
            mock_reranker, "test query", sample_documents
        )
        result_docs = {doc.page_content for doc, _ in result}
        original_contents = {doc.page_content for doc in sample_documents}
        assert result_docs == original_contents

    def test_equal_scores_preserves_order(self, mock_reranker, sample_documents):
        """Test rerank_with_scores with equal scores preserves input order."""
        mock_reranker.rank.return_value = [0.5, 0.5, 0.5]
        result = RerankerHelper.rerank_with_scores(
            mock_reranker, "test query", sample_documents
        )
        # Should maintain original order for ties
        for i, (doc, _) in enumerate(result):
            assert doc == sample_documents[i]


class TestRerankerHelperEdgeCases:
    """Edge case tests for RerankerHelper boundary conditions.

    Validates handling of edge cases including top_k=0, very long queries,
    special characters, unicode content, and model name formatting.

    Edge Cases Covered:
        - top_k=0 returns empty list
        - Very long queries (1000+ words)
        - Special characters in document content
        - Unicode characters (CJK, emoji)
        - Model names with special characters
    """

    @pytest.fixture
    def mock_reranker(self):
        """Create a mock cross-encoder reranker."""
        return MagicMock()

    def test_rerank_with_single_doc_top_k_zero(self, mock_reranker):
        """Test rerank with top_k=0 returns empty list."""
        doc = Document(page_content="Doc", metadata={})
        mock_reranker.rank.return_value = [0.9]
        result = RerankerHelper.rerank(mock_reranker, "query", [doc], top_k=0)
        assert result == []

    def test_rerank_with_scores_top_k_zero(self, mock_reranker):
        """Test rerank_with_scores with top_k=0 returns empty list."""
        doc = Document(page_content="Doc", metadata={})
        mock_reranker.rank.return_value = [0.9]
        result = RerankerHelper.rerank_with_scores(
            mock_reranker, "query", [doc], top_k=0
        )
        assert result == []

    def test_rerank_with_very_long_query(self, mock_reranker):
        """Test rerank with very long query string."""
        doc = Document(page_content="Doc content", metadata={})
        mock_reranker.rank.return_value = [0.5]
        long_query = " ".join(["test"] * 1000)
        result = RerankerHelper.rerank(mock_reranker, long_query, [doc])
        assert len(result) == 1
        # Verify the query was passed to rank method
        call_args = mock_reranker.rank.call_args[0][0][0]
        assert call_args[0] == long_query

    def test_rerank_with_special_characters_in_content(self, mock_reranker):
        """Test rerank with special characters in document content."""
        doc = Document(
            page_content="Doc with special chars: @#$%^&*()_+{}|:<>?", metadata={}
        )
        mock_reranker.rank.return_value = [0.5]
        result = RerankerHelper.rerank(mock_reranker, "query", [doc])
        assert len(result) == 1
        assert result[0].page_content == doc.page_content

    def test_rerank_with_unicode_content(self, mock_reranker):
        """Test rerank with unicode characters in document content."""
        doc = Document(page_content="Doc with unicode: 你好世界 🌍 αβγδ", metadata={})
        mock_reranker.rank.return_value = [0.5]
        result = RerankerHelper.rerank(mock_reranker, "query", [doc])
        assert len(result) == 1
        assert result[0].page_content == doc.page_content

    def test_create_reranker_with_special_model_name(self, mock_reranker):
        """Test create_reranker with model name containing special chars."""
        with patch("vectordb.langchain.utils.reranker.HuggingFaceCrossEncoder") as mock:
            config = {"reranker": {"model": "model/name-with-dashes"}}
            RerankerHelper.create_reranker(config)
            mock.assert_called_once_with(model_name="model/name-with-dashes")

    def test_reranker_rank_called_with_correct_pairs(self, mock_reranker):
        """Test that rank is called with correct query-document pairs."""
        documents = [
            Document(page_content="Content A", metadata={}),
            Document(page_content="Content B", metadata={}),
        ]
        mock_reranker.rank.return_value = [0.5, 0.3]
        query = "test query"
        RerankerHelper.rerank(mock_reranker, query, documents)
        call_args = mock_reranker.rank.call_args[0][0]
        assert call_args == [
            ["test query", "Content A"],
            ["test query", "Content B"],
        ]
