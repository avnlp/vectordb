"""Tests for sparse embeddings utilities using SparseEncoder (LangChain)."""

from unittest.mock import MagicMock, patch

import pytest

from vectordb.langchain.utils.sparse_embeddings import SparseEmbedder


@pytest.fixture
def mock_sparse_encoder():
    """Fixture to mock SparseEncoder globally."""
    with patch("vectordb.langchain.utils.sparse_embeddings.SparseEncoder") as mock:
        yield mock


class TestSparseEmbedderInit:
    """Unit tests for SparseEmbedder initialization."""

    def test_init_default(self, mock_sparse_encoder):
        """Test initialization with default parameters."""
        mock_sparse_encoder.return_value = MagicMock()
        embedder = SparseEmbedder()

        assert embedder.model_name == "naver/splade-v2"
        assert embedder.device == "cpu"
        mock_sparse_encoder.assert_called_once_with("naver/splade-v2", device="cpu")

    def test_init_custom_model(self, mock_sparse_encoder):
        """Test initialization with custom model name."""
        mock_sparse_encoder.return_value = MagicMock()
        embedder = SparseEmbedder(model_name="naver/splade-v3", device="cpu")

        assert embedder.model_name == "naver/splade-v3"
        mock_sparse_encoder.assert_called_once_with("naver/splade-v3", device="cpu")

    def test_init_cuda_device(self, mock_sparse_encoder):
        """Test initialization with CUDA device."""
        mock_sparse_encoder.return_value = MagicMock()
        embedder = SparseEmbedder(model_name="naver/splade-v2", device="cuda")

        assert embedder.device == "cuda"
        mock_sparse_encoder.assert_called_once_with("naver/splade-v2", device="cuda")


class TestEmbedDocuments:
    """Unit tests for SparseEmbedder.embed_documents method."""

    def test_embed_documents_basic(self, mock_sparse_encoder):
        """Test embedding documents returns sparse dicts."""
        mock_encoder = MagicMock()
        mock_sparse_encoder.return_value = mock_encoder

        # Mock tensor output: batch_size=2, vocab_size=100
        mock_tensor = MagicMock()
        mock_tensor.__iter__ = MagicMock(
            return_value=iter(
                [
                    MagicMock(
                        tolist=MagicMock(
                            return_value=[
                                0.5 if i in [10, 20] else 0.0 for i in range(100)
                            ]
                        )
                    ),
                    MagicMock(
                        tolist=MagicMock(
                            return_value=[
                                0.3 if i in [5, 15, 25] else 0.0 for i in range(100)
                            ]
                        )
                    ),
                ]
            )
        )
        mock_encoder.encode_document.return_value = mock_tensor

        embedder = SparseEmbedder()
        result = embedder.embed_documents(["text one", "text two"])

        assert len(result) == 2
        assert isinstance(result[0], dict)
        assert isinstance(result[1], dict)
        assert len(result[0]) == 2  # Only 2 non-zero entries
        assert len(result[1]) == 3  # Only 3 non-zero entries

    def test_embed_documents_empty_list(self, mock_sparse_encoder):
        """Test embedding empty list returns empty list."""
        mock_sparse_encoder.return_value = MagicMock()
        embedder = SparseEmbedder()
        result = embedder.embed_documents([])

        assert result == []

    def test_embed_documents_sparsity(self, mock_sparse_encoder):
        """Test that output is actually sparse (only non-zero weights)."""
        mock_encoder = MagicMock()
        mock_sparse_encoder.return_value = mock_encoder

        # Create a sparse tensor (mostly zeros)
        dense_vector = [0.0] * 1000
        dense_vector[5] = 0.8
        dense_vector[42] = 0.6
        dense_vector[999] = 0.2

        mock_tensor = MagicMock()
        mock_tensor.__iter__ = MagicMock(
            return_value=iter([MagicMock(tolist=MagicMock(return_value=dense_vector))])
        )
        mock_encoder.encode_document.return_value = mock_tensor

        embedder = SparseEmbedder()
        result = embedder.embed_documents(["test"])

        assert len(result) == 1
        sparse_vec = result[0]
        # Should only have 3 non-zero entries
        assert len(sparse_vec) == 3
        assert "5" in sparse_vec
        assert "42" in sparse_vec
        assert "999" in sparse_vec


class TestEmbedQuery:
    """Unit tests for SparseEmbedder.embed_query method."""

    def test_embed_query_basic(self, mock_sparse_encoder):
        """Test query embedding returns sparse dict."""
        mock_encoder = MagicMock()
        mock_sparse_encoder.return_value = mock_encoder

        # Mock tensor output: shape (1, vocab_size)
        dense_vector = [0.0] * 100
        dense_vector[10] = 0.9
        dense_vector[50] = 0.7

        mock_tensor = MagicMock()
        mock_tensor.__getitem__ = MagicMock(
            return_value=MagicMock(tolist=MagicMock(return_value=dense_vector))
        )
        mock_encoder.encode_query.return_value = mock_tensor

        embedder = SparseEmbedder()
        result = embedder.embed_query("test query")

        assert isinstance(result, dict)
        assert "10" in result
        assert "50" in result
        assert len(result) == 2

    def test_embed_query_empty_string(self, mock_sparse_encoder):
        """Test embedding empty query returns empty dict."""
        mock_sparse_encoder.return_value = MagicMock()
        embedder = SparseEmbedder()
        result = embedder.embed_query("")

        assert result == {}

    def test_embed_query_called_with_correct_method(self, mock_sparse_encoder):
        """Test that encode_query method is called."""
        mock_encoder = MagicMock()
        mock_sparse_encoder.return_value = mock_encoder

        mock_tensor = MagicMock()
        mock_tensor.__getitem__ = MagicMock(
            return_value=MagicMock(tolist=MagicMock(return_value=[0.0] * 100))
        )
        mock_encoder.encode_query.return_value = mock_tensor

        embedder = SparseEmbedder()
        embedder.embed_query("test")

        mock_encoder.encode_query.assert_called_once_with("test")


class TestEmbedDocumentsNormalized:
    """Unit tests for SparseEmbedder.embed_documents_normalized method."""

    def test_normalize_documents(self, mock_sparse_encoder):
        """Test L2 normalization of document embeddings."""
        mock_encoder = MagicMock()
        mock_sparse_encoder.return_value = mock_encoder

        # Create vector with known norm
        # {0: 3.0, 1: 4.0} has norm = sqrt(9 + 16) = 5
        dense_vector = [0.0] * 10
        dense_vector[0] = 3.0
        dense_vector[1] = 4.0

        mock_tensor = MagicMock()
        mock_tensor.__iter__ = MagicMock(
            return_value=iter([MagicMock(tolist=MagicMock(return_value=dense_vector))])
        )
        mock_encoder.encode_document.return_value = mock_tensor

        embedder = SparseEmbedder()
        result = embedder.embed_documents_normalized(["test"])

        assert len(result) == 1
        normalized_vec = result[0]
        assert "0" in normalized_vec
        assert "1" in normalized_vec
        # After normalization: 3/5 = 0.6, 4/5 = 0.8
        assert pytest.approx(normalized_vec["0"], rel=1e-5) == 0.6
        assert pytest.approx(normalized_vec["1"], rel=1e-5) == 0.8

    def test_normalize_empty_documents(self, mock_sparse_encoder):
        """Test normalization with empty input."""
        mock_sparse_encoder.return_value = MagicMock()
        embedder = SparseEmbedder()
        result = embedder.embed_documents_normalized([])

        assert result == []

    def test_normalize_zero_vector(self, mock_sparse_encoder):
        """Test normalization of zero vector."""
        mock_encoder = MagicMock()
        mock_sparse_encoder.return_value = mock_encoder

        mock_tensor = MagicMock()
        mock_tensor.__iter__ = MagicMock(
            return_value=iter([MagicMock(tolist=MagicMock(return_value=[0.0] * 10))])
        )
        mock_encoder.encode_document.return_value = mock_tensor

        embedder = SparseEmbedder()
        result = embedder.embed_documents_normalized(["test"])

        # Zero vector normalization returns empty dict
        assert result[0] == {}


class TestEmbedQueryNormalized:
    """Unit tests for SparseEmbedder.embed_query_normalized method."""

    def test_normalize_query(self, mock_sparse_encoder):
        """Test L2 normalization of query embedding."""
        mock_encoder = MagicMock()
        mock_sparse_encoder.return_value = mock_encoder

        # Vector {0: 0.6, 1: 0.8} has norm = 1.0
        dense_vector = [0.0] * 10
        dense_vector[0] = 0.6
        dense_vector[1] = 0.8

        mock_tensor = MagicMock()
        mock_tensor.__getitem__ = MagicMock(
            return_value=MagicMock(tolist=MagicMock(return_value=dense_vector))
        )
        mock_encoder.encode_query.return_value = mock_tensor

        embedder = SparseEmbedder()
        result = embedder.embed_query_normalized("test query")

        assert isinstance(result, dict)
        assert "0" in result
        assert "1" in result
        # Already has norm=1.0, so should be unchanged
        assert pytest.approx(result["0"], rel=1e-5) == 0.6
        assert pytest.approx(result["1"], rel=1e-5) == 0.8

    def test_normalize_empty_query(self, mock_sparse_encoder):
        """Test normalization of empty query."""
        mock_sparse_encoder.return_value = MagicMock()
        embedder = SparseEmbedder()
        result = embedder.embed_query_normalized("")

        assert result == {}


class TestSparseEmbedderEdgeCases:
    """Edge case tests for SparseEmbedder."""

    def test_very_large_sparse_vector(self, mock_sparse_encoder):
        """Test handling of large sparse vectors."""
        mock_encoder = MagicMock()
        mock_sparse_encoder.return_value = mock_encoder

        # Large vocab, sparse vector
        dense_vector = [0.0] * 30000
        dense_vector[100] = 0.5
        dense_vector[5000] = 0.3
        dense_vector[29999] = 0.2

        mock_tensor = MagicMock()
        mock_tensor.__iter__ = MagicMock(
            return_value=iter([MagicMock(tolist=MagicMock(return_value=dense_vector))])
        )
        mock_encoder.encode_document.return_value = mock_tensor

        embedder = SparseEmbedder()
        result = embedder.embed_documents(["test"])

        assert len(result) == 1
        assert len(result[0]) == 3

    def test_all_zeros_vector(self, mock_sparse_encoder):
        """Test handling of all-zero vector."""
        mock_encoder = MagicMock()
        mock_sparse_encoder.return_value = mock_encoder

        dense_vector = [0.0] * 1000

        mock_tensor = MagicMock()
        mock_tensor.__iter__ = MagicMock(
            return_value=iter([MagicMock(tolist=MagicMock(return_value=dense_vector))])
        )
        mock_encoder.encode_document.return_value = mock_tensor

        embedder = SparseEmbedder()
        result = embedder.embed_documents(["empty"])

        assert len(result) == 1
        assert result[0] == {}

    def test_single_nonzero_weight(self, mock_sparse_encoder):
        """Test vector with single non-zero weight."""
        mock_encoder = MagicMock()
        mock_sparse_encoder.return_value = mock_encoder

        dense_vector = [0.0] * 1000
        dense_vector[42] = 1.0

        mock_tensor = MagicMock()
        mock_tensor.__iter__ = MagicMock(
            return_value=iter([MagicMock(tolist=MagicMock(return_value=dense_vector))])
        )
        mock_encoder.encode_document.return_value = mock_tensor

        embedder = SparseEmbedder()
        result = embedder.embed_documents(["test"])

        assert len(result[0]) == 1
        assert result[0]["42"] == 1.0
