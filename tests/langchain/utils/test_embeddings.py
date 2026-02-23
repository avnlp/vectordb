"""Tests for embeddings utilities using HuggingFace models.

This module tests the EmbedderHelper class which provides utilities for
creating and using text embedding models in LangChain pipelines. Embeddings
convert text into dense vector representations for similarity search.

EmbedderHelper Methods:
    create_embedder: Factory for HuggingFaceEmbeddings with config
    embed_documents: Batch embed multiple documents
    embed_query: Embed a single query string

Test Classes:
    TestCreateEmbedder: Embedder factory with model, device, batch size config
    TestEmbedDocuments: Batch document embedding with metadata preservation
    TestEmbedQuery: Single query embedding
    TestEmbedderHelperEdgeCases: GPU devices, long texts, document ordering

Configuration:
    embeddings.model: HuggingFace model name (default: all-MiniLM-L6-v2)
    embeddings.device: Compute device - cpu, cuda, mps (default: cpu)
    embeddings.batch_size: Encoding batch size (default: 32)

All tests mock HuggingFaceEmbeddings to avoid model downloads.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from vectordb.langchain.utils.embeddings import EmbedderHelper


class TestCreateEmbedder:
    """Tests for EmbedderHelper.create_embedder factory method.

    Validates creation of HuggingFaceEmbeddings instances from config dicts.
    Supports model selection, device placement, and batch size configuration.

    Default Configuration:
        model: sentence-transformers/all-MiniLM-L6-v2
        device: cpu
        batch_size: 32
    """

    @patch("vectordb.langchain.utils.embeddings.HuggingFaceEmbeddings")
    def test_create_embedder_with_defaults(self, mock_hf_embeddings):
        """Test creating embedder with default values."""
        mock_embedder = MagicMock()
        mock_hf_embeddings.return_value = mock_embedder

        config = {}
        result = EmbedderHelper.create_embedder(config)

        mock_hf_embeddings.assert_called_once_with(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 32},
        )
        assert result == mock_embedder

    @patch("vectordb.langchain.utils.embeddings.HuggingFaceEmbeddings")
    def test_create_embedder_with_custom_model(self, mock_hf_embeddings):
        """Test creating embedder with custom model."""
        mock_embedder = MagicMock()
        mock_hf_embeddings.return_value = mock_embedder

        config = {
            "embeddings": {
                "model": "custom-model/all-mpnet-base-v2",
            }
        }
        EmbedderHelper.create_embedder(config)

        mock_hf_embeddings.assert_called_once()
        call_kwargs = mock_hf_embeddings.call_args[1]
        assert call_kwargs["model_name"] == "custom-model/all-mpnet-base-v2"

    @patch("vectordb.langchain.utils.embeddings.HuggingFaceEmbeddings")
    def test_create_embedder_with_custom_device(self, mock_hf_embeddings):
        """Test creating embedder with custom device."""
        mock_embedder = MagicMock()
        mock_hf_embeddings.return_value = mock_embedder

        config = {
            "embeddings": {
                "device": "cuda",
            }
        }
        EmbedderHelper.create_embedder(config)

        call_kwargs = mock_hf_embeddings.call_args[1]
        assert call_kwargs["model_kwargs"]["device"] == "cuda"

    @patch("vectordb.langchain.utils.embeddings.HuggingFaceEmbeddings")
    def test_create_embedder_with_custom_batch_size(self, mock_hf_embeddings):
        """Test creating embedder with custom batch size."""
        mock_embedder = MagicMock()
        mock_hf_embeddings.return_value = mock_embedder

        config = {
            "embeddings": {
                "batch_size": 64,
            }
        }
        EmbedderHelper.create_embedder(config)

        call_kwargs = mock_hf_embeddings.call_args[1]
        assert call_kwargs["encode_kwargs"]["batch_size"] == 64

    @patch("vectordb.langchain.utils.embeddings.HuggingFaceEmbeddings")
    def test_create_embedder_with_all_custom_values(self, mock_hf_embeddings):
        """Test creating embedder with all custom values."""
        mock_embedder = MagicMock()
        mock_hf_embeddings.return_value = mock_embedder

        config = {
            "embeddings": {
                "model": "custom-model",
                "device": "cuda",
                "batch_size": 128,
            }
        }
        EmbedderHelper.create_embedder(config)

        call_kwargs = mock_hf_embeddings.call_args[1]
        assert call_kwargs["model_name"] == "custom-model"
        assert call_kwargs["model_kwargs"]["device"] == "cuda"
        assert call_kwargs["encode_kwargs"]["batch_size"] == 128

    @patch("vectordb.langchain.utils.embeddings.HuggingFaceEmbeddings")
    def test_create_embedder_empty_embeddings_config(self, mock_hf_embeddings):
        """Test creating embedder when embeddings key exists but is empty."""
        mock_embedder = MagicMock()
        mock_hf_embeddings.return_value = mock_embedder

        config = {"embeddings": {}}
        EmbedderHelper.create_embedder(config)

        # Should use defaults
        call_kwargs = mock_hf_embeddings.call_args[1]
        assert call_kwargs["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"


class TestEmbedDocuments:
    """Tests for EmbedderHelper.embed_documents batch embedding.

    Validates batch embedding of LangChain Document objects. The method extracts
    page_content from documents, generates embeddings, and returns both the
    original documents and their corresponding embeddings.

    Return Value:
        Tuple of (documents, embeddings) where embeddings[i] corresponds to
        documents[i]. Document metadata is preserved.
    """

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(page_content="Document 1 content", metadata={"source": "doc1"}),
            Document(page_content="Document 2 content", metadata={"source": "doc2"}),
            Document(page_content="Document 3 content", metadata={"source": "doc3"}),
        ]

    @patch("vectordb.langchain.utils.embeddings.HuggingFaceEmbeddings")
    def test_embed_documents(self, mock_hf_embeddings, sample_documents):
        """Test embedding multiple documents."""
        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
        mock_hf_embeddings.return_value = mock_embedder

        config = {}
        embedder = EmbedderHelper.create_embedder(config)

        result = EmbedderHelper.embed_documents(embedder, sample_documents)

        assert len(result) == 2
        docs, embeddings = result
        assert docs == sample_documents
        assert len(embeddings) == 3
        mock_embedder.embed_documents.assert_called_once()

    @patch("vectordb.langchain.utils.embeddings.HuggingFaceEmbeddings")
    def test_embed_documents_extracts_page_content(
        self, mock_hf_embeddings, sample_documents
    ):
        """Test that embed_documents extracts page_content from documents."""
        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_hf_embeddings.return_value = mock_embedder

        config = {}
        embedder = EmbedderHelper.create_embedder(config)

        EmbedderHelper.embed_documents(embedder, sample_documents)

        # Verify embed_documents was called with page_content strings
        call_args = mock_embedder.embed_documents.call_args[0][0]
        assert len(call_args) == 3
        assert call_args[0] == "Document 1 content"
        assert call_args[1] == "Document 2 content"
        assert call_args[2] == "Document 3 content"

    @patch("vectordb.langchain.utils.embeddings.HuggingFaceEmbeddings")
    def test_embed_single_document(self, mock_hf_embeddings):
        """Test embedding single document."""
        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_hf_embeddings.return_value = mock_embedder

        config = {}
        embedder = EmbedderHelper.create_embedder(config)

        doc = [Document(page_content="Single doc", metadata={})]
        result = EmbedderHelper.embed_documents(embedder, doc)

        docs, embeddings = result
        assert len(docs) == 1
        assert len(embeddings) == 1

    @patch("vectordb.langchain.utils.embeddings.HuggingFaceEmbeddings")
    def test_embed_empty_list(self, mock_hf_embeddings):
        """Test embedding empty document list."""
        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = []
        mock_hf_embeddings.return_value = mock_embedder

        config = {}
        embedder = EmbedderHelper.create_embedder(config)

        result = EmbedderHelper.embed_documents(embedder, [])

        docs, embeddings = result
        assert docs == []
        assert embeddings == []

    @patch("vectordb.langchain.utils.embeddings.HuggingFaceEmbeddings")
    def test_embed_documents_preserves_metadata(
        self, mock_hf_embeddings, sample_documents
    ):
        """Test that embed_documents preserves document metadata."""
        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = [[0.1, 0.2, 0.3]] * 3
        mock_hf_embeddings.return_value = mock_embedder

        config = {}
        embedder = EmbedderHelper.create_embedder(config)

        result = EmbedderHelper.embed_documents(embedder, sample_documents)

        docs, _ = result
        assert docs[0].metadata == {"source": "doc1"}
        assert docs[1].metadata == {"source": "doc2"}
        assert docs[2].metadata == {"source": "doc3"}


class TestEmbedQuery:
    """Tests for EmbedderHelper.embed_query single query embedding.

    Validates embedding of a single query string. Query embedding uses the
    same model as document embedding but may use different encoding settings
    optimized for shorter text.

    Return Value:
        List of floats representing the query embedding vector.
    """

    @patch("vectordb.langchain.utils.embeddings.HuggingFaceEmbeddings")
    def test_embed_query(self, mock_hf_embeddings):
        """Test embedding single query."""
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        mock_hf_embeddings.return_value = mock_embedder

        config = {}
        embedder = EmbedderHelper.create_embedder(config)

        result = EmbedderHelper.embed_query(embedder, "test query")

        assert len(result) == 4
        mock_embedder.embed_query.assert_called_once_with("test query")

    @patch("vectordb.langchain.utils.embeddings.HuggingFaceEmbeddings")
    def test_embed_empty_query(self, mock_hf_embeddings):
        """Test embedding empty query string."""
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = []
        mock_hf_embeddings.return_value = mock_embedder

        config = {}
        embedder = EmbedderHelper.create_embedder(config)

        result = EmbedderHelper.embed_query(embedder, "")

        assert result == []

    @patch("vectordb.langchain.utils.embeddings.HuggingFaceEmbeddings")
    def test_embed_long_query(self, mock_hf_embeddings):
        """Test embedding long query string."""
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 384
        mock_hf_embeddings.return_value = mock_embedder

        config = {}
        embedder = EmbedderHelper.create_embedder(config)

        long_query = "word " * 1000
        result = EmbedderHelper.embed_query(embedder, long_query)

        assert len(result) == 384
        mock_embedder.embed_query.assert_called_once()

    @patch("vectordb.langchain.utils.embeddings.HuggingFaceEmbeddings")
    def test_embed_query_with_special_characters(self, mock_hf_embeddings):
        """Test embedding query with special characters."""
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2]
        mock_hf_embeddings.return_value = mock_embedder

        config = {}
        embedder = EmbedderHelper.create_embedder(config)

        EmbedderHelper.embed_query(embedder, "Test @#$%^&*() Query!")

        mock_embedder.embed_query.assert_called_once_with("Test @#$%^&*() Query!")

    @patch("vectordb.langchain.utils.embeddings.HuggingFaceEmbeddings")
    def test_embed_query_with_unicode(self, mock_hf_embeddings):
        """Test embedding query with unicode characters."""
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2]
        mock_hf_embeddings.return_value = mock_embedder

        config = {}
        embedder = EmbedderHelper.create_embedder(config)

        EmbedderHelper.embed_query(embedder, "你好世界 🌍 测试")

        mock_embedder.embed_query.assert_called_once()


class TestEmbedderHelperEdgeCases:
    """Edge case tests for EmbedderHelper boundary conditions.

    Validates handling of GPU devices, Apple Silicon MPS, very long documents,
    unicode content, and document ordering preservation.

    Edge Cases Covered:
        - CUDA and MPS device configuration
        - Batch size of 1 (sequential processing)
        - Very long document content (10k+ words)
        - Unicode and special characters
        - Document order preservation through embedding
    """

    @patch("vectordb.langchain.utils.embeddings.HuggingFaceEmbeddings")
    def test_create_embedder_gpu_device(self, mock_hf_embeddings):
        """Test creating embedder with GPU device."""
        mock_embedder = MagicMock()
        mock_hf_embeddings.return_value = mock_embedder

        config = {
            "embeddings": {
                "device": "cuda",
            }
        }
        EmbedderHelper.create_embedder(config)

        call_kwargs = mock_hf_embeddings.call_args[1]
        assert call_kwargs["model_kwargs"]["device"] == "cuda"

    @patch("vectordb.langchain.utils.embeddings.HuggingFaceEmbeddings")
    def test_create_embedder_mps_device(self, mock_hf_embeddings):
        """Test creating embedder with MPS device (Apple Silicon)."""
        mock_embedder = MagicMock()
        mock_hf_embeddings.return_value = mock_embedder

        config = {
            "embeddings": {
                "device": "mps",
            }
        }
        EmbedderHelper.create_embedder(config)

        call_kwargs = mock_hf_embeddings.call_args[1]
        assert call_kwargs["model_kwargs"]["device"] == "mps"

    @patch("vectordb.langchain.utils.embeddings.HuggingFaceEmbeddings")
    def test_create_embedder_batch_size_1(self, mock_hf_embeddings):
        """Test creating embedder with batch size of 1."""
        mock_embedder = MagicMock()
        mock_hf_embeddings.return_value = mock_embedder

        config = {
            "embeddings": {
                "batch_size": 1,
            }
        }
        EmbedderHelper.create_embedder(config)

        call_kwargs = mock_hf_embeddings.call_args[1]
        assert call_kwargs["encode_kwargs"]["batch_size"] == 1

    @patch("vectordb.langchain.utils.embeddings.HuggingFaceEmbeddings")
    def test_embed_documents_with_very_long_texts(self, mock_hf_embeddings):
        """Test embedding documents with very long text content."""
        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = [[0.1] * 384]
        mock_hf_embeddings.return_value = mock_embedder

        config = {}
        embedder = EmbedderHelper.create_embedder(config)

        long_text = "word " * 10000
        docs = [Document(page_content=long_text, metadata={})]

        result = EmbedderHelper.embed_documents(embedder, docs)

        assert len(result[0]) == 1
        mock_embedder.embed_documents.assert_called_once()

    @patch("vectordb.langchain.utils.embeddings.HuggingFaceEmbeddings")
    def test_embed_documents_preserves_document_order(self, mock_hf_embeddings):
        """Test that embed_documents preserves document order."""
        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = [
            [0.1],
            [0.2],
            [0.3],
        ]
        mock_hf_embeddings.return_value = mock_embedder

        config = {}
        embedder = EmbedderHelper.create_embedder(config)

        docs = [
            Document(page_content="First", metadata={"order": 1}),
            Document(page_content="Second", metadata={"order": 2}),
            Document(page_content="Third", metadata={"order": 3}),
        ]

        result = EmbedderHelper.embed_documents(embedder, docs)

        returned_docs, _ = result
        assert returned_docs[0].metadata["order"] == 1
        assert returned_docs[1].metadata["order"] == 2
        assert returned_docs[2].metadata["order"] == 3

    @patch("vectordb.langchain.utils.embeddings.HuggingFaceEmbeddings")
    def test_query_embedding_dimension_matches_model(self, mock_hf_embeddings):
        """Test that query embedding dimension is consistent."""
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 768
        mock_hf_embeddings.return_value = mock_embedder

        config = {
            "embeddings": {
                "model": "sentence-transformers/all-mpnet-base-v2",
            }
        }
        embedder = EmbedderHelper.create_embedder(config)

        result = EmbedderHelper.embed_query(embedder, "test query")

        assert len(result) == 768
