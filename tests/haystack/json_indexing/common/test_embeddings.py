"""Tests for JSON indexing embedding utilities.

Tests cover:
- Document embedder creation with various configurations
- Text embedder creation with various configurations
- Embedding dimension detection
- Document embedding generation
"""

from unittest.mock import MagicMock, patch

from haystack import Document

from vectordb.haystack.json_indexing.common.embeddings import (
    create_document_embedder,
    create_text_embedder,
    embed_documents,
    get_embedding_dimension,
)


class TestCreateDocumentEmbedder:
    """Tests for create_document_embedder function."""

    @patch(
        "vectordb.haystack.json_indexing.common.embeddings.SentenceTransformersDocumentEmbedder"
    )
    def test_create_embedder_default_model(
        self, mock_embedder_class: MagicMock
    ) -> None:
        """Test creating embedder with default model."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        config = {}
        result = create_document_embedder(config)

        mock_embedder_class.assert_called_once_with(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        mock_instance.warm_up.assert_called_once()
        assert result == mock_instance

    @patch(
        "vectordb.haystack.json_indexing.common.embeddings.SentenceTransformersDocumentEmbedder"
    )
    def test_create_embedder_custom_model(self, mock_embedder_class: MagicMock) -> None:
        """Test creating embedder with custom model."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        config = {"embeddings": {"model": "custom-model"}}
        result = create_document_embedder(config)

        mock_embedder_class.assert_called_once_with(model="custom-model")
        mock_instance.warm_up.assert_called_once()
        assert result == mock_instance

    @patch(
        "vectordb.haystack.json_indexing.common.embeddings.SentenceTransformersDocumentEmbedder"
    )
    def test_create_embedder_qwen3_alias(self, mock_embedder_class: MagicMock) -> None:
        """Test creating embedder with qwen3 alias."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        config = {"embeddings": {"model": "qwen3"}}
        result = create_document_embedder(config)

        mock_embedder_class.assert_called_once_with(model="Qwen/Qwen3-Embedding-0.6B")
        mock_instance.warm_up.assert_called_once()
        assert result == mock_instance

    @patch(
        "vectordb.haystack.json_indexing.common.embeddings.SentenceTransformersDocumentEmbedder"
    )
    def test_create_embedder_minilm_alias(self, mock_embedder_class: MagicMock) -> None:
        """Test creating embedder with minilm alias."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        config = {"embeddings": {"model": "minilm"}}
        result = create_document_embedder(config)

        mock_embedder_class.assert_called_once_with(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        mock_instance.warm_up.assert_called_once()
        assert result == mock_instance

    @patch(
        "vectordb.haystack.json_indexing.common.embeddings.SentenceTransformersDocumentEmbedder"
    )
    def test_create_embedder_case_insensitive_alias(
        self, mock_embedder_class: MagicMock
    ) -> None:
        """Test that model aliases are case insensitive."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        config = {"embeddings": {"model": "Qwen3"}}
        create_document_embedder(config)

        mock_embedder_class.assert_called_once_with(model="Qwen/Qwen3-Embedding-0.6B")


class TestCreateTextEmbedder:
    """Tests for create_text_embedder function."""

    @patch(
        "vectordb.haystack.json_indexing.common.embeddings.SentenceTransformersTextEmbedder"
    )
    def test_create_text_embedder_default_model(
        self, mock_embedder_class: MagicMock
    ) -> None:
        """Test creating text embedder with default model."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        config = {}
        result = create_text_embedder(config)

        mock_embedder_class.assert_called_once_with(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        mock_instance.warm_up.assert_called_once()
        assert result == mock_instance

    @patch(
        "vectordb.haystack.json_indexing.common.embeddings.SentenceTransformersTextEmbedder"
    )
    def test_create_text_embedder_custom_model(
        self, mock_embedder_class: MagicMock
    ) -> None:
        """Test creating text embedder with custom model."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        config = {"embeddings": {"model": "custom-model"}}
        result = create_text_embedder(config)

        mock_embedder_class.assert_called_once_with(model="custom-model")
        mock_instance.warm_up.assert_called_once()
        assert result == mock_instance

    @patch(
        "vectordb.haystack.json_indexing.common.embeddings.SentenceTransformersTextEmbedder"
    )
    def test_create_text_embedder_qwen3_alias(
        self, mock_embedder_class: MagicMock
    ) -> None:
        """Test creating text embedder with qwen3 alias."""
        mock_instance = MagicMock()
        mock_embedder_class.return_value = mock_instance

        config = {"embeddings": {"model": "qwen3"}}
        result = create_text_embedder(config)

        mock_embedder_class.assert_called_once_with(model="Qwen/Qwen3-Embedding-0.6B")
        mock_instance.warm_up.assert_called_once()
        assert result == mock_instance


class TestGetEmbeddingDimension:
    """Tests for get_embedding_dimension function."""

    def test_get_dimension_from_document_embedder(self) -> None:
        """Test getting dimension from document embedder."""
        from haystack.components.embedders import SentenceTransformersDocumentEmbedder

        mock_embedder = MagicMock(spec=SentenceTransformersDocumentEmbedder)
        embedded_doc = Document(content="test", embedding=[0.1] * 384)
        mock_embedder.run.return_value = {"documents": [embedded_doc]}

        dimension = get_embedding_dimension(mock_embedder)

        assert dimension == 384
        mock_embedder.run.assert_called_once()

    def test_get_dimension_from_text_embedder(self) -> None:
        """Test getting dimension from text embedder."""
        from haystack.components.embedders import SentenceTransformersTextEmbedder

        mock_embedder = MagicMock(spec=SentenceTransformersTextEmbedder)
        mock_embedder.run.return_value = {"embedding": [0.1] * 768}

        dimension = get_embedding_dimension(mock_embedder)

        assert dimension == 768
        mock_embedder.run.assert_called_once_with(
            text="sample text for dimension check"
        )

    def test_get_dimension_different_sizes(self) -> None:
        """Test getting dimensions of various sizes."""
        from haystack.components.embedders import SentenceTransformersDocumentEmbedder

        mock_embedder = MagicMock(spec=SentenceTransformersDocumentEmbedder)

        # Test with 128 dimensions
        embedded_doc = Document(content="test", embedding=[0.1] * 128)
        mock_embedder.run.return_value = {"documents": [embedded_doc]}

        dimension = get_embedding_dimension(mock_embedder)
        assert dimension == 128


class TestEmbedDocuments:
    """Tests for embed_documents function."""

    def test_embed_documents(self) -> None:
        """Test embedding documents."""
        mock_embedder = MagicMock()
        docs = [
            Document(content="doc1"),
            Document(content="doc2"),
        ]
        embedded_docs = [
            Document(content="doc1", embedding=[0.1] * 384),
            Document(content="doc2", embedding=[0.2] * 384),
        ]
        mock_embedder.run.return_value = {"documents": embedded_docs}

        result = embed_documents(docs, mock_embedder)

        mock_embedder.run.assert_called_once_with(documents=docs)
        assert result == embedded_docs
        assert len(result) == 2

    def test_embed_empty_documents(self) -> None:
        """Test embedding empty document list."""
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"documents": []}

        result = embed_documents([], mock_embedder)

        mock_embedder.run.assert_called_once_with(documents=[])
        assert result == []

    def test_embed_single_document(self) -> None:
        """Test embedding single document."""
        mock_embedder = MagicMock()
        docs = [Document(content="single doc")]
        embedded_docs = [Document(content="single doc", embedding=[0.5] * 256)]
        mock_embedder.run.return_value = {"documents": embedded_docs}

        result = embed_documents(docs, mock_embedder)

        assert len(result) == 1
        assert result[0].embedding == [0.5] * 256
