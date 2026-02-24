"""Tests for Chroma indexing pipeline in diversity filtering."""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.diversity_filtering.pipelines.chroma_indexing import (
    load_documents,
    run_indexing,
)


class TestLoadDocuments:
    """Tests for load_documents function."""

    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.chroma_indexing.DataloaderCatalog"
    )
    def test_load_documents_success(self, mock_catalog: MagicMock) -> None:
        """Test loading documents successfully."""
        mock_config = MagicMock()
        mock_config.dataset.name = "triviaqa"
        mock_config.dataset.split = "test"
        mock_config.dataset.max_documents = 10

        sample_documents = [
            Document(content="Document 1", meta={"source": "test1"}),
            Document(content="Document 2", meta={"source": "test2"}),
            Document(content="Document 3", meta={"source": "test3"}),
        ]

        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_catalog.create.return_value = mock_loader

        documents = load_documents(mock_config)

        assert len(documents) == 3
        assert isinstance(documents[0], Document)
        assert documents[0].content == "Document 1"
        assert documents[0].meta == {"source": "test1"}
        assert documents[2].content == "Document 3"

        mock_catalog.create.assert_called_once_with(
            "triviaqa",
            split="test",
            limit=10,
        )

    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.chroma_indexing.DataloaderCatalog"
    )
    def test_load_documents_empty(self, mock_catalog: MagicMock) -> None:
        """Test loading empty dataset."""
        mock_config = MagicMock()
        mock_config.dataset.name = "triviaqa"
        mock_config.dataset.split = "test"
        mock_config.dataset.max_documents = None

        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = []
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_catalog.create.return_value = mock_loader

        documents = load_documents(mock_config)

        assert documents == []

    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.chroma_indexing.DataloaderCatalog"
    )
    def test_load_documents_no_text_or_content(self, mock_catalog: MagicMock) -> None:
        """Test loading documents without text or content field."""
        mock_config = MagicMock()
        mock_config.dataset.name = "arc"
        mock_config.dataset.split = "validation"
        mock_config.dataset.max_documents = 5

        sample_documents = [Document(content="", meta={"source": "test"})]

        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_catalog.create.return_value = mock_loader

        documents = load_documents(mock_config)

        assert len(documents) == 1
        assert documents[0].content == ""  # Empty string fallback


class TestRunIndexing:
    """Tests for run_indexing function."""

    @pytest.fixture
    def mock_config(self) -> MagicMock:
        """Create a mock configuration."""
        config = MagicMock()
        config.dataset.name = "triviaqa"
        config.dataset.split = "test"
        config.dataset.max_documents = 100
        config.embedding.model = "sentence-transformers/all-MiniLM-L6-v2"
        config.embedding.dimension = 384
        config.embedding.batch_size = 32
        config.embedding.device = None
        config.index.name = "test_index"
        config.vectordb.chroma.host = "localhost"
        config.vectordb.chroma.port = 8000
        config.vectordb.chroma.is_persistent = False
        return config

    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.chroma_indexing.ChromaVectorDB"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.chroma_indexing.SentenceTransformersDocumentEmbedder"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.chroma_indexing.ConfigLoader"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.chroma_indexing.load_documents"
    )
    def test_run_indexing_success(
        self,
        mock_load_documents: MagicMock,
        mock_config_loader: MagicMock,
        mock_embedder_class: MagicMock,
        mock_db_class: MagicMock,
        mock_config: MagicMock,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """Test successful indexing pipeline run."""
        # Setup config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text("dummy: config")

        mock_config_loader.load.return_value = mock_config

        # Setup documents
        docs = [Document(content=f"Doc {i}") for i in range(5)]
        mock_load_documents.return_value = docs

        # Setup embedder - it receives docs and returns embedded docs
        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder
        embedded_docs = [
            Document(content=f"Doc {i}", embedding=[0.1] * 384) for i in range(5)
        ]
        mock_embedder.run.return_value = {"documents": embedded_docs}

        # Setup database
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db

        result = run_indexing(str(config_file))

        assert result["documents_indexed"] == 5
        assert result["index_name"] == "test_index"
        assert result["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert result["embedding_dimension"] == 384

        mock_embedder.warm_up.assert_called_once()
        # Verify embedder.run was called with the original docs (not embedded_docs)
        mock_embedder.run.assert_called_once()
        call_args = mock_embedder.run.call_args
        assert "documents" in call_args.kwargs
        assert len(call_args.kwargs["documents"]) == 5
        mock_db.index_documents.assert_called_once_with(embedded_docs)

    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.chroma_indexing.ConfigLoader"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.chroma_indexing.load_documents"
    )
    def test_run_indexing_no_documents(
        self,
        mock_load_documents: MagicMock,
        mock_config_loader: MagicMock,
        mock_config: MagicMock,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """Test indexing with no documents."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("dummy: config")

        mock_config_loader.load.return_value = mock_config
        mock_load_documents.return_value = []

        result = run_indexing(str(config_file))

        assert result["documents_indexed"] == 0
        assert result["error"] == "No documents loaded"

    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.chroma_indexing.ConfigLoader"
    )
    def test_run_indexing_config_not_found(
        self,
        mock_config_loader: MagicMock,
    ) -> None:
        """Test indexing with missing config file."""
        mock_config_loader.load.side_effect = FileNotFoundError("Config not found")

        with pytest.raises(FileNotFoundError, match="Config not found"):
            run_indexing("/nonexistent/config.yaml")

    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.chroma_indexing.ChromaVectorDB"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.chroma_indexing.SentenceTransformersDocumentEmbedder"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.chroma_indexing.ConfigLoader"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.chroma_indexing.load_documents"
    )
    def test_run_indexing_with_device(
        self,
        mock_load_documents: MagicMock,
        mock_config_loader: MagicMock,
        mock_embedder_class: MagicMock,
        mock_db_class: MagicMock,
        mock_config: MagicMock,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """Test indexing with specific device."""
        mock_config.embedding.device = "cuda"

        config_file = tmp_path / "config.yaml"
        config_file.write_text("dummy: config")

        mock_config_loader.load.return_value = mock_config

        docs = [Document(content="Test doc")]
        mock_load_documents.return_value = docs

        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder
        mock_embedder.run.return_value = {"documents": docs}

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db

        run_indexing(str(config_file))

        mock_embedder_class.assert_called_once_with(
            model="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=32,
            device="cuda",
        )

    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.chroma_indexing.ChromaVectorDB"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.chroma_indexing.SentenceTransformersDocumentEmbedder"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.chroma_indexing.ConfigLoader"
    )
    @patch(
        "vectordb.haystack.diversity_filtering.pipelines.chroma_indexing.load_documents"
    )
    def test_run_indexing_persistent_storage(
        self,
        mock_load_documents: MagicMock,
        mock_config_loader: MagicMock,
        mock_embedder_class: MagicMock,
        mock_db_class: MagicMock,
        mock_config: MagicMock,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """Test indexing with persistent storage enabled."""
        mock_config.vectordb.chroma.is_persistent = True

        config_file = tmp_path / "config.yaml"
        config_file.write_text("dummy: config")

        mock_config_loader.load.return_value = mock_config

        docs = [Document(content="Test doc")]
        mock_load_documents.return_value = docs

        mock_embedder = MagicMock()
        mock_embedder_class.return_value = mock_embedder
        mock_embedder.run.return_value = {"documents": docs}

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db

        run_indexing(str(config_file))

        mock_db_class.assert_called_once_with(
            host="localhost",
            port=8000,
            index="test_index",
            embedding_dim=384,
            is_persistent=True,
        )
