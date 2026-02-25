"""Tests for Qdrant Maximal Marginal Relevance (MMR) indexing pipeline (LangChain).

This module tests the QdrantMMRIndexingPipeline class, which handles:
- Qdrant vector database integration with MMR-based retrieval
- Collection creation and management with configurable parameters
- Document indexing with embedding generation
- URL, API key, and collection configuration handling
- Config file loading from YAML
"""

from unittest.mock import MagicMock, patch

from vectordb.langchain.mmr.indexing.qdrant import QdrantMMRIndexingPipeline


class TestQdrantMMRIndexingPipeline:
    """Tests for QdrantMMRIndexingPipeline initialization and configuration handling.

    Tests cover:
    - Initialization with config dictionary
    - Initialization from config file path
    - Qdrant-specific parameters (url, api_key, collection_name, dimension)
    """

    @patch("vectordb.langchain.mmr.indexing.qdrant.QdrantVectorDB")
    @patch("vectordb.langchain.mmr.indexing.qdrant.EmbedderHelper.create_embedder")
    def test_init_with_valid_config(
        self, mock_embedder_helper, mock_db_cls, qdrant_config
    ):
        """Test initialization with valid config dict."""
        mock_embedder = MagicMock()
        mock_embedder_helper.return_value = mock_embedder

        mock_db_instance = MagicMock()
        mock_db_cls.return_value = mock_db_instance

        pipeline = QdrantMMRIndexingPipeline(qdrant_config)
        assert pipeline is not None
        assert pipeline.collection_name == "test_mmr"

    @patch("vectordb.langchain.mmr.indexing.qdrant.QdrantVectorDB")
    @patch("vectordb.langchain.mmr.indexing.qdrant.EmbedderHelper.create_embedder")
    def test_init_with_config_path(self, mock_embedder_helper, mock_db_cls, tmp_path):
        """Test initialization with config file path."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
qdrant:
  url: http://localhost:6333
  api_key: ""
  collection_name: test_mmr
  dimension: 384
embeddings:
  model: sentence-transformers/all-MiniLM-L6-v2
  device: cpu
dataloader:
  type: arc
  split: test
  limit: 10
mmr:
  threshold: 0.5
  k: 5
            """
        )

        mock_embedder = MagicMock()
        mock_embedder_helper.return_value = mock_embedder

        mock_db_instance = MagicMock()
        mock_db_cls.return_value = mock_db_instance

        pipeline = QdrantMMRIndexingPipeline(str(config_file))
        assert pipeline is not None

    class TestRun:
        """Tests for the run() method.

        Tests cover:
        - Document indexing workflow with embedding generation
        - Empty document handling
        - Database upsert operations
        """

        @patch("vectordb.langchain.mmr.indexing.qdrant.QdrantVectorDB")
        @patch("vectordb.langchain.mmr.indexing.qdrant.EmbedderHelper.create_embedder")
        @patch("vectordb.langchain.mmr.indexing.qdrant.EmbedderHelper.embed_documents")
        @patch("vectordb.langchain.mmr.indexing.qdrant.DataloaderCatalog.create")
        def test_run_with_documents(
            self,
            mock_get_docs,
            mock_embed_docs,
            mock_embedder_helper,
            mock_db_cls,
            sample_documents,
            qdrant_config,
        ):
            """Test run() method with documents."""
            mock_dataset = MagicMock()
            mock_dataset.to_langchain.return_value = sample_documents
            mock_loader = MagicMock()
            mock_loader.load.return_value = mock_dataset
            mock_get_docs.return_value = mock_loader
            mock_embed_docs.return_value = (
                sample_documents,
                [[0.1] * 384 for _ in range(len(sample_documents))],
            )

            mock_db_instance = MagicMock()
            mock_db_instance.upsert.return_value = len(sample_documents)
            mock_db_cls.return_value = mock_db_instance

            pipeline = QdrantMMRIndexingPipeline(qdrant_config)
            result = pipeline.run()

            assert result["documents_indexed"] == len(sample_documents)

        @patch("vectordb.langchain.mmr.indexing.qdrant.QdrantVectorDB")
        @patch("vectordb.langchain.mmr.indexing.qdrant.EmbedderHelper.create_embedder")
        @patch("vectordb.langchain.mmr.indexing.qdrant.DataloaderCatalog.create")
        def test_run_with_empty_documents(
            self,
            mock_get_docs,
            mock_embedder_helper,
            mock_db_cls,
            qdrant_config,
        ):
            """Test run() method with empty documents (edge case)."""
            mock_dataset = MagicMock()
            mock_dataset.to_langchain.return_value = []
            mock_loader = MagicMock()
            mock_loader.load.return_value = mock_dataset
            mock_get_docs.return_value = mock_loader

            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_db_instance = MagicMock()
            mock_db_cls.return_value = mock_db_instance

            pipeline = QdrantMMRIndexingPipeline(qdrant_config)
            result = pipeline.run()

            assert result["documents_indexed"] == 0

    class TestQdrantConfiguration:
        """Tests for Qdrant-specific configuration handling.

        Tests cover:
        - URL and API key parameter passing to QdrantVectorDB
        - Collection name configuration
        - Dimension validation
        - Default values for optional parameters
        """

        @patch("vectordb.langchain.mmr.indexing.qdrant.QdrantVectorDB")
        @patch("vectordb.langchain.mmr.indexing.qdrant.EmbedderHelper.create_embedder")
        def test_url_from_config(
            self, mock_embedder_helper, mock_db_cls, qdrant_config
        ):
            """Test that URL is passed to QdrantVectorDB."""
            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_db_instance = MagicMock()
            mock_db_cls.return_value = mock_db_instance

            pipeline = QdrantMMRIndexingPipeline(qdrant_config)
            assert pipeline is not None

            mock_db_cls.assert_called_once_with(url="http://localhost:6333", api_key="")

        @patch("vectordb.langchain.mmr.indexing.qdrant.QdrantVectorDB")
        @patch("vectordb.langchain.mmr.indexing.qdrant.EmbedderHelper.create_embedder")
        def test_default_collection_name(self, mock_embedder_helper, mock_db_cls):
            """Test default collection name."""
            config = {
                "dataloader": {"type": "arc", "limit": 10},
                "embeddings": {"model": "test-model", "device": "cpu"},
                "qdrant": {
                    "url": "http://localhost:6333",
                    "dimension": 384,
                },
            }

            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_db_instance = MagicMock()
            mock_db_cls.return_value = mock_db_instance

            pipeline = QdrantMMRIndexingPipeline(config)
            assert pipeline.collection_name == "mmr"
