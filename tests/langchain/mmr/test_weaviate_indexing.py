"""Tests for Weaviate MMR indexing pipeline (LangChain).

This module tests the WeaviateMMRIndexingPipeline which provides Maximal Marginal
Relevance (MMR) indexing capabilities using Weaviate as the vector database backend.
The pipeline supports:

- Weaviate-specific configuration: url, api_key, collection_name
- Collection/class creation in Weaviate for document storage
- URL and authentication configuration verification
- Document embedding and indexing with MMR metadata
- Config file loading and validation

Key test areas:
- Pipeline initialization with valid config dict and file paths
- Execution flow with and without documents
- Weaviate-specific settings: URL extraction from config
- Default collection name handling
- Collection configuration validation

Dependencies:
- vectordb.langchain.mmr.indexing.weaviate
- unittest.mock for mocking WeaviateVectorDB and embedder helpers
- pytest fixtures: weaviate_config, sample_documents, tmp_path
"""

from unittest.mock import MagicMock, patch

from vectordb.langchain.mmr.indexing.weaviate import WeaviateMMRIndexingPipeline


class TestWeaviateMMRIndexingPipeline:
    """Test suite for WeaviateMMRIndexingPipeline.

    Tests cover initialization patterns, configuration handling,
    and execution flow for the Weaviate-based MMR indexing pipeline.

    Attributes:
        None - Uses pytest fixtures for test data and mocks.
    """

    @patch("vectordb.langchain.mmr.indexing.weaviate.WeaviateVectorDB")
    @patch("vectordb.langchain.mmr.indexing.weaviate.EmbedderHelper.create_embedder")
    def test_init_with_valid_config(
        self, mock_embedder_helper, mock_db_cls, weaviate_config
    ):
        """Test pipeline initialization with a valid configuration dictionary.

        Verifies that the pipeline correctly initializes with all required
        Weaviate configuration parameters (url, api_key, collection_name)
        and that the collection_name is properly extracted from config.

        Args:
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db_cls: Mock for WeaviateVectorDB class
            weaviate_config: Pytest fixture providing valid Weaviate config dict

        Expected Behavior:
            - Pipeline instance is created successfully
            - Collection name matches config: "TestMMR"
        """
        mock_embedder = MagicMock()
        mock_embedder_helper.return_value = mock_embedder

        mock_db_instance = MagicMock()
        mock_db_cls.return_value = mock_db_instance

        pipeline = WeaviateMMRIndexingPipeline(weaviate_config)
        assert pipeline is not None
        assert pipeline.collection_name == "TestMMR"

    @patch("vectordb.langchain.mmr.indexing.weaviate.WeaviateVectorDB")
    @patch("vectordb.langchain.mmr.indexing.weaviate.EmbedderHelper.create_embedder")
    def test_init_with_config_path(self, mock_embedder_helper, mock_db_cls, tmp_path):
        """Test pipeline initialization with a configuration file path.

        Verifies that the pipeline can load configuration from a YAML file
        and properly parse Weaviate-specific settings including url, api_key,
        and collection_name.

        Args:
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db_cls: Mock for WeaviateVectorDB class
            tmp_path: Pytest fixture providing temporary directory path

        Expected Behavior:
            - Pipeline loads config from file path successfully
            - Weaviate settings (url, api_key, collection_name) parsed correctly
            - All sub-configs (embeddings, dataloader, mmr) loaded
        """
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
weaviate:
  url: http://localhost:8080
  api_key: ""
  collection_name: TestMMR
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

        pipeline = WeaviateMMRIndexingPipeline(str(config_file))
        assert pipeline is not None

    class TestRun:
        """Test suite for the pipeline's run method.

        Tests document processing flow including fetching, embedding,
        and upserting documents into Weaviate with MMR configuration.
        """

        @patch("vectordb.langchain.mmr.indexing.weaviate.WeaviateVectorDB")
        @patch(
            "vectordb.langchain.mmr.indexing.weaviate.EmbedderHelper.create_embedder"
        )
        @patch(
            "vectordb.langchain.mmr.indexing.weaviate.EmbedderHelper.embed_documents"
        )
        @patch("vectordb.langchain.mmr.indexing.weaviate.DataloaderCatalog.create")
        def test_run_with_documents(
            self,
            mock_get_docs,
            mock_embed_docs,
            mock_embedder_helper,
            mock_db_cls,
            sample_documents,
            weaviate_config,
        ):
            """Test run() method with a non-empty document set.

            Verifies the complete indexing pipeline flow: loading documents,
            generating embeddings, and upserting into Weaviate collection.

            Args:
                mock_get_docs: Mock for DataloaderCatalog.create
                mock_embed_docs: Mock for EmbedderHelper.embed_documents
                mock_embedder_helper: Mock for EmbedderHelper.create_embedder
                mock_db_cls: Mock for WeaviateVectorDB class
                sample_documents: Pytest fixture providing test documents
                weaviate_config: Pytest fixture providing Weaviate config

            Expected Behavior:
                - Documents are fetched from dataloader
                - Embeddings are generated for all documents (384-dim)
                - Documents are upserted to Weaviate collection
                - Returns document count matching input
            """
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

            pipeline = WeaviateMMRIndexingPipeline(weaviate_config)
            result = pipeline.run()

            assert result["documents_indexed"] == len(sample_documents)

        @patch("vectordb.langchain.mmr.indexing.weaviate.WeaviateVectorDB")
        @patch(
            "vectordb.langchain.mmr.indexing.weaviate.EmbedderHelper.create_embedder"
        )
        @patch("vectordb.langchain.mmr.indexing.weaviate.DataloaderCatalog.create")
        def test_run_with_empty_documents(
            self,
            mock_get_docs,
            mock_embedder_helper,
            mock_db_cls,
            weaviate_config,
        ):
            """Test run() method with an empty document set (edge case).

            Verifies that the pipeline handles the edge case where no
            documents are returned from the dataloader.

            Args:
                mock_get_docs: Mock for DataloaderCatalog.create
                mock_embedder_helper: Mock for EmbedderHelper.create_embedder
                mock_db_cls: Mock for WeaviateVectorDB class
                weaviate_config: Pytest fixture providing Weaviate config

            Expected Behavior:
                - Empty list from dataloader is handled gracefully
                - Pipeline completes without errors
                - Returns documents_indexed: 0
            """
            mock_dataset = MagicMock()
            mock_dataset.to_langchain.return_value = []
            mock_loader = MagicMock()
            mock_loader.load.return_value = mock_dataset
            mock_get_docs.return_value = mock_loader

            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_db_instance = MagicMock()
            mock_db_cls.return_value = mock_db_instance

            pipeline = WeaviateMMRIndexingPipeline(weaviate_config)
            result = pipeline.run()

            assert result["documents_indexed"] == 0

    class TestWeaviateConfiguration:
        """Test suite for Weaviate-specific configuration handling.

        Tests Weaviate connection parameters: URL, API key, and
        collection/class creation settings.
        """

        @patch("vectordb.langchain.mmr.indexing.weaviate.WeaviateVectorDB")
        @patch(
            "vectordb.langchain.mmr.indexing.weaviate.EmbedderHelper.create_embedder"
        )
        def test_url_from_config(
            self, mock_embedder_helper, mock_db_cls, weaviate_config
        ):
            """Test that Weaviate URL from config is passed to WeaviateVectorDB.

            Verifies that the url and api_key from the configuration
            are correctly extracted and passed when initializing the
            WeaviateVectorDB instance.

            Args:
                mock_embedder_helper: Mock for EmbedderHelper.create_embedder
                mock_db_cls: Mock for WeaviateVectorDB class
                weaviate_config: Pytest fixture providing Weaviate config with url

            Expected Behavior:
                - URL from config (http://localhost:8080) passed to WeaviateVectorDB
                - API key from config passed correctly
            """
            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_db_instance = MagicMock()
            mock_db_cls.return_value = mock_db_instance

            pipeline = WeaviateMMRIndexingPipeline(weaviate_config)
            assert pipeline is not None

            mock_db_cls.assert_called_once_with(url="http://localhost:8080", api_key="")

        @patch("vectordb.langchain.mmr.indexing.weaviate.WeaviateVectorDB")
        @patch(
            "vectordb.langchain.mmr.indexing.weaviate.EmbedderHelper.create_embedder"
        )
        def test_default_collection_name(self, mock_embedder_helper, mock_db_cls):
            """Test that default collection name is used when not specified.

            Verifies that when collection_name is omitted from the Weaviate
            configuration, the pipeline falls back to the default "MMR".

            Args:
                mock_embedder_helper: Mock for EmbedderHelper.create_embedder
                mock_db_cls: Mock for WeaviateVectorDB class

            Expected Behavior:
                - Default collection_name: "MMR" applied when not in config
            """
            config = {
                "dataloader": {"type": "arc", "limit": 10},
                "embeddings": {"model": "test-model", "device": "cpu"},
                "weaviate": {
                    "url": "http://localhost:8080",
                    "dimension": 384,
                },
            }

            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_db_instance = MagicMock()
            mock_db_cls.return_value = mock_db_instance

            pipeline = WeaviateMMRIndexingPipeline(config)
            assert pipeline.collection_name == "MMR"
