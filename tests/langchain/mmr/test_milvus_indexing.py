"""Tests for Milvus MMR indexing pipeline (LangChain).

This module tests the MilvusMMRIndexingPipeline which prepares Milvus collections
for Maximal Marginal Relevance (MMR) retrieval operations. The pipeline handles
document loading, embedding generation, and indexing to Milvus with proper
collection configuration.

Milvus-Specific Configuration:
    - host: Milvus server hostname (default: "localhost")
    - port: Milvus server port (default: 19530)
    - collection_name: Collection identifier for MMR documents
    - dimension: Embedding dimension (must match embedder output)

Test Coverage:
    - Pipeline initialization with config dict and file paths
    - Document loading and embedding generation
    - Collection creation and document upsert operations
    - Empty document handling (edge case)
    - MMR configuration parameter validation
    - Default collection name behavior

The tests mock MilvusVectorDB and embedder components to test pipeline logic
without requiring a live Milvus instance. Uses pytest fixtures from conftest.py
for sample documents and configurations.
"""

from unittest.mock import MagicMock, patch

from vectordb.langchain.mmr.indexing.milvus import MilvusMMRIndexingPipeline


class TestMilvusMMRIndexingPipeline:
    """Tests for MilvusMMRIndexingPipeline initialization and configuration.

    Tests pipeline instantiation with various configuration sources (dict, file)
    and validates Milvus-specific settings are properly extracted.

    Attributes:
        Milvus configuration keys tested: host, port, collection_name, dimension
        Default behavior: collection_name defaults to "mmr" if not specified
    """

    @patch("vectordb.langchain.mmr.indexing.milvus.MilvusVectorDB")
    @patch("vectordb.langchain.mmr.indexing.milvus.EmbedderHelper.create_embedder")
    def test_init_with_valid_config(
        self, mock_embedder_helper, mock_db_cls, milvus_config
    ):
        """Test initialization with valid configuration dictionary.

        Validates that MilvusMMRIndexingPipeline correctly initializes when
        provided with a valid config dict containing Milvus connection params
        (host, port, collection_name) and MMR settings.

        Args:
            mock_embedder_helper: Mocked EmbedderHelper.create_embedder patch.
            mock_db_cls: Mocked MilvusVectorDB class patch.
            milvus_config: Fixture providing valid Milvus configuration dict with
                milvus.host, milvus.port, milvus.collection_name, milvus.dimension,
                and mmr.threshold settings.

        Asserts:
            Pipeline instance is created successfully.
            Collection name is extracted from config ("test_mmr").
        """
        mock_embedder = MagicMock()
        mock_embedder_helper.return_value = mock_embedder

        mock_db_instance = MagicMock()
        mock_db_cls.return_value = mock_db_instance

        pipeline = MilvusMMRIndexingPipeline(milvus_config)
        assert pipeline is not None
        assert pipeline.collection_name == "test_mmr"

    @patch("vectordb.langchain.mmr.indexing.milvus.MilvusVectorDB")
    @patch("vectordb.langchain.mmr.indexing.milvus.EmbedderHelper.create_embedder")
    def test_init_with_config_path(self, mock_embedder_helper, mock_db_cls, tmp_path):
        """Test initialization with YAML configuration file path.

        Validates that the pipeline can load configuration from a YAML file
        and properly parse Milvus settings (host, port, collection_name,
        dimension) along with embeddings and MMR parameters.

        Args:
            mock_embedder_helper: Mocked EmbedderHelper.create_embedder patch.
            mock_db_cls: Mocked MilvusVectorDB class patch.
            tmp_path: pytest fixture providing temporary directory path.

        Configuration tested in YAML:
            milvus: host, port, collection_name, dimension
            embeddings: model, device
            dataloader: type, split, limit
            mmr: threshold, k

        Asserts:
            Pipeline instance is created from file-based config.
        """
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
milvus:
  host: localhost
  port: 19530
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

        pipeline = MilvusMMRIndexingPipeline(str(config_file))
        assert pipeline is not None

    class TestRun:
        """Tests for pipeline run() method execution.

        Tests document indexing workflow: loading documents, generating embeddings,
        and upserting to Milvus collection. Covers normal operation and edge cases.

        Key operations tested:
            - Document loading via DataloaderCatalog.create
            - Embedding generation via EmbedderHelper.embed_documents
            - Collection upsert via MilvusVectorDB.upsert
        """

        @patch("vectordb.langchain.mmr.indexing.milvus.MilvusVectorDB")
        @patch("vectordb.langchain.mmr.indexing.milvus.EmbedderHelper.create_embedder")
        @patch("vectordb.langchain.mmr.indexing.milvus.EmbedderHelper.embed_documents")
        @patch("vectordb.langchain.mmr.indexing.milvus.DataloaderCatalog.create")
        def test_run_with_documents(
            self,
            mock_get_docs,
            mock_embed_docs,
            mock_embedder_helper,
            mock_db_cls,
            sample_documents,
            milvus_config,
        ):
            """Test run() method with valid documents for indexing.

            Validates complete indexing workflow: documents are loaded,
            embeddings generated (384-dim vectors), and upserted to Milvus.
            Verifies correct count of indexed documents is returned.

            Args:
                mock_get_docs: Mock for DataloaderCatalog.create.
                mock_embed_docs: Mock for EmbedderHelper.embed_documents.
                mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
                mock_db_cls: Mock for MilvusVectorDB class.
                sample_documents: Fixture providing 5 sample Document objects.
                milvus_config: Fixture with Milvus connection and MMR settings.

            Mocks behavior:
                - Returns sample_documents from loader
                - Returns 384-dim embeddings matching milvus_config dimension
                - Returns document count from db.upsert()

            Asserts:
                Result dict contains documents_indexed matching sample count.
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

            pipeline = MilvusMMRIndexingPipeline(milvus_config)
            result = pipeline.run()

            assert result["documents_indexed"] == len(sample_documents)

        @patch("vectordb.langchain.mmr.indexing.milvus.MilvusVectorDB")
        @patch("vectordb.langchain.mmr.indexing.milvus.EmbedderHelper.create_embedder")
        @patch("vectordb.langchain.mmr.indexing.milvus.DataloaderCatalog.create")
        def test_run_with_empty_documents(
            self,
            mock_get_docs,
            mock_embedder_helper,
            mock_db_cls,
            milvus_config,
        ):
            """Test run() method with empty document list (edge case).

            Validates pipeline handles empty document list gracefully without
            errors. Should return zero indexed documents without attempting
            embedding generation or Milvus upsert.

            Args:
                mock_get_docs: Mock for DataloaderCatalog.create.
                mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
                mock_db_cls: Mock for MilvusVectorDB class.
                milvus_config: Fixture with Milvus configuration settings.

            Mocks behavior:
                - Returns empty list from document loader

            Asserts:
                Result contains documents_indexed: 0.
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

            pipeline = MilvusMMRIndexingPipeline(milvus_config)
            result = pipeline.run()

            assert result["documents_indexed"] == 0

    class TestMMRConfiguration:
        """Tests for MMR (Maximal Marginal Relevance) configuration handling.

        Validates MMR-specific parameters in configuration are accessible
        and correctly stored. MMR parameters (threshold, k) are used during
        search phase but stored at indexing time for pipeline consistency.

        MMR Parameters tested:
            - threshold: Lambda value balancing relevance vs diversity (0.0-1.0)
            - k: Number of results to return in MMR reranking
        """

        @patch("vectordb.langchain.mmr.indexing.milvus.MilvusVectorDB")
        @patch("vectordb.langchain.mmr.indexing.milvus.EmbedderHelper.create_embedder")
        def test_mmr_threshold_from_config(
            self, mock_embedder_helper, mock_db_cls, milvus_config
        ):
            """Test MMR threshold parameter is correctly loaded from config.

            Validates that mmr.threshold from configuration is preserved
            and accessible via pipeline.config. Default threshold 0.5
            balances relevance (0.5) with diversity (0.5).

            Args:
                mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
                mock_db_cls: Mock for MilvusVectorDB class.
                milvus_config: Fixture with mmr.threshold: 0.5 setting.

            Asserts:
                Config contains mmr.threshold value of 0.5.
            """
            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_db_instance = MagicMock()
            mock_db_cls.return_value = mock_db_instance

            pipeline = MilvusMMRIndexingPipeline(milvus_config)
            assert pipeline.config.get("mmr", {}).get("threshold") == 0.5

        @patch("vectordb.langchain.mmr.indexing.milvus.MilvusVectorDB")
        @patch("vectordb.langchain.mmr.indexing.milvus.EmbedderHelper.create_embedder")
        def test_default_collection_name(self, mock_embedder_helper, mock_db_cls):
            """Test default collection name when not specified in config.

            Validates that when milvus.collection_name is omitted from
            configuration, the pipeline defaults to "mmr" as collection name.

            Args:
                mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
                mock_db_cls: Mock for MilvusVectorDB class.

            Configuration tested:
                Milvus config without collection_name key (only host, port, dimension).

            Asserts:
                pipeline.collection_name defaults to "mmr".
            """
            config = {
                "dataloader": {"type": "arc", "limit": 10},
                "embeddings": {"model": "test-model", "device": "cpu"},
                "milvus": {
                    "host": "localhost",
                    "port": 19530,
                    "dimension": 384,
                },
            }

            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_db_instance = MagicMock()
            mock_db_cls.return_value = mock_db_instance

            pipeline = MilvusMMRIndexingPipeline(config)
            assert pipeline.collection_name == "mmr"
