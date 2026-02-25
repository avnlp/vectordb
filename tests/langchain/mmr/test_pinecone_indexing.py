"""Tests for Pinecone MMR indexing pipeline (LangChain).

This module tests the PineconeMMRIndexingPipeline which handles document indexing
for Maximal Marginal Relevance (MMR) retrieval using Pinecone as the vector store.

The MMR indexing pipeline supports:
- Document embedding with configurable models
- Vector upsert to Pinecone indices with namespace isolation
- Index creation with dimension and metric configuration
- Multi-tenant data isolation via namespaces

Pinecone-specific Configuration:
    - api_key: Authentication credentials for Pinecone API
    - index_name: Target index for document storage
    - namespace: Logical partition for multi-tenant isolation
    - dimension: Vector dimension (must match embedding model)
    - metric: Similarity metric (cosine, euclidean, dotproduct)

Test Coverage:
    - TestPineconeMMRIndexingPipeline: Initialization and setup
    - TestRun: Document indexing and embedding workflow
    - TestPineconeConfiguration: Pinecone-specific settings and defaults
"""

from unittest.mock import MagicMock, patch

from vectordb.langchain.mmr.indexing.pinecone import PineconeMMRIndexingPipeline


class TestPineconeMMRIndexingPipeline:
    """Tests for PineconeMMRIndexingPipeline initialization and setup.

    Validates that the pipeline correctly initializes from configuration
    dictionaries or YAML config files, sets up the PineconeVectorDB
    connection, and extracts Pinecone-specific parameters.

    Configuration sources:
        - Dict with pinecone section (api_key, index_name, namespace, etc.)
        - Path to YAML configuration file

    Attributes:
        pipeline: PineconeMMRIndexingPipeline instance under test
    """

    @patch("vectordb.langchain.mmr.indexing.pinecone.PineconeVectorDB")
    @patch("vectordb.langchain.mmr.indexing.pinecone.EmbedderHelper.create_embedder")
    def test_init_with_valid_config(
        self, mock_embedder_helper, mock_db_cls, pinecone_config
    ):
        """Test pipeline initialization from configuration dictionary.

        Verifies that a valid config dict properly initializes:
        - PineconeVectorDB with api_key and index_name
        - Embedder with specified model and device
        - Pipeline attributes (index_name, namespace) are accessible

        Args:
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db_cls: Mock for PineconeVectorDB class.
            pinecone_config: Fixture providing valid Pinecone configuration.

        Asserts:
            Pipeline instance is created successfully
            index_name is "test-index" from config
            namespace is "test" from config
        """
        mock_embedder = MagicMock()
        mock_embedder_helper.return_value = mock_embedder

        mock_db_instance = MagicMock()
        mock_db_cls.return_value = mock_db_instance

        pipeline = PineconeMMRIndexingPipeline(pinecone_config)
        assert pipeline is not None
        assert pipeline.index_name == "test-index"
        assert pipeline.namespace == "test"

    @patch("vectordb.langchain.mmr.indexing.pinecone.PineconeVectorDB")
    @patch("vectordb.langchain.mmr.indexing.pinecone.EmbedderHelper.create_embedder")
    def test_init_with_config_path(self, mock_embedder_helper, mock_db_cls, tmp_path):
        """Test pipeline initialization from YAML configuration file.

        Validates loading configuration from external YAML file including
        all Pinecone-specific settings: api_key, index_name, namespace,
        dimension, and metric parameters.

        Args:
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db_cls: Mock for PineconeVectorDB class.
            tmp_path: pytest fixture providing temporary directory.

        Asserts:
            Pipeline instance is created from file path successfully
        """
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
pinecone:
  api_key: test-key
  index_name: test-index
  namespace: test
  dimension: 384
  metric: cosine
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

        pipeline = PineconeMMRIndexingPipeline(str(config_file))
        assert pipeline is not None

    class TestRun:
        """Tests for the run() method document indexing workflow.

        Validates the end-to-end indexing process including:
        - Document loading from configured dataloader
        - Document embedding via EmbedderHelper
        - Index creation in Pinecone (if not exists)
        - Vector upsert to Pinecone with namespace isolation
        - MMR threshold configuration for retrieval stage

        The run() method orchestrates the complete pipeline from
        raw documents to indexed vectors ready for MMR search.
        """

        @patch("vectordb.langchain.mmr.indexing.pinecone.PineconeVectorDB")
        @patch(
            "vectordb.langchain.mmr.indexing.pinecone.EmbedderHelper.create_embedder"
        )
        @patch(
            "vectordb.langchain.mmr.indexing.pinecone.EmbedderHelper.embed_documents"
        )
        @patch("vectordb.langchain.mmr.indexing.pinecone.DataloaderCatalog.create")
        def test_run_with_documents(
            self,
            mock_get_docs,
            mock_embed_docs,
            mock_embedder_helper,
            mock_db_cls,
            sample_documents,
            pinecone_config,
        ):
            """Test document indexing with valid documents.

            Verifies the full indexing workflow when documents are available:
            1. Dataloader returns sample documents
            2. Documents are embedded with 384-dim vectors
            3. Index is created if needed
            4. Vectors are upserted to Pinecone
            5. Returns count of indexed documents

            Args:
                mock_get_docs: Mock for DataloaderCatalog.create.
                mock_embed_docs: Mock for EmbedderHelper.embed_documents.
                mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
                mock_db_cls: Mock for PineconeVectorDB class.
                sample_documents: Fixture with 5 sample LangChain documents.
                pinecone_config: Fixture with Pinecone configuration.

            Asserts:
                Result contains documents_indexed equal to sample_documents length
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
            mock_db_instance.create_index.return_value = True
            mock_db_cls.return_value = mock_db_instance

            pipeline = PineconeMMRIndexingPipeline(pinecone_config)
            result = pipeline.run()

            assert result["documents_indexed"] == len(sample_documents)

        @patch("vectordb.langchain.mmr.indexing.pinecone.PineconeVectorDB")
        @patch(
            "vectordb.langchain.mmr.indexing.pinecone.EmbedderHelper.create_embedder"
        )
        @patch("vectordb.langchain.mmr.indexing.pinecone.DataloaderCatalog.create")
        def test_run_with_empty_documents(
            self,
            mock_get_docs,
            mock_embedder_helper,
            mock_db_cls,
            pinecone_config,
        ):
            """Test document indexing with empty document set.

            Validates graceful handling when no documents are loaded:
            - Empty list returned from dataloader
            - No embedding or upsert operations attempted
            - Returns documents_indexed = 0

            Args:
                mock_get_docs: Mock returning empty document list.
                mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
                mock_db_cls: Mock for PineconeVectorDB class.
                pinecone_config: Fixture with Pinecone configuration.

            Asserts:
                Result contains documents_indexed equal to 0
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

            pipeline = PineconeMMRIndexingPipeline(pinecone_config)
            result = pipeline.run()

            assert result["documents_indexed"] == 0

    class TestPineconeConfiguration:
        """Tests for Pinecone-specific configuration handling.

        Validates extraction and application of Pinecone-specific settings:
        - API key authentication passed to PineconeVectorDB
        - Namespace for multi-tenant data isolation
        - Dimension matching embedding model output
        - Default values for optional configuration fields

        These tests ensure proper integration with Pinecone's
        index and namespace management features.
        """

        @patch("vectordb.langchain.mmr.indexing.pinecone.PineconeVectorDB")
        @patch(
            "vectordb.langchain.mmr.indexing.pinecone.EmbedderHelper.create_embedder"
        )
        def test_api_key_from_config(
            self, mock_embedder_helper, mock_db_cls, pinecone_config
        ):
            """Test API key is passed correctly to PineconeVectorDB.

            Verifies that the api_key from configuration is properly
            extracted and passed to the PineconeVectorDB constructor.

            Args:
                mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
                mock_db_cls: Mock for PineconeVectorDB class.
                pinecone_config: Fixture with api_key="test-key".

            Asserts:
                PineconeVectorDB called with api_key="test-key" and index_name
            """
            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_db_instance = MagicMock()
            mock_db_cls.return_value = mock_db_instance

            pipeline = PineconeMMRIndexingPipeline(pinecone_config)
            assert pipeline is not None

            mock_db_cls.assert_called_once_with(
                api_key="test-key", index_name="test-index"
            )

        @patch("vectordb.langchain.mmr.indexing.pinecone.PineconeVectorDB")
        @patch(
            "vectordb.langchain.mmr.indexing.pinecone.EmbedderHelper.create_embedder"
        )
        def test_default_namespace(self, mock_embedder_helper, mock_db_cls):
            """Test default namespace when not specified in config.

            Validates that namespace defaults to empty string ("") when
            omitted from configuration, enabling operations on the
            default Pinecone namespace.

            Args:
                mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
                mock_db_cls: Mock for PineconeVectorDB class.

            Asserts:
                pipeline.namespace is empty string when config omits namespace
            """
            config = {
                "dataloader": {"type": "arc", "limit": 10},
                "embeddings": {"model": "test-model", "device": "cpu"},
                "pinecone": {
                    "api_key": "test-key",
                    "index_name": "test-index",
                    "dimension": 384,
                },
            }

            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_db_instance = MagicMock()
            mock_db_cls.return_value = mock_db_instance

            pipeline = PineconeMMRIndexingPipeline(config)
            assert pipeline.namespace == ""

        @patch("vectordb.langchain.mmr.indexing.pinecone.PineconeVectorDB")
        @patch(
            "vectordb.langchain.mmr.indexing.pinecone.EmbedderHelper.create_embedder"
        )
        def test_default_dimension(self, mock_embedder_helper, mock_db_cls):
            """Test default dimension when not specified in config.

            Validates that dimension defaults to 384 (MiniLM-L6-v2 output)
            when omitted from configuration. This matches the default
            embedding model used by the pipeline.

            Args:
                mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
                mock_db_cls: Mock for PineconeVectorDB class.

            Asserts:
                pipeline.dimension is 384 when config omits dimension
            """
            config = {
                "dataloader": {"type": "arc", "limit": 10},
                "embeddings": {"model": "test-model", "device": "cpu"},
                "pinecone": {
                    "api_key": "test-key",
                    "index_name": "test-index",
                },
            }

            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_db_instance = MagicMock()
            mock_db_cls.return_value = mock_db_instance

            pipeline = PineconeMMRIndexingPipeline(config)
            assert pipeline.dimension == 384
