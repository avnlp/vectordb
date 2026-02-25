"""Tests for Pinecone JSON indexing pipelines (LangChain).

This module tests the JSON indexing pipeline implementation for Pinecone vector
database. Pinecone offers managed vector search with native JSON metadata support.

Test Coverage:
    - Indexing pipeline initialization
    - JSON document indexing with metadata extraction
    - Empty batch handling during indexing
    - Namespace configuration
    - Search pipeline initialization
    - Search execution over JSON documents
    - RAG mode with JSON context
"""

from unittest.mock import MagicMock, patch


class TestPineconeJSONIndexing:
    """Unit tests for Pinecone JSON indexing pipeline.

    Validates the indexing pipeline which stores JSON documents with
    their vector embeddings in Pinecone. Pinecone's managed infrastructure
    enables automatic scaling and index management.
    """

    @patch("vectordb.langchain.json_indexing.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.indexing.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.json_indexing.indexing.pinecone.DataloaderCatalog.create"
    )
    def test_indexing_initialization(
        self,
        mock_get_docs: MagicMock,
        mock_embedder_helper: MagicMock,
        mock_db: MagicMock,
    ) -> None:
        """Test pipeline initialization with JSON indexing configuration.

        Verifies that:
        - Configuration dict is preserved on pipeline instance
        - Namespace is extracted from pinecone config section
        - No external calls during initialization
        """
        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-json-index",
                "namespace": "test",
            },
        }

        from vectordb.langchain.json_indexing.indexing.pinecone import (
            PineconeJsonIndexingPipeline,
        )

        pipeline = PineconeJsonIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.namespace == "test"

    @patch("vectordb.langchain.json_indexing.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.indexing.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.json_indexing.indexing.pinecone.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.json_indexing.indexing.pinecone.DataloaderCatalog.create"
    )
    def test_indexing_run_with_documents(
        self,
        mock_get_docs: MagicMock,
        mock_embed_docs: MagicMock,
        mock_embedder_helper: MagicMock,
        mock_db: MagicMock,
        sample_documents: list,
    ) -> None:
        """Test successful JSON document indexing.

        Validates the complete indexing workflow:
        1. DataLoaderHelper loads JSON documents
        2. EmbedderHelper generates embeddings
        3. PineconeVectorDB.upsert_documents stores with JSON metadata
        4. Result reports count of indexed documents
        """
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader
        mock_embed_docs.return_value = (
            sample_documents,
            [[0.1] * 384] * len(sample_documents),
        )

        mock_db_inst = MagicMock()
        mock_db_inst.upsert_documents.return_value = len(sample_documents)
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-json-index",
                "namespace": "test",
            },
        }

        from vectordb.langchain.json_indexing.indexing.pinecone import (
            PineconeJsonIndexingPipeline,
        )

        pipeline = PineconeJsonIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        assert result["index_name"] == "test-json-index"
        assert result["namespace"] == "test"
        mock_db_inst.upsert_documents.assert_called_once()

    @patch("vectordb.langchain.json_indexing.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.indexing.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.json_indexing.indexing.pinecone.DataloaderCatalog.create"
    )
    def test_indexing_run_no_documents(
        self,
        mock_get_docs: MagicMock,
        mock_embedder_helper: MagicMock,
        mock_db: MagicMock,
    ) -> None:
        """Test graceful handling of empty document batches.

        Ensures when the dataloader returns empty list:
        - No exceptions raised
        - Result reports 0 documents indexed
        - No upsert operations attempted
        """
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = []
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader

        mock_db_inst = MagicMock()
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-json-index",
                "namespace": "test",
            },
        }

        from vectordb.langchain.json_indexing.indexing.pinecone import (
            PineconeJsonIndexingPipeline,
        )

        pipeline = PineconeJsonIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0
        assert result["index_name"] == "test-json-index"
        # Verify upsert was not called since there were no documents
        mock_db_inst.upsert_documents.assert_not_called()

    @patch("vectordb.langchain.json_indexing.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.indexing.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.json_indexing.indexing.pinecone.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.json_indexing.indexing.pinecone.DataloaderCatalog.create"
    )
    def test_indexing_with_default_namespace(
        self,
        mock_get_docs: MagicMock,
        mock_embed_docs: MagicMock,
        mock_embedder_helper: MagicMock,
        mock_db: MagicMock,
        sample_documents: list,
    ) -> None:
        """Test indexing with default namespace (None)."""
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader
        mock_embed_docs.return_value = (
            sample_documents,
            [[0.1] * 384] * len(sample_documents),
        )

        mock_db_inst = MagicMock()
        mock_db_inst.upsert_documents.return_value = len(sample_documents)
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-json-index",
            },
        }

        from vectordb.langchain.json_indexing.indexing.pinecone import (
            PineconeJsonIndexingPipeline,
        )

        pipeline = PineconeJsonIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        assert result["namespace"] is None


class TestPineconeJSONSearch:
    """Unit tests for Pinecone JSON search pipeline.

    Tests validate search functionality over JSON documents:
    - Vector similarity search on embedded text content
    - JSON metadata retrieval with results
    - RAG with structured JSON context
    """

    @patch("vectordb.langchain.json_indexing.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.json_indexing.search.pinecone.RAGHelper.create_llm")
    def test_search_initialization(
        self,
        mock_llm_helper: MagicMock,
        mock_embedder_helper: MagicMock,
        mock_db: MagicMock,
    ) -> None:
        """Test search pipeline initialization."""
        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-json-index",
                "namespace": "test",
            },
            "rag": {"enabled": False},
        }

        from vectordb.langchain.json_indexing.search.pinecone import (
            PineconeJsonSearchPipeline,
        )

        pipeline = PineconeJsonSearchPipeline(config)
        assert pipeline.config == config

    @patch("vectordb.langchain.json_indexing.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.json_indexing.search.pinecone.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.json_indexing.search.pinecone.RAGHelper.create_llm")
    def test_search_execution(
        self,
        mock_llm_helper: MagicMock,
        mock_embed_query: MagicMock,
        mock_embedder_helper: MagicMock,
        mock_db: MagicMock,
        sample_documents: list,
    ) -> None:
        """Test search execution."""
        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-json-index",
                "namespace": "test",
            },
            "rag": {"enabled": False},
        }

        from vectordb.langchain.json_indexing.search.pinecone import (
            PineconeJsonSearchPipeline,
        )

        pipeline = PineconeJsonSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert len(result["documents"]) > 0

    @patch("vectordb.langchain.json_indexing.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.json_indexing.search.pinecone.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.json_indexing.search.pinecone.RAGHelper.create_llm")
    @patch("vectordb.langchain.json_indexing.search.pinecone.RAGHelper.generate")
    def test_search_with_rag(
        self,
        mock_rag_generate: MagicMock,
        mock_llm_helper: MagicMock,
        mock_embed_query: MagicMock,
        mock_embedder_helper: MagicMock,
        mock_db: MagicMock,
        sample_documents: list,
    ) -> None:
        """Test search with RAG generation."""
        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst

        mock_llm_inst = MagicMock()
        mock_llm_helper.return_value = mock_llm_inst
        mock_rag_generate.return_value = "Test answer"

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-json-index",
                "namespace": "test",
            },
            "rag": {"enabled": True},
        }

        from vectordb.langchain.json_indexing.search.pinecone import (
            PineconeJsonSearchPipeline,
        )

        pipeline = PineconeJsonSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert result["answer"] == "Test answer"
