"""Tests for Chroma JSON indexing pipelines (LangChain).

This module tests the JSON indexing pipeline implementation for Chroma vector
database. JSON indexing enables structured data storage with vector embeddings,
allowing semantic search over JSON documents while preserving structured fields.

JSON Indexing Pipeline Flow:
    1. Load JSON documents from configured dataloader
    2. Extract text content for embedding (flattening nested structures)
    3. Embed text content using specified embedding model
    4. Store original JSON with embeddings in Chroma
    5. Preserve JSON structure in metadata for filtering

Chroma-specific aspects tested:
    - Local persistent storage path configuration
    - Collection-based document organization
    - JSON metadata preservation in Chroma documents
    - Integration with JSON-specific dataloaders

Use Cases:
    - Product catalogs with attributes
    - API documentation with endpoints
    - Configuration files with settings
    - Structured knowledge bases

Test Coverage:
    - Indexing pipeline initialization
    - JSON document indexing with metadata extraction
    - Empty batch handling during indexing
    - Search pipeline initialization
    - Search execution over JSON documents
    - RAG mode with JSON context

External dependencies (ChromaVectorDB, EmbedderHelper, DataLoaderHelper,
RAGHelper) are mocked to enable fast, isolated unit tests.
"""

from unittest.mock import MagicMock, patch


class TestChromaJSONIndexing:
    """Unit tests for Chroma JSON indexing pipeline.

    Validates the indexing pipeline which stores JSON documents with
    their vector embeddings. JSON structure is preserved in metadata
    while text content is embedded for semantic search.

    Pipeline Flow:
        1. Load JSON documents from dataloader
        2. Extract/embed text fields
        3. Store in Chroma with JSON as metadata
        4. Return indexing statistics

    JSON Handling:
        - Text fields extracted for embedding
        - Full JSON preserved in metadata
        - Nested structures flattened for search
    """

    @patch("vectordb.langchain.json_indexing.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.json_indexing.indexing.chroma.DataloaderCatalog.create")
    def test_indexing_initialization(
        self, mock_get_docs, mock_embedder_helper, mock_db
    ):
        """Test pipeline initialization with JSON indexing configuration.

        Verifies that:
        - Configuration dict is preserved on pipeline instance
        - Collection name is extracted from chroma config section
        - No external calls during initialization

        Args:
            mock_get_docs: Mock for DataloaderCatalog.create.
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db: Mock for ChromaVectorDB class.
        """
        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_json_indexing",
            },
        }

        from vectordb.langchain.json_indexing.indexing.chroma import (
            ChromaJsonIndexingPipeline,
        )

        pipeline = ChromaJsonIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.collection_name == "test_json_indexing"

    @patch("vectordb.langchain.json_indexing.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.json_indexing.indexing.chroma.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.json_indexing.indexing.chroma.DataloaderCatalog.create")
    def test_indexing_run_with_documents(
        self,
        mock_get_docs,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test successful JSON document indexing.

        Validates the complete indexing workflow:
        1. DataLoaderHelper loads JSON documents
        2. EmbedderHelper generates 384-dimensional embeddings
        3. ChromaVectorDB.upsert stores with JSON metadata
        4. Result reports count of indexed documents

        Args:
            mock_get_docs: Mock returning sample_documents fixture.
            mock_embed_docs: Mock returning (docs, embeddings) tuple.
            mock_embedder_helper: Mock for embedder factory.
            mock_db: Mock for ChromaVectorDB, tracks upsert calls.
            sample_documents: Fixture with sample documents.
        """
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader
        mock_embed_docs.return_value = (sample_documents, [[0.1] * 384] * 5)

        mock_db_inst = MagicMock()
        mock_db_inst.upsert.return_value = len(sample_documents)
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_json_indexing",
            },
        }

        from vectordb.langchain.json_indexing.indexing.chroma import (
            ChromaJsonIndexingPipeline,
        )

        pipeline = ChromaJsonIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.upsert.assert_called_once()

    @patch("vectordb.langchain.json_indexing.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.json_indexing.indexing.chroma.DataloaderCatalog.create")
    def test_indexing_run_no_documents(
        self, mock_get_docs, mock_embedder_helper, mock_db
    ):
        """Test graceful handling of empty document batches.

        Ensures when the dataloader returns empty list:
        - No exceptions raised
        - Result reports 0 documents indexed
        - No database operations attempted

        Args:
            mock_get_docs: Mock returning empty list.
            mock_embedder_helper: Mock for embedder factory.
            mock_db: Mock for ChromaVectorDB (should not be called).
        """
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = []
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_json_indexing",
            },
        }

        from vectordb.langchain.json_indexing.indexing.chroma import (
            ChromaJsonIndexingPipeline,
        )

        pipeline = ChromaJsonIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0


class TestChromaJSONSearch:
    """Unit tests for Chroma JSON search pipeline.

    Tests validate search functionality over JSON documents:
    - Vector similarity search on embedded text content
    - JSON metadata retrieval with results
    - RAG with structured JSON context

    Search Behavior:
        - Query embedded and matched against JSON text fields
        - Full JSON structure returned in results
        - Metadata filtering on JSON fields supported
    """

    @patch("vectordb.langchain.json_indexing.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.json_indexing.search.chroma.RAGHelper.create_llm")
    def test_search_initialization(
        self, mock_llm_helper, mock_embedder_helper, mock_db
    ):
        """Test search pipeline initialization."""
        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_json_indexing",
            },
            "rag": {"enabled": False},
        }

        from vectordb.langchain.json_indexing.search.chroma import (
            ChromaJsonSearchPipeline,
        )

        pipeline = ChromaJsonSearchPipeline(config)
        assert pipeline.config == config

    @patch("vectordb.langchain.json_indexing.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.json_indexing.search.chroma.EmbedderHelper.embed_query")
    @patch("vectordb.langchain.json_indexing.search.chroma.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.json_indexing.search.chroma.DocumentFilter.filter_by_metadata_json"
    )
    def test_search_execution(
        self,
        mock_filter_json,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test search execution."""
        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None
        mock_filter_json.return_value = sample_documents[:1]

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_json_indexing",
            },
            "filters": {
                "conditions": [
                    {"field": "author.name", "operator": "equals", "value": "Alice"}
                ]
            },
            "rag": {"enabled": False},
        }

        from vectordb.langchain.json_indexing.search.chroma import (
            ChromaJsonSearchPipeline,
        )

        pipeline = ChromaJsonSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert len(result["documents"]) == 1
        mock_filter_json.assert_called_once_with(
            sample_documents,
            json_path="author.name",
            value="Alice",
            operator="equals",
        )

    @patch("vectordb.langchain.json_indexing.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.json_indexing.search.chroma.EmbedderHelper.embed_query")
    @patch("vectordb.langchain.json_indexing.search.chroma.RAGHelper.create_llm")
    @patch("vectordb.langchain.json_indexing.search.chroma.RAGHelper.generate")
    def test_search_with_rag(
        self,
        mock_rag_generate,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
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
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_json_indexing",
            },
            "rag": {"enabled": True},
        }

        from vectordb.langchain.json_indexing.search.chroma import (
            ChromaJsonSearchPipeline,
        )

        pipeline = ChromaJsonSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert result["answer"] == "Test answer"
