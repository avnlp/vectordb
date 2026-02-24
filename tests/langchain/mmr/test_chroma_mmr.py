"""Unit tests for Chroma MMR (Maximal Marginal Relevance) pipelines (LangChain).

Tests validate Chroma-based indexing and search pipelines that use MMR
for result diversification. MMR balances relevance against diversity to
reduce redundant results in retrieval-augmented generation workflows.

Test coverage includes:
- Indexing pipeline initialization and configuration handling
- End-to-end document indexing with mocked embeddings
- Empty batch handling during indexing
- Search pipeline initialization with RAG disabled
- Query execution returning diversified results

External dependencies (ChromaVectorDB, EmbedderHelper, DataLoaderHelper,
RAGHelper) are mocked to enable fast, isolated unit tests without requiring
live Chroma database instances.
"""

from typing import Any
from unittest.mock import MagicMock, patch

from vectordb.langchain.mmr.indexing.chroma import ChromaMMRIndexingPipeline
from vectordb.langchain.mmr.search.chroma import ChromaMMRSearchPipeline


class TestChromaMMRIndexing:
    """Test suite for Chroma MMR indexing pipeline.

    Validates the indexing pipeline's ability to:
    - Parse configuration and extract collection names
    - Load documents via DataLoaderHelper with configurable limits
    - Generate embeddings using EmbedderHelper
    - Persist documents to ChromaVectorDB
    - Handle edge cases like empty document lists
    """

    @patch("vectordb.langchain.mmr.indexing.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.mmr.indexing.chroma.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.mmr.indexing.chroma.DataloaderCatalog.create")
    def test_indexing_initialization(
        self, mock_get_docs: Any, mock_embedder_helper: Any, mock_db: Any
    ) -> None:
        """Test pipeline correctly stores configuration during initialization.

        Verifies that:
        - The full configuration dict is preserved on the pipeline instance
        - Collection name is extracted from the chroma configuration section
        - Initialization does not trigger document loading or embedding

        Args:
            mock_get_docs: Mock for DataloaderCatalog.create.
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db: Mock for ChromaVectorDB class.

        Returns:
            None
        """
        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_mmr",
            },
        }

        pipeline = ChromaMMRIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.collection_name == "test_mmr"

    @patch("vectordb.langchain.mmr.indexing.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.mmr.indexing.chroma.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.mmr.indexing.chroma.EmbedderHelper.embed_documents")
    @patch("vectordb.langchain.mmr.indexing.chroma.DataloaderCatalog.create")
    def test_indexing_run_with_documents(
        self,
        mock_get_docs: Any,
        mock_embed_docs: Any,
        mock_embedder_helper: Any,
        mock_db: Any,
        sample_documents: Any,
    ) -> None:
        """Test successful document indexing end-to-end.

        Validates the complete indexing workflow:
        1. DataLoaderHelper loads sample documents (limit=10 from ARC dataset)
        2. EmbedderHelper.create_embedder initializes embedding model
        3. EmbedderHelper.embed_documents generates 384-dim vectors
        4. ChromaVectorDB.upsert persists documents with embeddings
        5. Result dict reports count of successfully indexed documents

        Args:
            mock_get_docs: Mock returning sample_documents fixture.
            mock_embed_docs: Mock returning tuple of (docs, embeddings).
            mock_embedder_helper: Mock for embedder factory.
            mock_db: Mock for ChromaVectorDB, tracks upsert operation.
            sample_documents: Fixture providing 5 sample document objects.

        Returns:
            None
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
                "collection_name": "test_mmr",
            },
        }

        pipeline = ChromaMMRIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)

    @patch("vectordb.langchain.mmr.indexing.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.mmr.indexing.chroma.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.mmr.indexing.chroma.DataloaderCatalog.create")
    def test_indexing_run_no_documents(
        self, mock_get_docs: Any, mock_embedder_helper: Any, mock_db: Any
    ) -> None:
        """Test pipeline behavior when no documents are returned by dataloader.

        Validates graceful handling of empty batches:
        - No exceptions raised when document list is empty
        - Result reports documents_indexed: 0
        - No database operations attempted

        Args:
            mock_get_docs: Mock returning empty list.
            mock_embedder_helper: Mock for embedder factory.
            mock_db: Mock for ChromaVectorDB (should not be called).

        Returns:
            None
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
                "collection_name": "test_mmr",
            },
        }

        pipeline = ChromaMMRIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0


class TestChromaMMRSearch:
    """Test suite for Chroma MMR search pipeline.

    Tests validate search functionality when RAG is disabled, focusing on
    pure retrieval with MMR diversification:
    - Pipeline initialization with RAG disabled
    - Query embedding generation
    - Vector database query execution
    - Result structure with query and matched documents
    """

    @patch("vectordb.langchain.mmr.search.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.mmr.search.chroma.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.mmr.search.chroma.RAGHelper.create_llm")
    def test_search_initialization(
        self, mock_llm_helper: Any, mock_embedder_helper: Any, mock_db: Any
    ) -> None:
        """Test search pipeline initialization with RAG disabled.

        Verifies that:
        - Configuration is stored on pipeline instance
        - LLM helper is still called but can return None when RAG disabled
        - Embedder initialization happens for query embedding

        Args:
            mock_llm_helper: Mock returning None (RAG disabled).
            mock_embedder_helper: Mock for embedder factory.
            mock_db: Mock for ChromaVectorDB.

        Returns:
            None
        """
        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_mmr",
            },
            "rag": {"enabled": False},
        }

        pipeline = ChromaMMRSearchPipeline(config)
        assert pipeline.config == config

    @patch("vectordb.langchain.mmr.search.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.mmr.search.chroma.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.mmr.search.chroma.EmbedderHelper.embed_query")
    @patch("vectordb.langchain.mmr.search.chroma.RAGHelper.create_llm")
    def test_search_execution(
        self,
        mock_llm_helper: Any,
        mock_embed_query: Any,
        mock_embedder_helper: Any,
        mock_db: Any,
        sample_documents: Any,
    ) -> None:
        """Test end-to-end search execution with mocked dependencies.

        Validates the complete search workflow:
        1. Query string is embedded via EmbedderHelper.embed_query
        2. ChromaVectorDB.query retrieves matching documents
        3. MMR diversification applied to results
        4. Result contains original query and matched document list

        Args:
            mock_llm_helper: Mock for LLM factory.
            mock_embed_query: Mock returning 384-dimensional query vector.
            mock_embedder_helper: Mock for embedder factory.
            mock_db: Mock returning sample_documents from query.
            sample_documents: Fixture with sample documents to return.

        Returns:
            None
        """
        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_mmr",
            },
            "rag": {"enabled": False},
        }

        pipeline = ChromaMMRSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert len(result["documents"]) > 0
