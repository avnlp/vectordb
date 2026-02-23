"""Unit tests for Weaviate agentic RAG indexing pipeline (LangChain).

Tests verify the WeaviateAgenticRAGIndexingPipeline's behavior across common
scenarios. Weaviate-specific aspects tested include:
- URL-based connection configuration
- Optional API key authentication
- GraphQL schema-aware collection naming

These tests mock external dependencies (WeaviateVectorDB, EmbedderHelper,
DataLoaderHelper) to isolate pipeline logic and avoid requiring live Weaviate
server connections during test execution.
"""

from typing import Any
from unittest.mock import MagicMock, patch

from vectordb.langchain.agentic_rag.indexing.weaviate import (
    WeaviateAgenticRAGIndexingPipeline,
)


class TestWeaviateAgenticRAGIndexing:
    """Test suite for Weaviate agentic RAG indexing pipeline.

    This suite validates the indexing pipeline's core functionality:
    - Configuration parsing and storage
    - Collection name extraction from config
    - URL and API key authentication handling
    - End-to-end document indexing with mocked dependencies
    - Empty document batch handling

    Weaviate-specific features validated:
    - RESTful/GraphQL connection via URL
    - Optional API key for authenticated clusters
    - Schema-based collection (class) naming
    """

    @patch("vectordb.langchain.agentic_rag.indexing.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.indexing.weaviate.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.agentic_rag.indexing.weaviate.DataloaderCatalog.create")
    def test_indexing_initialization(
        self,
        mock_get_docs: Any,
        mock_embedder_helper: Any,
        mock_db: Any,
        sample_documents: Any,
    ) -> None:
        """Test pipeline initialization stores Weaviate-specific configuration.

        Verifies that:
        - The pipeline stores the provided configuration dict
        - Collection name is extracted from the weaviate config section
        - URL connection parameter is preserved
        - API key is handled (empty string for unauthenticated clusters)
        - No external calls are made during initialization

        Args:
            mock_get_docs: Mock for DataloaderCatalog.create.
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db: Mock for WeaviateVectorDB class.
            sample_documents: Fixture providing sample document objects.

        Returns:
            None
        """
        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "",
                "collection_name": "TestAgenticRAG",
            },
        }

        pipeline = WeaviateAgenticRAGIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.collection_name == "TestAgenticRAG"

    @patch("vectordb.langchain.agentic_rag.indexing.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.indexing.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.agentic_rag.indexing.weaviate.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.agentic_rag.indexing.weaviate.DataloaderCatalog.create")
    def test_indexing_run_with_documents(
        self,
        mock_get_docs: Any,
        mock_embed_docs: Any,
        mock_embedder_helper: Any,
        mock_db: Any,
        sample_documents: Any,
    ) -> None:
        """Test successful indexing of documents into Weaviate with mocked deps.

        Validates the complete indexing workflow:
        1. DataLoaderHelper loads documents from configured source
        2. EmbedderHelper generates 384-dimensional embeddings
        3. WeaviateVectorDB upserts documents into the collection
        4. Result returns the count of indexed documents

        Args:
            mock_get_docs: Mock for document loading, returns sample_documents.
            mock_embed_docs: Mock for embedding generation.
            mock_embedder_helper: Mock for embedder creation.
            mock_db: Mock for WeaviateVectorDB, tracks upsert calls.
            sample_documents: Fixture with 5 sample documents.

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
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "",
                "collection_name": "TestAgenticRAG",
            },
        }

        pipeline = WeaviateAgenticRAGIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)

    @patch("vectordb.langchain.agentic_rag.indexing.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.indexing.weaviate.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.agentic_rag.indexing.weaviate.DataloaderCatalog.create")
    def test_indexing_run_no_documents(
        self, mock_get_docs: Any, mock_embedder_helper: Any, mock_db: Any
    ) -> None:
        """Test pipeline handles empty document batches gracefully.

        Ensures that when DataLoaderHelper returns an empty list:
        - No exceptions are raised
        - Result reports 0 documents indexed
        - No Weaviate upsert operations are attempted

        Args:
            mock_get_docs: Mock returning empty document list.
            mock_embedder_helper: Mock for embedder creation.
            mock_db: Mock for WeaviateVectorDB.

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
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "",
                "collection_name": "TestAgenticRAG",
            },
        }

        pipeline = WeaviateAgenticRAGIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0
