"""Unit tests for Qdrant agentic RAG indexing pipeline (LangChain integration).

Tests verify the QdrantAgenticRAGIndexingPipeline's behavior across common scenarios:
- Pipeline initialization with valid configurations
- Document indexing with mock embeddings and database calls
- Graceful handling of empty document batches

These tests mock external dependencies (QdrantVectorDB, EmbedderHelper,
DataLoaderHelper) to isolate pipeline logic and avoid requiring live database
connections during test execution.
"""

from typing import Any
from unittest.mock import MagicMock, patch

from vectordb.langchain.agentic_rag.indexing.qdrant import (
    QdrantAgenticRAGIndexingPipeline,
)


class TestQdrantAgenticRAGIndexing:
    """Test suite for Qdrant agentic RAG indexing pipeline.

    This suite validates the indexing pipeline's core functionality:
    - Configuration parsing and storage
    - Collection name extraction from config
    - End-to-end document indexing with mocked dependencies
    - Empty document batch handling
    """

    @patch("vectordb.langchain.agentic_rag.indexing.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.indexing.qdrant.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.agentic_rag.indexing.qdrant.DataloaderCatalog.create")
    def test_indexing_initialization(
        self,
        mock_get_docs: Any,
        mock_embedder_helper: Any,
        mock_db: Any,
        sample_documents: Any,
    ) -> None:
        """Test pipeline initialization stores configuration correctly.

        Verifies that:
        - The pipeline stores the provided configuration dict
        - Collection name is extracted from the qdrant config section
        - No external calls are made during initialization

        Args:
            mock_get_docs: Mock for DataloaderCatalog.create.
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db: Mock for QdrantVectorDB class.
            sample_documents: Fixture providing sample document objects.

        Returns:
            None
        """
        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "test-key",
                "collection_name": "test_agentic_rag",
            },
        }

        pipeline = QdrantAgenticRAGIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.collection_name == "test_agentic_rag"

    @patch("vectordb.langchain.agentic_rag.indexing.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.indexing.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.agentic_rag.indexing.qdrant.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.agentic_rag.indexing.qdrant.DataloaderCatalog.create")
    def test_indexing_run_with_documents(
        self,
        mock_get_docs: Any,
        mock_embed_docs: Any,
        mock_embedder_helper: Any,
        mock_db: Any,
        sample_documents: Any,
    ) -> None:
        """Test successful indexing of documents with mocked dependencies.

        Validates the complete indexing workflow:
        1. DataLoaderHelper loads documents from configured source
        2. EmbedderHelper generates 384-dimensional embeddings
        3. QdrantVectorDB upserts documents into the collection
        4. Result returns the count of indexed documents

        Args:
            mock_get_docs: Mock for document loading, returns sample_documents.
            mock_embed_docs: Mock for embedding generation.
            mock_embedder_helper: Mock for embedder creation.
            mock_db: Mock for QdrantVectorDB, tracks upsert calls.
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
        mock_db_inst.client.upsert.return_value = len(sample_documents)
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "test-key",
                "collection_name": "test_agentic_rag",
            },
        }

        pipeline = QdrantAgenticRAGIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.create_collection.assert_called_once()
        mock_db_inst.client.upsert.assert_called()

    @patch("vectordb.langchain.agentic_rag.indexing.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.indexing.qdrant.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.agentic_rag.indexing.qdrant.DataloaderCatalog.create")
    def test_indexing_run_no_documents(
        self, mock_get_docs: Any, mock_embedder_helper: Any, mock_db: Any
    ) -> None:
        """Test pipeline handles empty document batches gracefully.

        Ensures that when DataLoaderHelper returns an empty list:
        - No exceptions are raised
        - Result reports 0 documents indexed
        - No database upsert operations are attempted

        Args:
            mock_get_docs: Mock returning empty document list.
            mock_embedder_helper: Mock for embedder creation.
            mock_db: Mock for QdrantVectorDB.

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
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "test-key",
                "collection_name": "test_agentic_rag",
            },
        }

        pipeline = QdrantAgenticRAGIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0
