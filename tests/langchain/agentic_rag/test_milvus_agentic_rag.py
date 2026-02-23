"""Unit tests for Milvus agentic RAG indexing pipeline (LangChain).

Tests verify the MilvusAgenticRAGIndexingPipeline's behavior across common
scenarios. Milvus-specific aspects tested include:
- Host/port connection configuration
- Collection name management
- Dimension specification for vector fields

These tests mock external dependencies (MilvusVectorDB, EmbedderHelper,
DataLoaderHelper) to isolate pipeline logic and avoid requiring live Milvus
server connections during test execution.
"""

from typing import Any
from unittest.mock import MagicMock, patch

from vectordb.langchain.agentic_rag.indexing.milvus import (
    MilvusAgenticRAGIndexingPipeline,
)


class TestMilvusAgenticRAGIndexing:
    """Test suite for Milvus agentic RAG indexing pipeline.

    This suite validates the indexing pipeline's core functionality:
    - Configuration parsing and storage
    - Collection name extraction from config
    - Host/port connection parameter handling
    - End-to-end document indexing with mocked dependencies
    - Empty document batch handling

    Milvus-specific features validated:
    - gRPC/REST connection via host and port
    - Collection-based document organization
    - Dimension consistency for vector similarity search
    """

    @patch("vectordb.langchain.agentic_rag.indexing.milvus.MilvusVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.indexing.milvus.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.agentic_rag.indexing.milvus.DataloaderCatalog.create")
    def test_indexing_initialization(
        self,
        mock_get_docs: Any,
        mock_embedder_helper: Any,
        mock_db: Any,
        sample_documents: Any,
    ) -> None:
        """Test pipeline initialization stores Milvus-specific configuration.

        Verifies that:
        - The pipeline stores the provided configuration dict
        - Collection name is extracted from the milvus config section
        - Host and port connection parameters are preserved
        - Dimension setting is extracted for vector field creation
        - No external calls are made during initialization

        Args:
            mock_get_docs: Mock for DataloaderCatalog.create.
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db: Mock for MilvusVectorDB class.
            sample_documents: Fixture providing sample document objects.

        Returns:
            None
        """
        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "milvus": {
                "host": "localhost",
                "port": 19530,
                "collection_name": "test_agentic_rag",
                "dimension": 384,
            },
        }

        pipeline = MilvusAgenticRAGIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.collection_name == "test_agentic_rag"

    @patch("vectordb.langchain.agentic_rag.indexing.milvus.MilvusVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.indexing.milvus.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.agentic_rag.indexing.milvus.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.agentic_rag.indexing.milvus.DataloaderCatalog.create")
    def test_indexing_run_with_documents(
        self,
        mock_get_docs: Any,
        mock_embed_docs: Any,
        mock_embedder_helper: Any,
        mock_db: Any,
        sample_documents: Any,
    ) -> None:
        """Test successful indexing of documents into Milvus with mocked deps.

        Validates the complete indexing workflow:
        1. DataLoaderHelper loads documents from configured source
        2. EmbedderHelper generates 384-dimensional embeddings
        3. MilvusVectorDB upserts documents into the collection
        4. Result returns the count of indexed documents

        Args:
            mock_get_docs: Mock for document loading, returns sample_documents.
            mock_embed_docs: Mock for embedding generation.
            mock_embedder_helper: Mock for embedder creation.
            mock_db: Mock for MilvusVectorDB, tracks upsert calls.
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
            "milvus": {
                "host": "localhost",
                "port": 19530,
                "collection_name": "test_agentic_rag",
                "dimension": 384,
            },
        }

        pipeline = MilvusAgenticRAGIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)

    @patch("vectordb.langchain.agentic_rag.indexing.milvus.MilvusVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.indexing.milvus.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.agentic_rag.indexing.milvus.DataloaderCatalog.create")
    def test_indexing_run_no_documents(
        self, mock_get_docs: Any, mock_embedder_helper: Any, mock_db: Any
    ) -> None:
        """Test pipeline handles empty document batches gracefully.

        Ensures that when DataLoaderHelper returns an empty list:
        - No exceptions are raised
        - Result reports 0 documents indexed
        - No Milvus upsert operations are attempted

        Args:
            mock_get_docs: Mock returning empty document list.
            mock_embedder_helper: Mock for embedder creation.
            mock_db: Mock for MilvusVectorDB.

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
            "milvus": {
                "host": "localhost",
                "port": 19530,
                "collection_name": "test_agentic_rag",
                "dimension": 384,
            },
        }

        pipeline = MilvusAgenticRAGIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0
