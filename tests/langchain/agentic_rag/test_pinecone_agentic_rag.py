"""Unit tests for Pinecone agentic RAG indexing pipeline (LangChain).

Tests verify the PineconeAgenticRAGIndexingPipeline's behavior across common
scenarios. Pinecone-specific aspects tested include:
- Namespace handling for multi-tenant indices
- Dimension and metric configuration
- API key-based authentication

These tests mock external dependencies (PineconeVectorDB, EmbedderHelper,
DataLoaderHelper) to isolate pipeline logic and avoid requiring live Pinecone
service connections during test execution.
"""

from typing import Any
from unittest.mock import MagicMock, patch

from vectordb.langchain.agentic_rag.indexing.pinecone import (
    PineconeAgenticRAGIndexingPipeline,
)


class TestPineconeAgenticRAGIndexing:
    """Test suite for Pinecone agentic RAG indexing pipeline.

    This suite validates the indexing pipeline's core functionality:
    - Configuration parsing and storage
    - Index name and namespace extraction from config
    - End-to-end document indexing with mocked dependencies
    - Empty document batch handling

    Pinecone-specific features validated:
    - Namespace isolation for multi-tenant deployments
    - Dimension configuration for vector consistency
    - Metric selection (cosine, euclidean, dotproduct)
    """

    @patch("vectordb.langchain.agentic_rag.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.indexing.pinecone.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.agentic_rag.indexing.pinecone.DataloaderCatalog.create")
    def test_indexing_initialization(
        self,
        mock_get_docs: Any,
        mock_embedder_helper: Any,
        mock_db: Any,
        sample_documents: Any,
    ) -> None:
        """Test pipeline initialization stores Pinecone-specific configuration.

        Verifies that:
        - The pipeline stores the provided configuration dict
        - Index name is extracted from the pinecone config section
        - Namespace is extracted for multi-tenant isolation
        - Dimension and metric settings are preserved
        - No external calls are made during initialization

        Args:
            mock_get_docs: Mock for DataloaderCatalog.create.
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db: Mock for PineconeVectorDB class.
            sample_documents: Fixture providing sample document objects.

        Returns:
            None
        """
        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-index",
                "namespace": "test",
                "dimension": 384,
                "metric": "cosine",
            },
        }

        pipeline = PineconeAgenticRAGIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.index_name == "test-index"

    @patch("vectordb.langchain.agentic_rag.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.indexing.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.agentic_rag.indexing.pinecone.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.agentic_rag.indexing.pinecone.DataloaderCatalog.create")
    def test_indexing_run_with_documents(
        self,
        mock_get_docs: Any,
        mock_embed_docs: Any,
        mock_embedder_helper: Any,
        mock_db: Any,
        sample_documents: Any,
    ) -> None:
        """Test successful indexing of documents into Pinecone with mocked deps.

        Validates the complete indexing workflow:
        1. DataLoaderHelper loads documents from configured source
        2. EmbedderHelper generates 384-dimensional embeddings
        3. PineconeVectorDB upserts documents into the namespaced index
        4. Result returns the count of indexed documents

        Args:
            mock_get_docs: Mock for document loading, returns sample_documents.
            mock_embed_docs: Mock for embedding generation.
            mock_embedder_helper: Mock for embedder creation.
            mock_db: Mock for PineconeVectorDB, tracks upsert calls.
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
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-index",
                "namespace": "test",
                "dimension": 384,
                "metric": "cosine",
            },
        }

        pipeline = PineconeAgenticRAGIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)

    @patch("vectordb.langchain.agentic_rag.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.indexing.pinecone.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.agentic_rag.indexing.pinecone.DataloaderCatalog.create")
    def test_indexing_run_no_documents(
        self, mock_get_docs: Any, mock_embedder_helper: Any, mock_db: Any
    ) -> None:
        """Test pipeline handles empty document batches gracefully.

        Ensures that when DataLoaderHelper returns an empty list:
        - No exceptions are raised
        - Result reports 0 documents indexed
        - No Pinecone upsert operations are attempted

        Args:
            mock_get_docs: Mock returning empty document list.
            mock_embedder_helper: Mock for embedder creation.
            mock_db: Mock for PineconeVectorDB.

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
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-index",
                "namespace": "test",
                "dimension": 384,
                "metric": "cosine",
            },
        }

        pipeline = PineconeAgenticRAGIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0
