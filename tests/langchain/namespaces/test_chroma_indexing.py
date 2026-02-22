"""Tests for Chroma namespace indexing pipeline (LangChain).

This module tests the ChromaNamespaceIndexingPipeline which orchestrates
document loading, embedding generation, and indexing into namespace-specific
Chroma collections.

Test Coverage:
    - Pipeline initialization with valid and invalid namespaces
    - End-to-end indexing pipeline execution
    - Empty document handling

All tests use mocking to avoid requiring actual Chroma database.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


class TestChromaNamespaceIndexingPipeline:
    """Unit tests for ChromaNamespaceIndexingPipeline high-level workflow.

    Validates the complete indexing pipeline that orchestrates document loading,
    embedding generation, and namespace-scoped indexing into Chroma collections.

    Tested Scenarios:
        - Pipeline initialization with valid namespace
        - Empty namespace rejection
        - End-to-end run returning indexed count
        - Empty dataset handling
    """

    @patch("vectordb.langchain.namespaces.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    def test_init_with_valid_namespace(
        self, mock_create_embedder, mock_db_cls, chroma_namespace_config: dict
    ):
        """Test pipeline initialization with valid namespace.

        Validates:
            - Namespace is stored correctly
            - Config is accessible after initialization
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db
        mock_create_embedder.return_value = MagicMock()

        from vectordb.langchain.namespaces.indexing.chroma import (
            ChromaNamespaceIndexingPipeline,
        )

        pipeline = ChromaNamespaceIndexingPipeline(chroma_namespace_config, "arc_train")

        assert pipeline.namespace == "arc_train"
        assert pipeline.config == chroma_namespace_config

    @patch("vectordb.langchain.namespaces.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    def test_init_with_empty_namespace_raises_error(
        self, mock_create_embedder, mock_db_cls, chroma_namespace_config: dict
    ):
        """Test initialization with empty namespace raises ValueError.

        Validates:
            - Empty string namespace is rejected
            - Appropriate error message is raised
        """
        from vectordb.langchain.namespaces.indexing.chroma import (
            ChromaNamespaceIndexingPipeline,
        )

        with pytest.raises(ValueError, match="namespace cannot be empty"):
            ChromaNamespaceIndexingPipeline(chroma_namespace_config, "")

    @patch("vectordb.langchain.namespaces.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.dataloaders.DataloaderCatalog.create")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_documents")
    def test_run_returns_indexed_count(
        self,
        mock_embed_documents,
        mock_get_documents,
        mock_create_embedder,
        mock_db_cls,
        chroma_namespace_config: dict,
    ):
        """Test run returns correct indexed document count.

        Validates:
            - Result contains documents_indexed key
            - Result contains namespace key
            - Indexed count matches upserted count
        """
        mock_db = MagicMock()
        mock_db.upsert.return_value = 3
        mock_db.list_collections.return_value = ["ns_arc_train"]
        mock_db_cls.return_value = mock_db

        mock_create_embedder.return_value = MagicMock()

        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = [
            Document(page_content="doc1", metadata={"id": "1"}),
            Document(page_content="doc2", metadata={"id": "2"}),
            Document(page_content="doc3", metadata={"id": "3"}),
        ]
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_documents.return_value = mock_loader

        mock_embed_documents.return_value = (
            [
                Document(page_content="doc1", metadata={"id": "1"}),
                Document(page_content="doc2", metadata={"id": "2"}),
                Document(page_content="doc3", metadata={"id": "3"}),
            ],
            [[0.1] * 384] * 3,
        )

        from vectordb.langchain.namespaces.indexing.chroma import (
            ChromaNamespaceIndexingPipeline,
        )

        pipeline = ChromaNamespaceIndexingPipeline(chroma_namespace_config, "arc_train")
        result = pipeline.run()

        assert "documents_indexed" in result
        assert result["namespace"] == "arc_train"
        assert result["documents_indexed"] == 3

    @patch("vectordb.langchain.namespaces.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.dataloaders.DataloaderCatalog.create")
    def test_run_with_no_documents_returns_zero(
        self,
        mock_get_documents,
        mock_create_embedder,
        mock_db_cls,
        chroma_namespace_config: dict,
    ):
        """Test run with no documents returns 0 indexed.

        Validates:
            - Empty dataset results in 0 documents indexed
            - Result namespace matches configured namespace
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        mock_create_embedder.return_value = MagicMock()

        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = []
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_documents.return_value = mock_loader

        from vectordb.langchain.namespaces.indexing.chroma import (
            ChromaNamespaceIndexingPipeline,
        )

        pipeline = ChromaNamespaceIndexingPipeline(chroma_namespace_config, "arc_train")
        result = pipeline.run()

        assert result["documents_indexed"] == 0
        assert result["namespace"] == "arc_train"
