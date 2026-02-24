"""Comprehensive unit tests for Pinecone multi-tenancy indexing pipeline.

This module tests the PineconeMultitenancyIndexingPipeline class with focus on:
- Initialization with various config formats
- Connection handling and index management
- Document loading from dataloaders
- Indexing run operations with tenant namespace isolation
- Batch processing and timing metrics
- Error handling scenarios
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.multi_tenancy.common.tenant_context import TenantContext
from vectordb.haystack.multi_tenancy.pinecone.indexing import (
    PineconeMultitenancyIndexingPipeline,
)


class TestPineconeMultitenancyIndexingPipeline:
    """Test suite for PineconeMultitenancyIndexingPipeline."""

    @patch("pinecone.Pinecone")
    @patch("pinecone.ServerlessSpec")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.load_config")
    def test_initialization_with_config_path(
        self,
        mock_load_config: MagicMock,
        mock_create_embedder: MagicMock,
        mock_serverless_spec: MagicMock,
        mock_pinecone_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test pipeline initialization with config file path."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
pinecone:
  api_key: test-key
  index: test-index
  dimension: 384
embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
  dimension: 384
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_config = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384,
            },
            "tenant": {"id": "test-tenant"},
        }
        mock_load_config.return_value = mock_config

        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.list_indexes.return_value = []
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline with config path
        pipeline = PineconeMultitenancyIndexingPipeline(str(config_file))

        # Assertions
        assert pipeline.config == mock_config
        assert pipeline.tenant_context.tenant_id == "test-tenant"
        mock_load_config.assert_called_once_with(str(config_file))

    @patch("pinecone.Pinecone")
    @patch("pinecone.ServerlessSpec")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder")
    def test_initialization_with_dict_config(
        self,
        mock_create_embedder: MagicMock,
        mock_serverless_spec: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test pipeline initialization with dictionary config."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
                "metric": "cosine",
                "cloud": "aws",
                "region": "us-east-1",
            },
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384,
            },
            "tenant": {"id": "dict-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.list_indexes.return_value = []
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline with dict config
        pipeline = PineconeMultitenancyIndexingPipeline(config)

        # Assertions
        assert pipeline.config == config
        assert pipeline.tenant_context.tenant_id == "dict-tenant"

    @patch("pinecone.Pinecone")
    @patch("pinecone.ServerlessSpec")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder")
    def test_initialization_with_tenant_context(
        self,
        mock_create_embedder: MagicMock,
        mock_serverless_spec: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test pipeline initialization with explicit tenant context."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384,
            },
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.list_indexes.return_value = []
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create explicit tenant context
        tenant_context = TenantContext(tenant_id="explicit-tenant")

        # Create pipeline with tenant context
        pipeline = PineconeMultitenancyIndexingPipeline(config, tenant_context)

        # Assertions
        assert pipeline.tenant_context.tenant_id == "explicit-tenant"

    @patch("pinecone.Pinecone")
    @patch("pinecone.ServerlessSpec")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder")
    def test_connect_creates_index_when_not_exists(
        self,
        mock_create_embedder: MagicMock,
        mock_serverless_spec: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test that _connect creates index when it doesn't exist."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "new-index",
                "dimension": 512,
                "metric": "euclidean",
                "cloud": "gcp",
                "region": "us-west1",
            },
            "embedding": {"dimension": 512},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        # Simulate index not existing
        mock_client.list_indexes.return_value = []
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = PineconeMultitenancyIndexingPipeline(config)

        # Assertions
        mock_client.create_index.assert_called_once()
        call_kwargs = mock_client.create_index.call_args.kwargs
        assert call_kwargs["name"] == "new-index"
        assert call_kwargs["dimension"] == 512
        assert call_kwargs["metric"] == "euclidean"
        mock_serverless_spec.assert_called_once_with(cloud="gcp", region="us-west1")
        mock_client.Index.assert_called_once_with("new-index")
        assert pipeline._index == mock_index

    @patch("pinecone.Pinecone")
    @patch("pinecone.ServerlessSpec")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder")
    def test_connect_uses_existing_index(
        self,
        mock_create_embedder: MagicMock,
        mock_serverless_spec: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test that _connect uses existing index without creating new one."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "existing-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        # Simulate index already existing
        mock_index_info = MagicMock()
        mock_index_info.name = "existing-index"
        mock_client.list_indexes.return_value = [mock_index_info]
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = PineconeMultitenancyIndexingPipeline(config)

        # Assertions
        mock_client.create_index.assert_not_called()
        mock_client.Index.assert_called_once_with("existing-index")
        assert pipeline._index == mock_index

    @patch("pinecone.Pinecone")
    @patch("pinecone.ServerlessSpec")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder")
    def test_connect_uses_environment_api_key(
        self,
        mock_create_embedder: MagicMock,
        mock_serverless_spec: MagicMock,
        mock_pinecone_class: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that _connect uses PINECONE_API_KEY from environment."""
        monkeypatch.setenv("PINECONE_API_KEY", "env-api-key")

        config: dict[str, Any] = {
            "pinecone": {
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.list_indexes.return_value = []
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        PineconeMultitenancyIndexingPipeline(config)

        # Assertions
        mock_pinecone_class.assert_called_once_with(api_key="env-api-key")

    @patch("pinecone.Pinecone")
    @patch("pinecone.ServerlessSpec")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder")
    def test_connect_uses_environment_index_name(
        self,
        mock_create_embedder: MagicMock,
        mock_serverless_spec: MagicMock,
        mock_pinecone_class: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that _connect uses PINECONE_INDEX from environment."""
        monkeypatch.setenv("PINECONE_INDEX", "env-index-name")

        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.list_indexes.return_value = []
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = PineconeMultitenancyIndexingPipeline(config)

        # Assertions
        mock_client.Index.assert_called_once_with("env-index-name")
        assert pipeline._get_index_name() == "env-index-name"

    @patch("pinecone.Pinecone")
    @patch("pinecone.ServerlessSpec")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.DataloaderCatalog")
    def test_load_documents_from_dataloader(
        self,
        mock_catalog_class: MagicMock,
        mock_create_embedder: MagicMock,
        mock_serverless_spec: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test document loading from dataloader registry."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "dataloader": {
                "dataset": "triviaqa",
                "params": {"limit": 10},
            },
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.list_indexes.return_value = []
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Setup new API mock pattern
        mock_docs = [
            Document(content="Test doc 1", meta={"id": "1"}),
            Document(content="Test doc 2", meta={"id": "2"}),
        ]
        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = mock_docs

        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset

        mock_catalog_class.create.return_value = mock_loader

        # Create pipeline
        pipeline = PineconeMultitenancyIndexingPipeline(config)

        # Load documents
        documents = pipeline._load_documents_from_dataloader()

        # Assertions
        mock_catalog_class.create.assert_called_once_with(
            "triviaqa",
            split="test",
            limit=10,
            dataset_id=None,
        )
        assert documents == mock_docs

    @patch("pinecone.Pinecone")
    @patch("pinecone.ServerlessSpec")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.DataloaderCatalog")
    def test_load_documents_default_dataset(
        self,
        mock_catalog_class: MagicMock,
        mock_create_embedder: MagicMock,
        mock_serverless_spec: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test document loading uses default dataset when not specified."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.list_indexes.return_value = []
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Setup new API mock pattern
        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = []

        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset

        mock_catalog_class.create.return_value = mock_loader

        # Create pipeline
        pipeline = PineconeMultitenancyIndexingPipeline(config)

        # Load documents
        pipeline._load_documents_from_dataloader()

        # Assertions - should default to triviaqa
        mock_catalog_class.create.assert_called_once_with(
            "triviaqa",
            split="test",
            limit=None,
            dataset_id=None,
        )

    @patch("pinecone.Pinecone")
    @patch("pinecone.ServerlessSpec")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder")
    def test_run_indexes_documents_success(
        self,
        mock_create_embedder: MagicMock,
        mock_serverless_spec: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test successful indexing run with provided documents."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.list_indexes.return_value = []
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        embedded_docs = [
            Document(content="Doc 1", meta={"id": "1"}, embedding=[0.1] * 384),
            Document(content="Doc 2", meta={"id": "2"}, embedding=[0.2] * 384),
        ]
        mock_embedder.run.return_value = {"documents": embedded_docs}
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = PineconeMultitenancyIndexingPipeline(config)

        # Run with documents
        documents = [
            Document(content="Doc 1", meta={"id": "1"}),
            Document(content="Doc 2", meta={"id": "2"}),
        ]
        result = pipeline.run(documents=documents)

        # Assertions
        assert result.tenant_id == "test-tenant"
        assert result.documents_indexed == 2
        assert result.collection_name == "test-index"
        mock_embedder.run.assert_called_once_with(documents=documents)
        mock_index.upsert.assert_called_once()

        # Check namespace was used
        call_kwargs = mock_index.upsert.call_args.kwargs
        assert call_kwargs["namespace"] == "test-tenant"

    @patch("pinecone.Pinecone")
    @patch("pinecone.ServerlessSpec")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder")
    def test_run_with_tenant_namespace(
        self,
        mock_create_embedder: MagicMock,
        mock_serverless_spec: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test that documents are indexed with correct tenant namespace."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "tenant-a"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.list_indexes.return_value = []
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        embedded_docs = [
            Document(content="Doc 1", meta={"id": "1"}, embedding=[0.1] * 384),
        ]
        mock_embedder.run.return_value = {"documents": embedded_docs}
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = PineconeMultitenancyIndexingPipeline(config)

        # Run with explicit tenant_id
        documents = [Document(content="Doc 1", meta={"id": "1"})]
        result = pipeline.run(documents=documents, tenant_id="tenant-b")

        # Assertions - should use explicit tenant_id
        assert result.tenant_id == "tenant-b"
        call_kwargs = mock_index.upsert.call_args.kwargs
        assert call_kwargs["namespace"] == "tenant-b"

    @patch("pinecone.Pinecone")
    @patch("pinecone.ServerlessSpec")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder")
    def test_run_with_batch_processing(
        self,
        mock_create_embedder: MagicMock,
        mock_serverless_spec: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test batch processing with more than 100 documents."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.list_indexes.return_value = []
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        # Create 150 documents to trigger batching
        embedded_docs = [
            Document(content=f"Doc {i}", meta={"id": str(i)}, embedding=[0.1] * 384)
            for i in range(150)
        ]
        mock_embedder.run.return_value = {"documents": embedded_docs}
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = PineconeMultitenancyIndexingPipeline(config)

        # Run with 150 documents
        documents = [
            Document(content=f"Doc {i}", meta={"id": str(i)}) for i in range(150)
        ]
        result = pipeline.run(documents=documents)

        # Assertions
        assert result.documents_indexed == 150
        # Should have 2 batch calls (100 + 50)
        assert mock_index.upsert.call_count == 2

    @patch("pinecone.Pinecone")
    @patch("pinecone.ServerlessSpec")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder")
    def test_run_with_empty_documents(
        self,
        mock_create_embedder: MagicMock,
        mock_serverless_spec: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test behavior when no documents are provided."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.list_indexes.return_value = []
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = PineconeMultitenancyIndexingPipeline(config)

        # Run with empty documents
        result = pipeline.run(documents=[])

        # Assertions
        assert result.documents_indexed == 0
        assert result.tenant_id == "test-tenant"
        mock_embedder.run.assert_not_called()
        mock_index.upsert.assert_not_called()

    @patch("pinecone.Pinecone")
    @patch("pinecone.ServerlessSpec")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.DataloaderCatalog")
    def test_run_loads_documents_from_dataloader(
        self,
        mock_catalog_class: MagicMock,
        mock_create_embedder: MagicMock,
        mock_serverless_spec: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test that run() loads documents from dataloader when not provided."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "dataloader": {"dataset": "triviaqa"},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.list_indexes.return_value = []
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Setup new API mock pattern
        mock_docs = [
            Document(content="Loaded doc 1", meta={"id": "1"}),
            Document(content="Loaded doc 2", meta={"id": "2"}),
        ]
        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = mock_docs

        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset

        mock_catalog_class.create.return_value = mock_loader

        # Setup embedder to return embedded versions
        embedded_docs = [
            Document(content="Loaded doc 1", meta={"id": "1"}, embedding=[0.1] * 384),
            Document(content="Loaded doc 2", meta={"id": "2"}, embedding=[0.2] * 384),
        ]
        mock_embedder.run.return_value = {"documents": embedded_docs}

        # Create pipeline
        pipeline = PineconeMultitenancyIndexingPipeline(config)

        # Run without providing documents
        result = pipeline.run()

        # Assertions
        assert result.documents_indexed == 2
        mock_catalog_class.create.assert_called_once_with(
            "triviaqa",
            split="test",
            limit=None,
            dataset_id=None,
        )
        mock_embedder.run.assert_called_once()

    @patch("pinecone.Pinecone")
    @patch("pinecone.ServerlessSpec")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder")
    def test_run_with_embedding_truncation(
        self,
        mock_create_embedder: MagicMock,
        mock_serverless_spec: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test that embeddings are truncated when output_dimension is specified."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {
                "dimension": 384,
                "output_dimension": 256,  # Truncate to 256
            },
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.list_indexes.return_value = []
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        # Return documents with 384-dim embeddings
        embedded_docs = [
            Document(content="Doc 1", meta={"id": "1"}, embedding=[0.1] * 384),
        ]
        mock_embedder.run.return_value = {"documents": embedded_docs}
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = PineconeMultitenancyIndexingPipeline(config)

        # Run
        documents = [Document(content="Doc 1", meta={"id": "1"})]
        pipeline.run(documents=documents)

        # Assertions - check that vectors were upserted with truncated embeddings
        call_args = mock_index.upsert.call_args.kwargs["vectors"]
        assert len(call_args[0][1]) == 256  # embedding should be truncated to 256

    @patch("pinecone.Pinecone")
    @patch("pinecone.ServerlessSpec")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder")
    def test_create_timing_metrics(
        self,
        mock_create_embedder: MagicMock,
        mock_serverless_spec: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test timing metrics creation."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.list_indexes.return_value = []
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = PineconeMultitenancyIndexingPipeline(config)

        # Create timing metrics
        metrics = pipeline._create_timing_metrics(num_documents=100, total_ms=1500.5)

        # Assertions
        assert metrics.tenant_id == "test-tenant"
        assert metrics.num_documents == 100
        assert metrics.total_ms == 1500.5
        assert metrics.index_operation_ms == 1500.5
        assert metrics.tenant_resolution_ms == 0.0
        assert metrics.retrieval_ms == 0.0

    @patch("pinecone.Pinecone")
    def test_error_handling_on_connection_failure(
        self,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test error handling when Pinecone connection fails."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "invalid-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mock to raise exception
        mock_pinecone_class.side_effect = Exception("Connection failed")

        # Attempt to create pipeline should raise exception
        with pytest.raises(Exception, match="Connection failed"):
            PineconeMultitenancyIndexingPipeline(config)

    @patch("pinecone.Pinecone")
    @patch("pinecone.ServerlessSpec")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder")
    def test_run_with_database_config_fallback(
        self,
        mock_create_embedder: MagicMock,
        mock_serverless_spec: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test pipeline falls back to 'database' config if 'pinecone' not present."""
        config: dict[str, Any] = {
            "database": {  # Using 'database' instead of 'pinecone'
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.list_indexes.return_value = []
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        embedded_docs = [
            Document(content="Doc 1", meta={"id": "1"}, embedding=[0.1] * 384),
        ]
        mock_embedder.run.return_value = {"documents": embedded_docs}
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = PineconeMultitenancyIndexingPipeline(config)

        # Assertions
        assert pipeline._get_index_name() == "test-index"

        # Run to verify everything works
        result = pipeline.run(documents=[Document(content="Doc 1", meta={"id": "1"})])
        assert result.documents_indexed == 1

    @patch("pinecone.Pinecone")
    @patch("pinecone.ServerlessSpec")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder")
    def test_run_default_index_name(
        self,
        mock_create_embedder: MagicMock,
        mock_serverless_spec: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test that default index name 'multitenancy' is used when not specified."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.list_indexes.return_value = []
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = PineconeMultitenancyIndexingPipeline(config)

        # Assertions - should use default "multitenancy"
        assert pipeline._get_index_name() == "multitenancy"
        mock_client.Index.assert_called_with("multitenancy")

    @patch("pinecone.Pinecone")
    @patch("pinecone.ServerlessSpec")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder")
    def test_document_metadata_includes_tenant_id(
        self,
        mock_create_embedder: MagicMock,
        mock_serverless_spec: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test that document metadata includes tenant_id when upserting."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.list_indexes.return_value = []
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        embedded_docs = [
            Document(
                content="Test content",
                meta={"source": "test", "category": "A"},
                embedding=[0.1] * 384,
            ),
        ]
        mock_embedder.run.return_value = {"documents": embedded_docs}
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = PineconeMultitenancyIndexingPipeline(config)

        # Run
        documents = [
            Document(content="Test content", meta={"source": "test", "category": "A"})
        ]
        pipeline.run(documents=documents)

        # Check metadata in upsert call
        vectors = mock_index.upsert.call_args.kwargs["vectors"]
        doc_id, embedding, metadata = vectors[0]

        assert metadata["tenant_id"] == "test-tenant"
        assert metadata["content"] == "Test content"
        assert metadata["source"] == "test"
        assert metadata["category"] == "A"

    @patch("pinecone.Pinecone")
    @patch("pinecone.ServerlessSpec")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder")
    def test_run_with_documents_without_embeddings(
        self,
        mock_create_embedder: MagicMock,
        mock_serverless_spec: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test that documents without embeddings are skipped."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.list_indexes.return_value = []
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        # Return a document without embedding
        embedded_docs = [
            Document(content="Doc 1", meta={"id": "1"}),  # No embedding
        ]
        mock_embedder.run.return_value = {"documents": embedded_docs}
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = PineconeMultitenancyIndexingPipeline(config)

        # Run with documents that will have no embeddings
        documents = [Document(content="Doc 1", meta={"id": "1"})]
        result = pipeline.run(documents=documents)

        # Assertions - should skip documents without embeddings, resulting in 0 vectors
        assert result.documents_indexed == 1  # Embedded count still counts the document
        mock_index.upsert.assert_not_called()  # No vectors to upsert

    @patch("pinecone.Pinecone")
    @patch("pinecone.ServerlessSpec")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder")
    def test_run_with_mixed_embeddings(
        self,
        mock_create_embedder: MagicMock,
        mock_serverless_spec: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test that only documents with embeddings are indexed."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.list_indexes.return_value = []
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        # Return mixed - some with embeddings, some without
        embedded_docs = [
            Document(content="Doc 1", meta={"id": "1"}, embedding=[0.1] * 384),
            Document(content="Doc 2", meta={"id": "2"}),  # No embedding
            Document(content="Doc 3", meta={"id": "3"}, embedding=[0.3] * 384),
        ]
        mock_embedder.run.return_value = {"documents": embedded_docs}
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = PineconeMultitenancyIndexingPipeline(config)

        # Run
        documents = [
            Document(content="Doc 1", meta={"id": "1"}),
            Document(content="Doc 2", meta={"id": "2"}),
            Document(content="Doc 3", meta={"id": "3"}),
        ]
        result = pipeline.run(documents=documents)

        # Assertions - only 2 documents should be indexed (those with embeddings)
        assert result.documents_indexed == 3  # Total embedded docs
        # Check that upsert was called with only 2 vectors
        vectors = mock_index.upsert.call_args.kwargs["vectors"]
        assert len(vectors) == 2  # Only 2 documents had embeddings

    @patch("pinecone.Pinecone")
    @patch("pinecone.ServerlessSpec")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder")
    def test_close_method(
        self,
        mock_create_embedder: MagicMock,
        mock_serverless_spec: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test that close method is callable and does not raise."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.list_indexes.return_value = []
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline and close
        pipeline = PineconeMultitenancyIndexingPipeline(config)
        pipeline.close()  # Should not raise

    @patch("pinecone.Pinecone")
    @patch("pinecone.ServerlessSpec")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder")
    def test_run_with_single_document_batch(
        self,
        mock_create_embedder: MagicMock,
        mock_serverless_spec: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test batch processing with exactly 1 document (single batch)."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.list_indexes.return_value = []
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        embedded_docs = [
            Document(content="Single doc", meta={"id": "1"}, embedding=[0.1] * 384)
        ]
        mock_embedder.run.return_value = {"documents": embedded_docs}
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = PineconeMultitenancyIndexingPipeline(config)

        # Run with 1 document
        documents = [Document(content="Single doc", meta={"id": "1"})]
        result = pipeline.run(documents=documents)

        # Assertions - should have 1 upsert call (single batch)
        assert result.documents_indexed == 1
        assert mock_index.upsert.call_count == 1

    @patch("pinecone.Pinecone")
    @patch("pinecone.ServerlessSpec")
    @patch("vectordb.haystack.multi_tenancy.pinecone.indexing.create_document_embedder")
    def test_embedder_warm_up_called(
        self,
        mock_create_embedder: MagicMock,
        mock_serverless_spec: MagicMock,
        mock_pinecone_class: MagicMock,
    ) -> None:
        """Test that embedder warm_up is called during initialization."""
        config: dict[str, Any] = {
            "pinecone": {
                "api_key": "test-key",
                "index": "test-index",
                "dimension": 384,
            },
            "embedding": {"dimension": 384},
            "tenant": {"id": "test-tenant"},
        }

        # Setup mocks
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.list_indexes.return_value = []
        mock_client.Index.return_value = mock_index
        mock_pinecone_class.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        PineconeMultitenancyIndexingPipeline(config)

        # Assertions - warm_up should be called
        mock_embedder.warm_up.assert_called_once()
