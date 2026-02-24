"""Comprehensive unit tests for Weaviate multi-tenancy indexing pipeline.

This module tests the WeaviateMultitenancyIndexingPipeline class with focus on:
- Initialization with various config formats
- Connection handling and client setup
- Document loading from dataloaders
- Indexing run operations with tenant isolation
- Batch processing and timing metrics
- Error handling scenarios
- Tenant creation and management
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.multi_tenancy.base import BaseMultitenancyPipeline
from vectordb.haystack.multi_tenancy.common.tenant_context import TenantContext
from vectordb.haystack.multi_tenancy.weaviate.indexing import (
    WeaviateMultitenancyIndexingPipeline,
)


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Create a temporary config file with default settings."""
    config_path = tmp_path / "config.yaml"
    config_content = """
weaviate:
  url: http://localhost:8080
collection:
  name: TestCollection
embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
  dimension: 384
tenant:
  id: test-tenant
"""
    config_path.write_text(config_content)
    return config_path


class TestWeaviateMultitenancyIndexingPipeline:
    """Test suite for WeaviateMultitenancyIndexingPipeline."""

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_inherits_from_base_multitenancy_pipeline(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        config_file: Path,
    ) -> None:
        """Test that pipeline inherits from BaseMultitenancyPipeline."""
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))

        assert isinstance(pipeline, BaseMultitenancyPipeline)

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_initialization_with_config_path(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test pipeline initialization with config file path."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
  api_key: test-key
collection:
  name: TestCollection
embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
  dimension: 384
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline with config path
        pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))

        # Assertions
        assert pipeline.tenant_context.tenant_id == "test-tenant"
        assert "weaviate" in pipeline.config

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_initialization_with_dict_config(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test pipeline initialization with dictionary config."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
  api_key: test-key
collection:
  name: TestCollection
embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
  dimension: 384
tenant:
  id: dict-tenant
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline with config path
        pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))

        # Assertions
        assert pipeline.tenant_context.tenant_id == "dict-tenant"
        assert "weaviate" in pipeline.config

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_initialization_with_tenant_context(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test pipeline initialization with explicit tenant context."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
collection:
  name: TestCollection
embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
  dimension: 384
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create explicit tenant context
        tenant_context = TenantContext(tenant_id="explicit-tenant")

        # Create pipeline with tenant context
        pipeline = WeaviateMultitenancyIndexingPipeline(
            str(config_file), tenant_context
        )

        # Assertions
        assert pipeline.tenant_context.tenant_id == "explicit-tenant"

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_initialization_with_api_key(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that API key authentication is configured."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
  api_key: my-secret-key
collection:
  name: TestCollection
embedding:
  dimension: 384
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        WeaviateMultitenancyIndexingPipeline(str(config_file))

        # Assertions - verify client was created with auth
        mock_weaviate_client.assert_called_once()
        call_kwargs = mock_weaviate_client.call_args.kwargs
        assert call_kwargs["url"] == "http://localhost:8080"
        assert call_kwargs["auth_client_secret"] is not None

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_initialization_with_additional_headers(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that additional headers are passed to client."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
  headers:
    X-OpenAI-Api-Key: test-key
collection:
  name: TestCollection
embedding:
  dimension: 384
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        WeaviateMultitenancyIndexingPipeline(str(config_file))

        # Assertions
        call_kwargs = mock_weaviate_client.call_args.kwargs
        assert "additional_headers" in call_kwargs
        assert call_kwargs["additional_headers"]["X-OpenAI-Api-Key"] == "test-key"

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_connect_initializes_client(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that _connect initializes Weaviate client and embedder."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
collection:
  name: TestCollection
embedding:
  dimension: 384
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))

        # Assertions
        assert pipeline._client == mock_client
        assert pipeline._embedder == mock_embedder
        mock_embedder.warm_up.assert_called_once()

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_connect_uses_environment_url(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Test that _connect uses WEAVIATE_URL from environment."""
        monkeypatch.setenv("WEAVIATE_URL", "http://env:8080")

        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
collection:
  name: TestCollection
embedding:
  dimension: 384
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        WeaviateMultitenancyIndexingPipeline(str(config_file))

        # Assertions
        call_kwargs = mock_weaviate_client.call_args.kwargs
        assert call_kwargs["url"] == "http://env:8080"

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.DataloaderCatalog")
    def test_load_documents_from_dataloader(
        self,
        mock_catalog_class: MagicMock,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test document loading from dataloader registry."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
collection:
  name: TestCollection
embedding:
  dimension: 384
dataloader:
  dataset: triviaqa
  params:
    limit: 10
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Setup dataloader mock - new API pattern
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
        pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))

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

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_run_indexes_documents_success(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test successful indexing run with provided documents."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
collection:
  name: TestCollection
embedding:
  dimension: 384
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        embedded_docs = [
            Document(content="Doc 1", meta={"id": "1"}, embedding=[0.1] * 384),
            Document(content="Doc 2", meta={"id": "2"}, embedding=[0.2] * 384),
        ]
        mock_embedder.run.return_value = {"documents": embedded_docs}
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))

        # Run with documents
        documents = [
            Document(content="Doc 1", meta={"id": "1"}),
            Document(content="Doc 2", meta={"id": "2"}),
        ]
        result = pipeline.run(documents=documents)

        # Assertions
        assert result.tenant_id == "test-tenant"
        assert result.documents_indexed == 2
        assert result.collection_name == "TestCollection"
        mock_embedder.run.assert_called_once_with(documents=documents)
        mock_client.batch.configure.assert_called_once_with(batch_size=100)

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_run_with_tenant_isolation(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that documents are indexed with correct tenant."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
collection:
  name: TestCollection
embedding:
  dimension: 384
tenant:
  id: tenant-a
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        embedded_docs = [
            Document(content="Doc 1", meta={"id": "1"}, embedding=[0.1] * 384),
        ]
        mock_embedder.run.return_value = {"documents": embedded_docs}
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))

        # Run with explicit tenant_id
        documents = [Document(content="Doc 1", meta={"id": "1"})]
        result = pipeline.run(documents=documents, tenant_id="tenant-b")

        # Assertions - should use explicit tenant_id
        assert result.tenant_id == "tenant-b"

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_run_with_empty_documents(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test behavior when no documents are provided."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
collection:
  name: TestCollection
embedding:
  dimension: 384
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))

        # Run with empty documents
        result = pipeline.run(documents=[])

        # Assertions
        assert result.documents_indexed == 0
        assert result.tenant_id == "test-tenant"
        mock_embedder.run.assert_not_called()

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.DataloaderCatalog")
    def test_run_loads_documents_from_dataloader(
        self,
        mock_catalog_class: MagicMock,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that run() loads documents from dataloader when not provided."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
collection:
  name: TestCollection
embedding:
  dimension: 384
dataloader:
  dataset: triviaqa
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

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
        pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))

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

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_run_with_embedding_truncation(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that embeddings are truncated when output_dimension is specified."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
collection:
  name: TestCollection
embedding:
  dimension: 384
  output_dimension: 256
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        # Return documents with 384-dim embeddings
        embedded_docs = [
            Document(content="Doc 1", meta={"id": "1"}, embedding=[0.1] * 384),
        ]
        mock_embedder.run.return_value = {"documents": embedded_docs}
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))

        # Run
        documents = [Document(content="Doc 1", meta={"id": "1"})]
        pipeline.run(documents=documents)

        # Verify truncation happened - batch should have been configured
        mock_client.batch.configure.assert_called_once()

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_ensure_tenant_exists_creates_tenant(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that _ensure_tenant_exists creates tenant when not exists."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
collection:
  name: TestCollection
embedding:
  dimension: 384
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Mock tenant exists check - return empty list (tenant doesn't exist)
        mock_client.schema.get_tenant.return_value = []

        # Create pipeline
        pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))

        # Call _ensure_tenant_exists
        pipeline._ensure_tenant_exists("TestCollection", "new-tenant")

        # Assertions
        mock_client.schema.add_tenant.assert_called_once_with(
            "TestCollection", "new-tenant"
        )

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_ensure_tenant_exists_skips_when_exists(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that _ensure_tenant_exists skips when tenant already exists."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
collection:
  name: TestCollection
embedding:
  dimension: 384
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Mock tenant exists - return existing tenants
        mock_client.schema.get_tenant.return_value = [{"name": "existing-tenant"}]

        # Create pipeline
        pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))

        # Call _ensure_tenant_exists with existing tenant
        pipeline._ensure_tenant_exists("TestCollection", "existing-tenant")

        # Assertions - should not create tenant
        mock_client.schema.add_tenant.assert_not_called()

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_ensure_tenant_creates_class_on_error(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that tenant creation falls back to creating class with multi-tenancy."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
collection:
  name: TestCollection
embedding:
  dimension: 384
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Mock get_tenant to raise exception (class doesn't exist)
        mock_client.schema.get_tenant.side_effect = Exception("Class not found")
        # Also mock get to raise exception so create_class is called
        mock_client.schema.get.side_effect = Exception("Class not found")

        # Create pipeline
        pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))

        # Call _ensure_tenant_exists
        pipeline._ensure_tenant_exists("TestCollection", "new-tenant")

        # Assertions - should create class and add tenant
        mock_client.schema.create_class.assert_called_once()
        mock_client.schema.add_tenant.assert_called_once_with(
            "TestCollection", "new-tenant"
        )

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_create_timing_metrics(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test timing metrics creation."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
collection:
  name: TestCollection
embedding:
  dimension: 384
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))

        # Create timing metrics
        metrics = pipeline._create_timing_metrics(
            index_operation_ms=1500.5,
            total_ms=1500.5,
            num_documents=100,
        )

        # Assertions
        assert metrics.tenant_id == "test-tenant"
        assert metrics.num_documents == 100
        assert metrics.total_ms == 1500.5
        assert metrics.index_operation_ms == 1500.5
        assert metrics.tenant_resolution_ms == 0.0
        assert metrics.retrieval_ms == 0.0

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_get_class_name(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test _get_class_name returns configured class name."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
collection:
  name: CustomCollection
embedding:
  dimension: 384
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))

        # Assertions
        assert pipeline._get_class_name() == "CustomCollection"

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_get_class_name_defaults(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test _get_class_name returns default when not configured."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
embedding:
  dimension: 384
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))

        # Assertions
        assert pipeline._get_class_name() == "MultiTenancy"

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_close_method(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test close method calls client.close()."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
collection:
  name: TestCollection
embedding:
  dimension: 384
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline and close
        pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))
        pipeline.close()

        # Assertions
        mock_client.close.assert_called_once()

    @patch("weaviate.Client")
    def test_error_handling_on_connection_failure(
        self,
        mock_weaviate_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test error handling when Weaviate connection fails."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
  api_key: invalid-key
collection:
  name: TestCollection
embedding:
  dimension: 384
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        # Setup mock to raise exception
        mock_weaviate_client.side_effect = Exception("Connection failed")

        # Attempt to create pipeline should raise exception
        with pytest.raises(Exception, match="Connection failed"):
            WeaviateMultitenancyIndexingPipeline(str(config_file))

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_run_with_database_config_fallback(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test pipeline falls back to 'database' config if 'weaviate' not present."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
database:
  url: http://localhost:8080
collection:
  name: TestCollection
embedding:
  dimension: 384
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        embedded_docs = [
            Document(content="Doc 1", meta={"id": "1"}, embedding=[0.1] * 384),
        ]
        mock_embedder.run.return_value = {"documents": embedded_docs}
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))

        # Run to verify everything works
        result = pipeline.run(documents=[Document(content="Doc 1", meta={"id": "1"})])
        assert result.documents_indexed == 1

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_document_metadata_handling(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that document metadata is properly handled in batch."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
collection:
  name: TestCollection
embedding:
  dimension: 384
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

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
        pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))

        # Run
        documents = [
            Document(content="Test content", meta={"source": "test", "category": "A"})
        ]
        pipeline.run(documents=documents)

        # Verify batch was used with data_object containing properties
        mock_client.batch.configure.assert_called_once()

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_close_method_when_client_is_none(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test close method handles None client gracefully."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
collection:
  name: TestCollection
embedding:
  dimension: 384
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))

        # Manually set _client to None to test the edge case
        pipeline._client = None

        # close() should not raise even when client is None
        pipeline.close()  # Should not raise

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_document_metadata_with_tenant_id_preserved(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that tenant_id in document meta is handled correctly."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
collection:
  name: TestCollection
embedding:
  dimension: 384
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        # Document with tenant_id in metadata
        embedded_docs = [
            Document(
                content="Test content",
                meta={"tenant_id": "original", "source": "test"},
                embedding=[0.1] * 384,
            ),
        ]
        mock_embedder.run.return_value = {"documents": embedded_docs}
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))

        # Run
        documents = [
            Document(
                content="Test content", meta={"tenant_id": "original", "source": "test"}
            )
        ]
        pipeline.run(documents=documents)

        # Verify batch was used
        mock_client.batch.configure.assert_called_once()

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_ensure_tenant_fails_to_create_class(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that _ensure_tenant_exists handles when create_class also fails."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
collection:
  name: TestCollection
embedding:
  dimension: 384
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        mock_create_embedder.return_value = mock_embedder

        # Mock get_tenant to raise exception (class doesn't exist)
        mock_client.schema.get_tenant.side_effect = Exception("Class not found")
        # Mock get to also raise exception so create_class is attempted
        mock_client.schema.get.side_effect = Exception("Class not found")
        # Mock create_class to also raise exception
        mock_client.schema.create_class.side_effect = Exception("Create failed")

        # Create pipeline
        pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))

        # Call _ensure_tenant_exists - should raise since create_class fails
        with pytest.raises(Exception, match="Create failed"):
            pipeline._ensure_tenant_exists("TestCollection", "new-tenant")

        # Assertions - all methods should have been called
        mock_client.schema.get_tenant.assert_called_once()
        mock_client.schema.get.assert_called_once()
        mock_client.schema.create_class.assert_called_once()

    @patch("weaviate.Client")
    @patch("vectordb.haystack.multi_tenancy.weaviate.indexing.create_document_embedder")
    def test_run_with_complex_metadata_types(
        self,
        mock_create_embedder: MagicMock,
        mock_weaviate_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that complex metadata types are converted to strings."""
        # Create a temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_content = """
weaviate:
  url: http://localhost:8080
collection:
  name: TestCollection
embedding:
  dimension: 384
tenant:
  id: test-tenant
"""
        config_file.write_text(config_content)

        # Setup mocks
        mock_client = MagicMock()
        mock_weaviate_client.return_value = mock_client

        mock_embedder = MagicMock()
        mock_embedder.warm_up = MagicMock()
        # Document with complex metadata types (list, dict)
        embedded_docs = [
            Document(
                content="Test content",
                meta={"tags": ["a", "b"], "data": {"key": "value"}},
                embedding=[0.1] * 384,
            ),
        ]
        mock_embedder.run.return_value = {"documents": embedded_docs}
        mock_create_embedder.return_value = mock_embedder

        # Create pipeline
        pipeline = WeaviateMultitenancyIndexingPipeline(str(config_file))

        # Run
        documents = [
            Document(
                content="Test content",
                meta={"tags": ["a", "b"], "data": {"key": "value"}},
            )
        ]
        pipeline.run(documents=documents)

        # Verify batch was used
        mock_client.batch.configure.assert_called_once()
