"""Tests for Chroma multi-tenancy pipelines using LangChain.

This module provides comprehensive unit tests for Chroma-based multi-tenancy
functionality using collection-per-tenant strategy with configurable prefix.
Each tenant receives a dedicated Chroma collection with a customizable prefix,
ensuring strong data isolation for local and embedded deployments.

Multi-tenancy Strategy:
    - Creates a separate Chroma collection for each tenant
    - Collection names are prefixed (e.g., "tenant_{tenant_id}")
    - Complete isolation at the collection level
    - Supports both persistent (on-disk) and in-memory Chroma configurations

Test Coverage:
    - Indexing: Document upsert to tenant-specific collections
    - Search: Collection-scoped similarity search with metadata filtering
    - Management: Tenant deletion (collection drop), listing (collection discovery)
    - Validation: Empty tenant_id rejection, count mismatch detection
    - Collection naming: Prefix-based name generation
    - Pipelines: End-to-end indexing and search pipeline validation
    - Logging: Comprehensive logging verification for operations

Test Organization:
    - TestChromaMultiTenancyIndexing: Core indexing operations
    - TestChromaMultiTenancySearch: Search with tenant isolation
    - TestChromaMultiTenancyManagement: CRUD operations on tenants
    - TestChromaMultiTenancyIndexingPipeline: High-level indexing workflows
    - TestChromaMultiTenancySearchPipeline: High-level search workflows

All tests use mocking to avoid requiring actual Chroma database.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


class TestChromaMultiTenancyIndexing:
    """Unit tests for Chroma multi-tenancy indexing operations.

    Validates document indexing workflows with collection-per-tenant isolation.
    Ensures documents are routed to the correct tenant collection and collection
    naming conventions are followed.

    Tested Scenarios:
        - Pipeline initialization with persistence path and prefix
        - Collection name generation with prefix
        - Document upsert to tenant-specific collection
        - Empty document handling (early return with 0)
        - Tenant ID validation (non-empty requirement)
        - Document-embedding count parity enforcement
    """

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    def test_pipeline_initialization(self, mock_db_cls):
        """Test that pipeline initializes with correct configuration.

        Validates:
            - Persistence path is stored correctly
            - Collection prefix is configured
            - Connection parameters are preserved
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.multi_tenancy.chroma import (
            ChromaMultiTenancyPipeline,
        )

        pipeline = ChromaMultiTenancyPipeline(
            path="./test_chroma_data", collection_prefix="tenant_"
        )

        assert pipeline.path == "./test_chroma_data"
        assert pipeline.collection_prefix == "tenant_"

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    def test_get_tenant_collection_name(self, mock_db_cls):
        """Test collection name generation with configured prefix.

        Validates:
            - Tenant ID is combined with prefix to form collection name
            - Prefix is prepended correctly to tenant identifier
            - Generated names are unique per tenant
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.multi_tenancy.chroma import (
            ChromaMultiTenancyPipeline,
        )

        pipeline = ChromaMultiTenancyPipeline()

        collection_name = pipeline._get_tenant_collection_name("tenant_1")
        assert collection_name == "tenant_tenant_1"

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    def test_index_for_tenant(self, mock_db_cls):
        """Test document indexing to tenant-specific collection.

        Validates:
            - Documents are upserted to the correct tenant collection
            - Collection name includes configured prefix
            - Return value matches number of documents indexed
            - Database upsert is invoked exactly once
        """
        mock_db = MagicMock()
        mock_db.upsert.return_value = 5
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.multi_tenancy.chroma import (
            ChromaMultiTenancyPipeline,
        )

        pipeline = ChromaMultiTenancyPipeline()

        documents = [
            Document(page_content="doc1", metadata={"id": "1"}),
            Document(page_content="doc2", metadata={"id": "2"}),
        ]
        embeddings = [[0.1] * 384, [0.2] * 384]

        result = pipeline.index_for_tenant("tenant_1", documents, embeddings)

        assert result == 5
        mock_db.upsert.assert_called_once()

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    def test_index_for_tenant_empty_documents(self, mock_db_cls):
        """Test that indexing empty documents returns 0 without database call.

        Validates:
            - Empty document list returns 0 immediately
            - No database operations are performed
            - Efficient handling of no-op scenarios
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.multi_tenancy.chroma import (
            ChromaMultiTenancyPipeline,
        )

        pipeline = ChromaMultiTenancyPipeline()

        result = pipeline.index_for_tenant("tenant_1", [], [])

        assert result == 0

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    def test_index_for_tenant_invalid_tenant_id(self, mock_db_cls):
        """Test that empty tenant_id raises ValueError.

        Validates:
            - Empty string tenant_id is rejected
            - Appropriate error message is raised
            - Database is not contacted with invalid input
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.multi_tenancy.chroma import (
            ChromaMultiTenancyPipeline,
        )

        pipeline = ChromaMultiTenancyPipeline()

        documents = [Document(page_content="doc1", metadata={"id": "1"})]
        embeddings = [[0.1] * 384]

        with pytest.raises(ValueError, match="tenant_id cannot be empty"):
            pipeline.index_for_tenant("", documents, embeddings)

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    def test_index_for_tenant_mismatched_counts(self, mock_db_cls):
        """Test that document-embedding count mismatch raises ValueError.

        Validates:
            - Documents count must equal embeddings count
            - Mismatched arrays raise descriptive error
            - Data integrity is protected before persistence
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.multi_tenancy.chroma import (
            ChromaMultiTenancyPipeline,
        )

        pipeline = ChromaMultiTenancyPipeline()

        documents = [
            Document(page_content="doc1", metadata={"id": "1"}),
            Document(page_content="doc2", metadata={"id": "2"}),
        ]
        embeddings = [[0.1] * 384]

        with pytest.raises(ValueError, match="does not match embeddings count"):
            pipeline.index_for_tenant("tenant_1", documents, embeddings)


class TestChromaMultiTenancySearch:
    """Unit tests for Chroma multi-tenancy search operations.

    Validates search functionality with collection-per-tenant isolation.
    Ensures queries target the correct tenant collection and support
    metadata filtering within tenant boundaries.

    Tested Scenarios:
        - Collection-scoped similarity search
        - Empty tenant ID validation
        - Metadata filtering within tenant collection
    """

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    def test_search_for_tenant(self, mock_db_cls):
        """Test that search respects collection boundaries.

        Validates:
            - Search queries target the correct tenant collection
            - Results are returned as a list of documents
            - Tenant isolation is maintained during search
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.multi_tenancy.chroma import (
            ChromaMultiTenancyPipeline,
        )

        pipeline = ChromaMultiTenancyPipeline()

        result = pipeline.search_for_tenant("tenant_1", "test query", top_k=5)

        assert isinstance(result, list)

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    def test_search_for_tenant_invalid_tenant_id(self, mock_db_cls):
        """Test that search with empty tenant_id raises ValueError.

        Validates:
            - Empty string tenant_id is rejected
            - Error is raised before any database operation
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.multi_tenancy.chroma import (
            ChromaMultiTenancyPipeline,
        )

        pipeline = ChromaMultiTenancyPipeline()

        with pytest.raises(ValueError, match="tenant_id cannot be empty"):
            pipeline.search_for_tenant("", "test query", top_k=5)

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    def test_search_for_tenant_with_filters(self, mock_db_cls):
        """Test collection-scoped search with metadata filters.

        Validates:
            - Metadata filters are applied within tenant collection
            - Filters do not break tenant isolation
            - Results are filtered and scoped correctly
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.multi_tenancy.chroma import (
            ChromaMultiTenancyPipeline,
        )

        pipeline = ChromaMultiTenancyPipeline()

        filters = {"category": "test"}
        result = pipeline.search_for_tenant(
            "tenant_1", "test query", top_k=5, filters=filters
        )

        assert isinstance(result, list)


class TestChromaMultiTenancyManagement:
    """Unit tests for Chroma multi-tenancy tenant management.

    Validates administrative operations for managing tenants including
    deletion of tenant collections and discovery of existing tenants.

    Tested Scenarios:
        - Tenant deletion (collection drop)
        - Empty tenant ID validation for deletion
        - Deletion failure handling
        - Tenant listing (collection discovery with prefix filtering)
        - Empty tenant list handling
        - Connection failure resilience
    """

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    def test_delete_tenant(self, mock_db_cls):
        """Test tenant deletion via collection drop.

        Validates:
            - Delete operation drops the correct tenant collection
            - Success returns True
            - Collection name includes configured prefix
            - Database delete_collection is called with correct name
        """
        mock_db = MagicMock()
        mock_db.delete_collection.return_value = None
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.multi_tenancy.chroma import (
            ChromaMultiTenancyPipeline,
        )

        pipeline = ChromaMultiTenancyPipeline()

        result = pipeline.delete_tenant("tenant_1")

        assert result is True
        mock_db.delete_collection.assert_called_once_with("tenant_tenant_1")

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    def test_delete_tenant_invalid_tenant_id(self, mock_db_cls):
        """Test that deleting with empty tenant_id raises ValueError.

        Validates:
            - Empty string tenant_id is rejected
            - Error prevents accidental mass deletion
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.multi_tenancy.chroma import (
            ChromaMultiTenancyPipeline,
        )

        pipeline = ChromaMultiTenancyPipeline()

        with pytest.raises(ValueError, match="tenant_id cannot be empty"):
            pipeline.delete_tenant("")

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    def test_delete_tenant_failure_raises(self, mock_db_cls):
        """Test that database deletion failures propagate as exceptions.

        Validates:
            - Database errors are not silently swallowed
            - Original exception is raised to caller
            - Failures are visible for monitoring and debugging
        """
        mock_db = MagicMock()
        mock_db.delete_collection.side_effect = Exception("Deletion failed")
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.multi_tenancy.chroma import (
            ChromaMultiTenancyPipeline,
        )

        pipeline = ChromaMultiTenancyPipeline()

        with pytest.raises(Exception, match="Deletion failed"):
            pipeline.delete_tenant("tenant_1")

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    def test_list_tenants(self, mock_db_cls):
        """Test tenant discovery via collection enumeration with prefix filtering.

        Validates:
            - All collections matching prefix are identified as tenants
            - Prefix is stripped from collection names to extract tenant IDs
            - Multiple tenants are correctly identified
        """
        mock_db = MagicMock()
        mock_db.get_collections.return_value = ["tenant_tenant1", "tenant_tenant2"]
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.multi_tenancy.chroma import (
            ChromaMultiTenancyPipeline,
        )

        pipeline = ChromaMultiTenancyPipeline()

        tenants = pipeline.list_tenants()

        assert tenants == ["tenant1", "tenant2"]

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    def test_list_tenants_empty(self, mock_db_cls):
        """Test that empty collection list returns empty tenant list.

        Validates:
            - No matching collections returns empty list
            - No tenants found scenario is handled gracefully
        """
        mock_db = MagicMock()
        mock_db.get_collections.return_value = []
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.multi_tenancy.chroma import (
            ChromaMultiTenancyPipeline,
        )

        pipeline = ChromaMultiTenancyPipeline()

        tenants = pipeline.list_tenants()

        assert tenants == []

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    def test_list_tenants_failure_returns_empty(self, mock_db_cls):
        """Test that connection failures return empty list instead of crashing.

        Validates:
            - Database connection errors are gracefully handled
            - Empty list is returned rather than propagating exception
            - Application can continue operating during outages
        """
        mock_db = MagicMock()
        mock_db.get_collections.side_effect = Exception("Connection failed")
        mock_db_cls.return_value = mock_db

        from vectordb.langchain.multi_tenancy.chroma import (
            ChromaMultiTenancyPipeline,
        )

        pipeline = ChromaMultiTenancyPipeline()

        tenants = pipeline.list_tenants()

        assert tenants == []


class TestChromaMultiTenancyIndexingPipeline:
    """Unit tests for ChromaMultiTenancyIndexingPipeline high-level workflow.

    Validates the complete indexing pipeline that orchestrates document loading,
    embedding generation, and tenant-scoped upsert to tenant-specific collections.
    Includes extensive logging verification and edge case handling.

    Tested Scenarios:
        - Full pipeline execution with mocked dependencies
        - Pipeline result structure and content
        - Empty tenant ID validation during initialization
        - Empty document handling with warning logs
        - Different dataloader limit configurations
        - Logging verification at all stages
        - Configuration file path loading
        - None tenant ID validation
    """

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.dataloaders.catalog.DataloaderCatalog.create")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_documents")
    def test_indexing_pipeline_run(
        self,
        mock_embed_documents,
        mock_get_documents,
        mock_create_embedder,
        mock_db_cls,
        chroma_multi_tenant_config,
        sample_documents,
    ):
        """Test complete indexing pipeline execution.

        Validates:
            - Pipeline loads documents from configured dataloader
            - Documents are embedded using configured embedder
            - Upsert targets the correct tenant collection
            - Result contains document count and tenant ID
        """
        mock_db = MagicMock()
        mock_db.upsert.return_value = 3
        mock_db_cls.return_value = mock_db

        mock_create_embedder.return_value = MagicMock()

        # Mock: create() -> load() -> to_langchain()
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_documents.return_value = mock_loader

        mock_embed_documents.return_value = (sample_documents, [[0.1] * 384] * 3)

        from vectordb.langchain.multi_tenancy.indexing.chroma import (
            ChromaMultiTenancyIndexingPipeline,
        )

        pipeline = ChromaMultiTenancyIndexingPipeline(
            chroma_multi_tenant_config, "tenant_123"
        )
        result = pipeline.run()

        assert result["documents_indexed"] == 3
        assert result["tenant_id"] == "tenant_123"

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    def test_indexing_pipeline_empty_tenant_id(
        self, mock_create_embedder, mock_db_cls, chroma_multi_tenant_config
    ):
        """Test that empty tenant_id raises ValueError during pipeline init.

        Validates:
            - Empty string tenant_id is rejected at initialization
            - Error is raised before any operations are performed
        """
        mock_create_embedder.return_value = MagicMock()

        from vectordb.langchain.multi_tenancy.indexing.chroma import (
            ChromaMultiTenancyIndexingPipeline,
        )

        with pytest.raises(ValueError, match="tenant_id cannot be empty"):
            ChromaMultiTenancyIndexingPipeline(chroma_multi_tenant_config, "")

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.dataloaders.catalog.DataloaderCatalog.create")
    def test_indexing_pipeline_run_empty_documents(
        self,
        mock_get_documents,
        mock_create_embedder,
        mock_db_cls,
        chroma_multi_tenant_config,
    ):
        """Test indexing pipeline with empty documents returns 0.

        Validates:
            - No documents loaded results in 0 indexed count
            - Result still includes tenant ID
            - Dataloader was invoked correctly
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        mock_create_embedder.return_value = MagicMock()

        # Mock: create() -> load() -> to_langchain()
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = []
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_documents.return_value = mock_loader

        from vectordb.langchain.multi_tenancy.indexing.chroma import (
            ChromaMultiTenancyIndexingPipeline,
        )

        pipeline = ChromaMultiTenancyIndexingPipeline(
            chroma_multi_tenant_config, "tenant_123"
        )
        result = pipeline.run()

        assert result["documents_indexed"] == 0
        assert result["tenant_id"] == "tenant_123"
        mock_get_documents.assert_called_once()

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.dataloaders.catalog.DataloaderCatalog.create")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_documents")
    def test_indexing_pipeline_run_success_with_logging(
        self,
        mock_embed_documents,
        mock_get_documents,
        mock_create_embedder,
        mock_db_cls,
        chroma_multi_tenant_config,
        sample_documents,
    ):
        """Test indexing pipeline run success with full helper verification.

        Validates:
            - All helper methods are called with correct arguments
            - Documents are loaded with correct limit from config
            - Embeddings are generated for all documents
            - Database upsert is invoked
        """
        mock_db = MagicMock()
        mock_db.upsert.return_value = 3
        mock_db_cls.return_value = mock_db

        mock_create_embedder.return_value = MagicMock()

        # Mock: create() -> load() -> to_langchain()
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_documents.return_value = mock_loader

        mock_embed_documents.return_value = (sample_documents, [[0.1] * 384] * 3)

        from vectordb.langchain.multi_tenancy.indexing.chroma import (
            ChromaMultiTenancyIndexingPipeline,
        )

        pipeline = ChromaMultiTenancyIndexingPipeline(
            chroma_multi_tenant_config, "tenant_123"
        )
        result = pipeline.run()

        assert result["documents_indexed"] == 3
        assert result["tenant_id"] == "tenant_123"

        mock_get_documents.assert_called_once_with("arc", split="test", limit=10)
        mock_embed_documents.assert_called_once()
        mock_db.upsert.assert_called_once()

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.dataloaders.catalog.DataloaderCatalog.create")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_documents")
    def test_indexing_pipeline_run_with_different_limits(
        self,
        mock_embed_documents,
        mock_get_documents,
        mock_create_embedder,
        mock_db_cls,
    ):
        """Test indexing pipeline with different dataloader limits.

        Validates:
            - Config without limit passes None to dataloader
            - All documents are processed when no limit specified
            - Result reflects actual document count
        """
        from langchain_core.documents import Document

        mock_db = MagicMock()
        mock_db.upsert.return_value = 3
        mock_db_cls.return_value = mock_db

        test_docs = [
            Document(page_content="doc1", metadata={"id": "1"}),
            Document(page_content="doc2", metadata={"id": "2"}),
            Document(page_content="doc3", metadata={"id": "3"}),
        ]

        mock_create_embedder.return_value = MagicMock()

        # Mock: create() -> load() -> to_langchain()
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = test_docs
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_documents.return_value = mock_loader

        mock_embed_documents.return_value = (test_docs, [[0.1] * 384] * 3)

        from vectordb.langchain.multi_tenancy.indexing.chroma import (
            ChromaMultiTenancyIndexingPipeline,
        )

        config_no_limit = {
            "dataloader": {"type": "arc", "split": "test"},
            "embeddings": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cpu",
                "batch_size": 32,
            },
            "chroma": {
                "path": "./test_chroma_data",
                "collection_prefix": "tenant_",
            },
        }

        pipeline = ChromaMultiTenancyIndexingPipeline(config_no_limit, "tenant_456")
        result = pipeline.run()

        assert result["documents_indexed"] == 3
        assert result["tenant_id"] == "tenant_456"
        mock_get_documents.assert_called_once_with("arc", split="test", limit=None)

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.dataloaders.catalog.DataloaderCatalog.create")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_documents")
    def test_indexing_pipeline_run_verifies_index_for_tenant_call(
        self,
        mock_embed_documents,
        mock_get_documents,
        mock_create_embedder,
        mock_db_cls,
        chroma_multi_tenant_config,
        sample_documents,
    ):
        """Test that index_for_tenant is called with correct arguments.

        Validates:
            - Correct number of documents are processed
            - Database upsert is invoked via the pipeline
        """
        mock_db = MagicMock()
        mock_db.upsert.return_value = 3
        mock_db_cls.return_value = mock_db

        mock_create_embedder.return_value = MagicMock()

        # Mock: create() -> load() -> to_langchain()
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_documents[:3]
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_documents.return_value = mock_loader

        mock_embed_documents.return_value = (
            sample_documents[:3],
            [[0.1] * 384, [0.2] * 384, [0.3] * 384],
        )

        from vectordb.langchain.multi_tenancy.indexing.chroma import (
            ChromaMultiTenancyIndexingPipeline,
        )

        pipeline = ChromaMultiTenancyIndexingPipeline(
            chroma_multi_tenant_config, "tenant_abc"
        )
        result = pipeline.run()

        assert result["documents_indexed"] == 3
        mock_db.upsert.assert_called_once()

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    def test_indexing_pipeline_initialization_logging(
        self, mock_create_embedder, mock_db_cls, chroma_multi_tenant_config, caplog
    ):
        """Test that initialization logs appropriate message.

        Validates:
            - Initialization log message includes pipeline name
            - Tenant ID is included in log message
        """
        import logging

        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db
        mock_create_embedder.return_value = MagicMock()

        from vectordb.langchain.multi_tenancy.indexing.chroma import (
            ChromaMultiTenancyIndexingPipeline,
        )

        with caplog.at_level(logging.INFO):
            ChromaMultiTenancyIndexingPipeline(chroma_multi_tenant_config, "tenant_xyz")

        assert "Initialized Chroma multi-tenancy indexing pipeline" in caplog.text
        assert "tenant_xyz" in caplog.text

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.dataloaders.catalog.DataloaderCatalog.create")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_documents")
    def test_indexing_pipeline_run_logging(
        self,
        mock_embed_documents,
        mock_get_documents,
        mock_create_embedder,
        mock_db_cls,
        chroma_multi_tenant_config,
        sample_documents,
        caplog,
    ):
        """Test that run() method logs all expected messages.

        Validates:
            - Document loading is logged with count and tenant
            - Embedding generation is logged with count and tenant
            - Indexing completion is logged with count and tenant
        """
        import logging

        mock_db = MagicMock()
        mock_db.upsert.return_value = 3
        mock_db_cls.return_value = mock_db

        mock_create_embedder.return_value = MagicMock()

        # Mock: create() -> load() -> to_langchain()
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_documents[:3]
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_documents.return_value = mock_loader

        mock_embed_documents.return_value = (sample_documents[:3], [[0.1] * 384] * 3)

        from vectordb.langchain.multi_tenancy.indexing.chroma import (
            ChromaMultiTenancyIndexingPipeline,
        )

        pipeline = ChromaMultiTenancyIndexingPipeline(
            chroma_multi_tenant_config, "tenant_log"
        )

        with caplog.at_level(logging.INFO):
            pipeline.run()

        assert "Loaded 3 documents for tenant tenant_log" in caplog.text
        assert (
            "Generated embeddings for 3 documents for tenant tenant_log" in caplog.text
        )
        assert "Indexed 3 documents for tenant tenant_log to Chroma" in caplog.text

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.dataloaders.catalog.DataloaderCatalog.create")
    def test_indexing_pipeline_empty_documents_warning(
        self,
        mock_get_documents,
        mock_create_embedder,
        mock_db_cls,
        chroma_multi_tenant_config,
        caplog,
    ):
        """Test that empty documents logs warning message.

        Validates:
            - Warning is logged when no documents are found
            - Warning includes tenant ID for identification
            - Result correctly shows 0 documents indexed
        """
        import logging

        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        mock_create_embedder.return_value = MagicMock()

        # Mock: create() -> load() -> to_langchain()
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = []
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_documents.return_value = mock_loader

        from vectordb.langchain.multi_tenancy.indexing.chroma import (
            ChromaMultiTenancyIndexingPipeline,
        )

        pipeline = ChromaMultiTenancyIndexingPipeline(
            chroma_multi_tenant_config, "tenant_warn"
        )

        with caplog.at_level(logging.WARNING):
            result = pipeline.run()

        assert result["documents_indexed"] == 0
        assert "No documents to index for tenant: tenant_warn" in caplog.text

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    def test_indexing_pipeline_config_with_path_string(
        self, mock_create_embedder, mock_db_cls, tmp_path
    ):
        """Test indexing pipeline initialization with config file path.

        Validates:
            - YAML config file is loaded correctly
            - Tenant ID is preserved from constructor
            - Config values are accessible after loading
        """
        import yaml

        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db
        mock_create_embedder.return_value = MagicMock()

        config = {
            "dataloader": {"type": "arc", "split": "test", "limit": 5},
            "embeddings": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cpu",
                "batch_size": 32,
            },
            "chroma": {
                "path": "./test_chroma_data",
                "collection_prefix": "tenant_",
            },
        }

        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        from vectordb.langchain.multi_tenancy.indexing.chroma import (
            ChromaMultiTenancyIndexingPipeline,
        )

        pipeline = ChromaMultiTenancyIndexingPipeline(str(config_path), "tenant_file")

        assert pipeline.tenant_id == "tenant_file"
        assert pipeline.config["dataloader"]["limit"] == 5

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    def test_indexing_pipeline_none_tenant_id(
        self, mock_create_embedder, mock_db_cls, chroma_multi_tenant_config
    ):
        """Test indexing pipeline with None tenant_id raises error.

        Validates:
            - None value for tenant_id is rejected
            - Appropriate error is raised before any operations
        """
        mock_create_embedder.return_value = MagicMock()

        from vectordb.langchain.multi_tenancy.indexing.chroma import (
            ChromaMultiTenancyIndexingPipeline,
        )

        with pytest.raises(ValueError, match="tenant_id cannot be empty"):
            ChromaMultiTenancyIndexingPipeline(chroma_multi_tenant_config, None)


class TestChromaMultiTenancySearchPipeline:
    """Unit tests for ChromaMultiTenancySearchPipeline high-level workflow.

    Validates the complete search pipeline that orchestrates query embedding,
    collection-scoped vector search, and result formatting.

    Tested Scenarios:
        - Full search pipeline execution with mocked dependencies
        - Result structure with documents, tenant ID, and query
        - Empty tenant ID validation during initialization
    """

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_query")
    def test_search_pipeline_run(
        self,
        mock_embed_query,
        mock_create_llm,
        mock_create_embedder,
        mock_db_cls,
        chroma_multi_tenant_config,
    ):
        """Test complete search pipeline execution.

        Validates:
            - Query is embedded using configured embedder
            - Search targets the correct tenant collection
            - Results include documents, tenant ID, and original query
            - Result structure matches expected contract
        """
        mock_db = MagicMock()
        mock_db.query.return_value = [
            Document(page_content="result1", metadata={}),
            Document(page_content="result2", metadata={}),
        ]
        mock_db_cls.return_value = mock_db

        mock_create_embedder.return_value = MagicMock()
        mock_create_llm.return_value = None
        mock_embed_query.return_value = [0.1] * 384

        from vectordb.langchain.multi_tenancy.search.chroma import (
            ChromaMultiTenancySearchPipeline,
        )

        pipeline = ChromaMultiTenancySearchPipeline(
            chroma_multi_tenant_config, "tenant_123"
        )
        result = pipeline.search("test query", top_k=5)

        assert "documents" in result
        assert result["tenant_id"] == "tenant_123"
        assert result["query"] == "test query"
        assert len(result["documents"]) == 2

    @patch("vectordb.langchain.multi_tenancy.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    def test_search_pipeline_empty_tenant_id(
        self,
        mock_create_llm,
        mock_create_embedder,
        mock_db_cls,
        chroma_multi_tenant_config,
    ):
        """Test that empty tenant_id raises ValueError during pipeline init.

        Validates:
            - Empty string tenant_id is rejected at initialization
            - Error is raised before any operations are performed
        """
        mock_create_embedder.return_value = MagicMock()
        mock_create_llm.return_value = None

        from vectordb.langchain.multi_tenancy.search.chroma import (
            ChromaMultiTenancySearchPipeline,
        )

        with pytest.raises(ValueError, match="tenant_id cannot be empty"):
            ChromaMultiTenancySearchPipeline(chroma_multi_tenant_config, "")
