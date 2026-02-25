"""Tests for Weaviate multi-tenancy search pipeline (LangChain)."""

from unittest.mock import MagicMock, patch

import pytest


class TestWeaviateMultiTenancySearchPipeline:
    """Unit tests for Weaviate multi-tenancy search pipeline."""

    @patch("vectordb.langchain.multi_tenancy.weaviate.WeaviateVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    def test_init_with_valid_tenant_id(
        self, mock_llm, mock_embedder, mock_db_cls, weaviate_multi_tenant_config: dict
    ):
        """Test pipeline initialization with valid tenant ID."""
        mock_embedder.return_value = MagicMock()
        mock_llm.return_value = None

        from vectordb.langchain.multi_tenancy.search.weaviate import (
            WeaviateMultiTenancySearchPipeline,
        )

        pipeline = WeaviateMultiTenancySearchPipeline(
            weaviate_multi_tenant_config, "tenant_123"
        )

        assert pipeline.tenant_id == "tenant_123"
        assert pipeline.config == weaviate_multi_tenant_config

    @patch("vectordb.langchain.multi_tenancy.weaviate.WeaviateVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    def test_init_with_empty_tenant_id_raises_error(
        self, mock_llm, mock_embedder, mock_db_cls, weaviate_multi_tenant_config: dict
    ):
        """Test initialization with empty tenant_id raises ValueError."""
        from vectordb.langchain.multi_tenancy.search.weaviate import (
            WeaviateMultiTenancySearchPipeline,
        )

        with pytest.raises(ValueError, match="tenant_id cannot be empty"):
            WeaviateMultiTenancySearchPipeline(weaviate_multi_tenant_config, "")

    @patch("vectordb.langchain.multi_tenancy.weaviate.WeaviateVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_query")
    def test_search_returns_documents_from_tenant_collection(
        self,
        mock_embed_query,
        mock_llm,
        mock_embedder,
        mock_db_cls,
        weaviate_multi_tenant_config: dict,
    ):
        """Test search returns documents from tenant-specific collection."""
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance
        mock_llm.return_value = None
        mock_embed_query.return_value = [0.1] * 384

        mock_db_instance = MagicMock()
        mock_db_instance.query.return_value = ["doc1", "doc2"]
        mock_db_cls.return_value = mock_db_instance

        from vectordb.langchain.multi_tenancy.search.weaviate import (
            WeaviateMultiTenancySearchPipeline,
        )

        pipeline = WeaviateMultiTenancySearchPipeline(
            weaviate_multi_tenant_config, "tenant_123"
        )
        result = pipeline.search("test query", top_k=5)

        assert "documents" in result
        assert result["tenant_id"] == "tenant_123"
        assert result["query"] == "test query"
        mock_db_instance.query.assert_called_once()

    @patch("vectordb.langchain.multi_tenancy.weaviate.WeaviateVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_query")
    def test_search_with_filters(
        self,
        mock_embed_query,
        mock_llm,
        mock_embedder,
        mock_db_cls,
        weaviate_multi_tenant_config: dict,
    ):
        """Test search applies metadata filters."""
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance
        mock_llm.return_value = None
        mock_embed_query.return_value = [0.1] * 384

        mock_db_instance = MagicMock()
        mock_db_instance.query.return_value = ["filtered_doc"]
        mock_db_cls.return_value = mock_db_instance

        from vectordb.langchain.multi_tenancy.search.weaviate import (
            WeaviateMultiTenancySearchPipeline,
        )

        pipeline = WeaviateMultiTenancySearchPipeline(
            weaviate_multi_tenant_config, "tenant_123"
        )
        filters = {"source": "wiki"}
        result = pipeline.search("test query", top_k=5, filters=filters)

        assert "documents" in result
        call_kwargs = mock_db_instance.query.call_args.kwargs
        assert call_kwargs["filters"] == filters

    @patch("vectordb.langchain.multi_tenancy.weaviate.WeaviateVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_query")
    def test_search_uses_tenant_collection(
        self,
        mock_embed_query,
        mock_llm,
        mock_embedder,
        mock_db_cls,
        weaviate_multi_tenant_config: dict,
    ):
        """Test search queries tenant-specific collection."""
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance
        mock_llm.return_value = None
        mock_embed_query.return_value = [0.1] * 384

        mock_db_instance = MagicMock()
        mock_db_instance.query.return_value = ["doc1"]
        mock_db_cls.return_value = mock_db_instance

        from vectordb.langchain.multi_tenancy.search.weaviate import (
            WeaviateMultiTenancySearchPipeline,
        )

        pipeline = WeaviateMultiTenancySearchPipeline(
            weaviate_multi_tenant_config, "tenant_abc"
        )
        pipeline.search("test query")

        # Verify collection name is tenant-specific
        call_kwargs = mock_db_instance.query.call_args.kwargs
        assert "collection_name" in call_kwargs
        assert "tenant_abc" in call_kwargs["collection_name"]

    @patch("vectordb.langchain.multi_tenancy.weaviate.WeaviateVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_query")
    def test_search_with_top_k_parameter(
        self,
        mock_embed_query,
        mock_llm,
        mock_embedder,
        mock_db_cls,
        weaviate_multi_tenant_config: dict,
    ):
        """Test search respects top_k parameter."""
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance
        mock_llm.return_value = None
        mock_embed_query.return_value = [0.1] * 384

        mock_db_instance = MagicMock()
        mock_db_instance.query.return_value = ["doc1", "doc2", "doc3"]
        mock_db_cls.return_value = mock_db_instance

        from vectordb.langchain.multi_tenancy.search.weaviate import (
            WeaviateMultiTenancySearchPipeline,
        )

        pipeline = WeaviateMultiTenancySearchPipeline(
            weaviate_multi_tenant_config, "tenant_123"
        )
        pipeline.search("test query", top_k=3)

        call_kwargs = mock_db_instance.query.call_args.kwargs
        assert call_kwargs["top_k"] == 3

    @patch("vectordb.langchain.multi_tenancy.weaviate.WeaviateVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_query")
    def test_search_different_tenants_get_isolated_results(
        self,
        mock_embed_query,
        mock_llm,
        mock_embedder,
        mock_db_cls,
        weaviate_multi_tenant_config: dict,
    ):
        """Test that different tenants get isolated results."""
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance
        mock_llm.return_value = None
        mock_embed_query.return_value = [0.1] * 384

        mock_db_instance = MagicMock()
        mock_db_cls.return_value = mock_db_instance

        from vectordb.langchain.multi_tenancy.search.weaviate import (
            WeaviateMultiTenancySearchPipeline,
        )

        pipeline = WeaviateMultiTenancySearchPipeline(
            weaviate_multi_tenant_config, "tenant_A"
        )

        # First tenant
        mock_db_instance.query.return_value = ["docs_from_tenant_A"]
        result_a = pipeline.search("query")
        assert result_a["tenant_id"] == "tenant_A"

        # Change tenant ID and search again
        pipeline.tenant_id = "tenant_B"
        mock_db_instance.query.return_value = ["docs_from_tenant_B"]
        result_b = pipeline.search("query")
        assert result_b["tenant_id"] == "tenant_B"
