"""Tests for Weaviate namespace search pipeline (LangChain)."""

from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import Document as HaystackDocument


class TestWeaviateNamespaceSearchPipeline:
    """Unit tests for Weaviate namespace search pipeline."""

    @patch("vectordb.langchain.namespaces.weaviate.WeaviateVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    def test_init_with_valid_namespace(
        self,
        mock_llm: MagicMock,
        mock_embedder: MagicMock,
        mock_db_cls: MagicMock,
        weaviate_namespace_config: dict,
    ) -> None:
        """Test pipeline initialization with valid namespace."""
        mock_embedder.return_value = MagicMock()
        mock_llm.return_value = None

        from vectordb.langchain.namespaces.search.weaviate import (
            WeaviateNamespaceSearchPipeline,
        )

        pipeline = WeaviateNamespaceSearchPipeline(weaviate_namespace_config, "ns_123")

        assert pipeline.namespace == "ns_123"
        assert pipeline.config == weaviate_namespace_config

    @patch("vectordb.langchain.namespaces.weaviate.WeaviateVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    def test_init_with_empty_namespace_raises_error(
        self,
        mock_llm: MagicMock,
        mock_embedder: MagicMock,
        mock_db_cls: MagicMock,
        weaviate_namespace_config: dict,
    ) -> None:
        """Test initialization with empty namespace raises ValueError."""
        from vectordb.langchain.namespaces.search.weaviate import (
            WeaviateNamespaceSearchPipeline,
        )

        with pytest.raises(ValueError, match="namespace cannot be empty"):
            WeaviateNamespaceSearchPipeline(weaviate_namespace_config, "")

    @patch("vectordb.langchain.namespaces.weaviate.WeaviateVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_query")
    def test_search_returns_documents(
        self,
        mock_embed_query: MagicMock,
        mock_llm: MagicMock,
        mock_embedder: MagicMock,
        mock_db_cls: MagicMock,
        weaviate_namespace_config: dict,
    ) -> None:
        """Test search returns documents from namespace."""
        mock_embedder.return_value = MagicMock()
        mock_llm.return_value = None
        mock_embed_query.return_value = [0.1] * 384

        haystack_docs = [
            HaystackDocument(content="doc1", meta={}, id="1"),
            HaystackDocument(content="doc2", meta={}, id="2"),
        ]
        mock_db_instance = MagicMock()
        mock_db_instance.query.return_value = haystack_docs
        mock_db_cls.return_value = mock_db_instance

        from vectordb.langchain.namespaces.search.weaviate import (
            WeaviateNamespaceSearchPipeline,
        )

        pipeline = WeaviateNamespaceSearchPipeline(weaviate_namespace_config, "ns_123")
        result = pipeline.search("test query", top_k=5)

        assert "documents" in result
        assert result["namespace"] == "ns_123"
        assert result["query"] == "test query"
        assert len(result["documents"]) == 2

    @patch("vectordb.langchain.namespaces.weaviate.WeaviateVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_query")
    def test_search_uses_tenant(
        self,
        mock_embed_query: MagicMock,
        mock_llm: MagicMock,
        mock_embedder: MagicMock,
        mock_db_cls: MagicMock,
        weaviate_namespace_config: dict,
    ) -> None:
        """Test search queries with correct return_documents parameter."""
        mock_embedder.return_value = MagicMock()
        mock_llm.return_value = None
        mock_embed_query.return_value = [0.1] * 384

        haystack_docs = [
            HaystackDocument(content="doc1", meta={}, id="1"),
        ]
        mock_db_instance = MagicMock()
        mock_db_instance.query.return_value = haystack_docs
        mock_db_cls.return_value = mock_db_instance

        from vectordb.langchain.namespaces.search.weaviate import (
            WeaviateNamespaceSearchPipeline,
        )

        pipeline = WeaviateNamespaceSearchPipeline(weaviate_namespace_config, "ns_abc")
        pipeline.search("test query")

        call_kwargs = mock_db_instance.query.call_args.kwargs
        assert call_kwargs["return_documents"] is True

    @patch("vectordb.langchain.namespaces.weaviate.WeaviateVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    @patch("vectordb.langchain.utils.RAGHelper.generate")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_query")
    def test_search_with_llm_generates_answer(
        self,
        mock_embed_query: MagicMock,
        mock_generate: MagicMock,
        mock_llm: MagicMock,
        mock_embedder: MagicMock,
        mock_db_cls: MagicMock,
        weaviate_namespace_config: dict,
    ) -> None:
        """Test search includes generated answer when LLM is configured."""
        llm = MagicMock()
        mock_embedder.return_value = MagicMock()
        mock_llm.return_value = llm
        mock_embed_query.return_value = [0.1] * 384

        haystack_docs = [
            HaystackDocument(content="doc1", meta={}, id="1"),
            HaystackDocument(content="doc2", meta={}, id="2"),
        ]
        mock_db_instance = MagicMock()
        mock_db_instance.query.return_value = haystack_docs
        mock_db_cls.return_value = mock_db_instance
        mock_generate.return_value = "generated answer"

        from vectordb.langchain.namespaces.search.weaviate import (
            WeaviateNamespaceSearchPipeline,
        )

        pipeline = WeaviateNamespaceSearchPipeline(weaviate_namespace_config, "ns_123")
        result = pipeline.search("test query", top_k=5)

        assert "answer" in result
        assert result["answer"] == "generated answer"
        mock_generate.assert_called_once()
        assert mock_generate.call_args[0][0] == llm
        assert mock_generate.call_args[0][1] == "test query"
