"""Tests for Pinecone namespace search pipeline (LangChain)."""

from unittest.mock import MagicMock, patch

import pytest


class TestPineconeNamespaceSearchPipeline:
    """Unit tests for Pinecone namespace search pipeline."""

    @patch("vectordb.langchain.namespaces.pinecone.PineconeVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    def test_init_with_valid_namespace(
        self, mock_llm, mock_embedder, mock_db_cls, pinecone_namespace_config: dict
    ):
        """Test pipeline initialization with valid namespace."""
        mock_embedder.return_value = MagicMock()
        mock_llm.return_value = None

        from vectordb.langchain.namespaces.search.pinecone import (
            PineconeNamespaceSearchPipeline,
        )

        pipeline = PineconeNamespaceSearchPipeline(pinecone_namespace_config, "ns_123")

        assert pipeline.namespace == "ns_123"
        assert pipeline.config == pinecone_namespace_config

    @patch("vectordb.langchain.namespaces.pinecone.PineconeVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    def test_init_with_empty_namespace_raises_error(
        self, mock_llm, mock_embedder, mock_db_cls, pinecone_namespace_config: dict
    ):
        """Test initialization with empty namespace raises ValueError."""
        from vectordb.langchain.namespaces.search.pinecone import (
            PineconeNamespaceSearchPipeline,
        )

        with pytest.raises(ValueError, match="namespace cannot be empty"):
            PineconeNamespaceSearchPipeline(pinecone_namespace_config, "")

    @patch("vectordb.langchain.namespaces.pinecone.PineconeVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_query")
    def test_search_returns_documents_from_namespace(
        self,
        mock_embed_query,
        mock_llm,
        mock_embedder,
        mock_db_cls,
        pinecone_namespace_config: dict,
    ):
        """Test search returns documents from namespace-specific namespace."""
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance
        mock_llm.return_value = None
        mock_embed_query.return_value = [0.1] * 384

        mock_db_instance = MagicMock()
        mock_db_instance.query.return_value = ["doc1", "doc2"]
        mock_db_cls.return_value = mock_db_instance

        from vectordb.langchain.namespaces.search.pinecone import (
            PineconeNamespaceSearchPipeline,
        )

        pipeline = PineconeNamespaceSearchPipeline(pinecone_namespace_config, "ns_123")
        result = pipeline.search("test query", top_k=5)

        assert "documents" in result
        assert result["namespace"] == "ns_123"
        assert result["query"] == "test query"
        mock_db_instance.query.assert_called_once()

    @patch("vectordb.langchain.namespaces.pinecone.PineconeVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_query")
    def test_search_with_filters(
        self,
        mock_embed_query,
        mock_llm,
        mock_embedder,
        mock_db_cls,
        pinecone_namespace_config: dict,
    ):
        """Test search applies metadata filters."""
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance
        mock_llm.return_value = None
        mock_embed_query.return_value = [0.1] * 384

        mock_db_instance = MagicMock()
        mock_db_instance.query.return_value = ["filtered_doc"]
        mock_db_cls.return_value = mock_db_instance

        from vectordb.langchain.namespaces.search.pinecone import (
            PineconeNamespaceSearchPipeline,
        )

        pipeline = PineconeNamespaceSearchPipeline(pinecone_namespace_config, "ns_123")
        filters = {"source": "wiki"}
        result = pipeline.search("test query", top_k=5, filters=filters)

        assert "documents" in result
        call_kwargs = mock_db_instance.query.call_args.kwargs
        assert call_kwargs["filters"] == filters

    @patch("vectordb.langchain.namespaces.pinecone.PineconeVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_query")
    def test_search_uses_namespace(
        self,
        mock_embed_query,
        mock_llm,
        mock_embedder,
        mock_db_cls,
        pinecone_namespace_config: dict,
    ):
        """Test search queries namespace-specific namespace."""
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance
        mock_llm.return_value = None
        mock_embed_query.return_value = [0.1] * 384

        mock_db_instance = MagicMock()
        mock_db_instance.query.return_value = ["doc1"]
        mock_db_cls.return_value = mock_db_instance

        from vectordb.langchain.namespaces.search.pinecone import (
            PineconeNamespaceSearchPipeline,
        )

        pipeline = PineconeNamespaceSearchPipeline(pinecone_namespace_config, "ns_abc")
        pipeline.search("test query")

        call_kwargs = mock_db_instance.query.call_args.kwargs
        assert call_kwargs["namespace"] == "ns_abc"

    @patch("vectordb.langchain.namespaces.pinecone.PineconeVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    @patch("vectordb.langchain.utils.RAGHelper.generate")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_query")
    def test_search_with_llm_generates_answer(
        self,
        mock_embed_query,
        mock_generate,
        mock_llm,
        mock_embedder,
        mock_db_cls,
        pinecone_namespace_config: dict,
    ):
        """Test search includes generated answer when LLM is configured."""
        llm = MagicMock()
        mock_embedder.return_value = MagicMock()
        mock_llm.return_value = llm
        mock_embed_query.return_value = [0.1] * 384

        documents = ["doc1", "doc2"]
        mock_db_instance = MagicMock()
        mock_db_instance.query.return_value = documents
        mock_db_cls.return_value = mock_db_instance
        mock_generate.return_value = "generated answer"

        from vectordb.langchain.namespaces.search.pinecone import (
            PineconeNamespaceSearchPipeline,
        )

        pipeline = PineconeNamespaceSearchPipeline(pinecone_namespace_config, "ns_123")
        result = pipeline.search("test query", top_k=5)

        assert "answer" in result
        assert result["answer"] == "generated answer"
        mock_generate.assert_called_once_with(llm, "test query", documents)

    @patch("vectordb.langchain.namespaces.pinecone.PineconeVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_query")
    def test_search_with_top_k_parameter(
        self,
        mock_embed_query,
        mock_llm,
        mock_embedder,
        mock_db_cls,
        pinecone_namespace_config: dict,
    ):
        """Test search respects top_k parameter."""
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance
        mock_llm.return_value = None
        mock_embed_query.return_value = [0.1] * 384

        mock_db_instance = MagicMock()
        mock_db_instance.query.return_value = ["doc1", "doc2", "doc3"]
        mock_db_cls.return_value = mock_db_instance

        from vectordb.langchain.namespaces.search.pinecone import (
            PineconeNamespaceSearchPipeline,
        )

        pipeline = PineconeNamespaceSearchPipeline(pinecone_namespace_config, "ns_123")
        pipeline.search("test query", top_k=3)

        call_kwargs = mock_db_instance.query.call_args.kwargs
        assert call_kwargs["top_k"] == 3
