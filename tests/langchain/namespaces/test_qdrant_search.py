"""Tests for Qdrant namespace search pipeline (LangChain)."""

from unittest.mock import MagicMock, patch

import pytest


class TestQdrantNamespaceSearchPipeline:
    """Unit tests for Qdrant namespace search pipeline."""

    @patch("vectordb.langchain.namespaces.qdrant.QdrantVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    def test_init_with_valid_namespace(
        self, mock_llm, mock_embedder, mock_db_cls, qdrant_namespace_config: dict
    ):
        """Test pipeline initialization with valid namespace."""
        mock_embedder.return_value = MagicMock()
        mock_llm.return_value = None

        from vectordb.langchain.namespaces.search.qdrant import (
            QdrantNamespaceSearchPipeline,
        )

        pipeline = QdrantNamespaceSearchPipeline(qdrant_namespace_config, "ns_abc")

        assert pipeline.namespace == "ns_abc"
        assert pipeline.config == qdrant_namespace_config

    @patch("vectordb.langchain.namespaces.qdrant.QdrantVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    def test_init_with_empty_namespace_raises_error(
        self, mock_llm, mock_embedder, mock_db_cls, qdrant_namespace_config: dict
    ):
        """Test initialization with empty namespace raises ValueError."""
        from vectordb.langchain.namespaces.search.qdrant import (
            QdrantNamespaceSearchPipeline,
        )

        with pytest.raises(ValueError, match="namespace cannot be empty"):
            QdrantNamespaceSearchPipeline(qdrant_namespace_config, "")

    @patch("vectordb.langchain.namespaces.qdrant.QdrantVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_query")
    def test_search_returns_documents(
        self,
        mock_embed_query,
        mock_llm,
        mock_embedder,
        mock_db_cls,
        qdrant_namespace_config: dict,
    ):
        """Test search returns documents from namespace."""
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance
        mock_llm.return_value = None
        mock_embed_query.return_value = [0.1] * 384

        mock_db_instance = MagicMock()
        mock_db_instance.query.return_value = ["doc1", "doc2"]
        mock_db_cls.return_value = mock_db_instance

        from vectordb.langchain.namespaces.search.qdrant import (
            QdrantNamespaceSearchPipeline,
        )

        pipeline = QdrantNamespaceSearchPipeline(qdrant_namespace_config, "ns_abc")
        result = pipeline.search("test query", top_k=5)

        assert "documents" in result
        assert result["namespace"] == "ns_abc"
        assert result["query"] == "test query"
        mock_db_instance.query.assert_called_once()

    @patch("vectordb.langchain.namespaces.qdrant.QdrantVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_query")
    def test_search_uses_namespace(
        self,
        mock_embed_query,
        mock_llm,
        mock_embedder,
        mock_db_cls,
        qdrant_namespace_config: dict,
    ):
        """Test search queries namespace-specific data."""
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance
        mock_llm.return_value = None
        mock_embed_query.return_value = [0.1] * 384

        mock_db_instance = MagicMock()
        mock_db_instance.query.return_value = ["doc1"]
        mock_db_cls.return_value = mock_db_instance

        from vectordb.langchain.namespaces.search.qdrant import (
            QdrantNamespaceSearchPipeline,
        )

        pipeline = QdrantNamespaceSearchPipeline(qdrant_namespace_config, "ns_abc")
        pipeline.search("test query")

        call_kwargs = mock_db_instance.query.call_args.kwargs
        assert call_kwargs["namespace"] == "ns_abc"

    @patch("vectordb.langchain.namespaces.qdrant.QdrantVectorDB")
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
        qdrant_namespace_config: dict,
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

        from vectordb.langchain.namespaces.search.qdrant import (
            QdrantNamespaceSearchPipeline,
        )

        pipeline = QdrantNamespaceSearchPipeline(qdrant_namespace_config, "ns_abc")
        result = pipeline.search("test query", top_k=5)

        assert "answer" in result
        assert result["answer"] == "generated answer"
        mock_generate.assert_called_once_with(llm, "test query", documents)
