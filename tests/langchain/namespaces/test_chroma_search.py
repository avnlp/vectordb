"""Tests for Chroma namespace search pipeline (LangChain).

This module tests the ChromaNamespaceSearchPipeline which orchestrates
query embedding, collection-scoped vector search, and result formatting
within a single namespace.

Test Coverage:
    - Pipeline initialization with valid and invalid namespaces
    - Search result structure with documents, query, and namespace
    - Collection name resolution for namespace-scoped queries

All tests use mocking to avoid requiring actual Chroma database.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


class TestChromaNamespaceSearchPipeline:
    """Unit tests for ChromaNamespaceSearchPipeline high-level workflow.

    Validates the complete search pipeline that orchestrates query embedding,
    collection-scoped vector search, and result formatting.

    Tested Scenarios:
        - Pipeline initialization with valid namespace
        - Empty namespace rejection
        - Search returning documents with correct structure
        - Collection name resolution using prefix
    """

    @patch("vectordb.langchain.namespaces.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    def test_init_with_valid_namespace(
        self,
        mock_create_llm,
        mock_create_embedder,
        mock_db_cls,
        chroma_namespace_config: dict,
    ):
        """Test pipeline initialization with valid namespace.

        Validates:
            - Namespace is stored correctly
            - Config is accessible after initialization
        """
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db
        mock_create_embedder.return_value = MagicMock()
        mock_create_llm.return_value = None

        from vectordb.langchain.namespaces.search.chroma import (
            ChromaNamespaceSearchPipeline,
        )

        pipeline = ChromaNamespaceSearchPipeline(chroma_namespace_config, "arc_train")

        assert pipeline.namespace == "arc_train"
        assert pipeline.config == chroma_namespace_config

    @patch("vectordb.langchain.namespaces.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    def test_init_with_empty_namespace_raises_error(
        self,
        mock_create_llm,
        mock_create_embedder,
        mock_db_cls,
        chroma_namespace_config: dict,
    ):
        """Test initialization with empty namespace raises ValueError.

        Validates:
            - Empty string namespace is rejected at initialization
            - Error is raised before any operations are performed
        """
        mock_create_embedder.return_value = MagicMock()
        mock_create_llm.return_value = None

        from vectordb.langchain.namespaces.search.chroma import (
            ChromaNamespaceSearchPipeline,
        )

        with pytest.raises(ValueError, match="namespace cannot be empty"):
            ChromaNamespaceSearchPipeline(chroma_namespace_config, "")

    @patch("vectordb.langchain.namespaces.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_query")
    def test_search_returns_documents(
        self,
        mock_embed_query,
        mock_create_llm,
        mock_create_embedder,
        mock_db_cls,
        chroma_namespace_config: dict,
    ):
        """Test complete search pipeline execution.

        Validates:
            - Results include documents, namespace, and original query
            - Result structure matches expected contract
            - Correct number of documents returned
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

        from vectordb.langchain.namespaces.search.chroma import (
            ChromaNamespaceSearchPipeline,
        )

        pipeline = ChromaNamespaceSearchPipeline(chroma_namespace_config, "arc_train")
        result = pipeline.search("test query", top_k=5)

        assert "documents" in result
        assert result["namespace"] == "arc_train"
        assert result["query"] == "test query"
        assert len(result["documents"]) == 2

    @patch("vectordb.langchain.namespaces.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_query")
    def test_search_uses_collection_name(
        self,
        mock_embed_query,
        mock_create_llm,
        mock_create_embedder,
        mock_db_cls,
        chroma_namespace_config: dict,
    ):
        """Test search uses correct collection name for namespace.

        Validates:
            - db.query is called with collection_name derived from prefix + namespace
            - For namespace "ns_abc", collection_name is "ns_ns_abc"
        """
        mock_db = MagicMock()
        mock_db.query.return_value = []
        mock_db_cls.return_value = mock_db

        mock_create_embedder.return_value = MagicMock()
        mock_create_llm.return_value = None
        mock_embed_query.return_value = [0.1] * 384

        from vectordb.langchain.namespaces.search.chroma import (
            ChromaNamespaceSearchPipeline,
        )

        pipeline = ChromaNamespaceSearchPipeline(chroma_namespace_config, "ns_abc")
        pipeline.search("test query")

        call_kwargs = mock_db.query.call_args.kwargs
        assert call_kwargs["collection_name"] == "ns_ns_abc"

    @patch("vectordb.langchain.namespaces.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.utils.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.utils.RAGHelper.create_llm")
    @patch("vectordb.langchain.utils.RAGHelper.generate")
    @patch("vectordb.langchain.utils.EmbedderHelper.embed_query")
    def test_search_with_llm_generates_answer(
        self,
        mock_embed_query,
        mock_generate,
        mock_create_llm,
        mock_create_embedder,
        mock_db_cls,
        chroma_namespace_config: dict,
    ):
        """Test search includes generated answer when LLM is configured."""
        llm = MagicMock()

        documents = [
            Document(page_content="result1", metadata={}),
            Document(page_content="result2", metadata={}),
        ]

        mock_db = MagicMock()
        mock_db.query.return_value = documents
        mock_db_cls.return_value = mock_db

        mock_create_embedder.return_value = MagicMock()
        mock_create_llm.return_value = llm
        mock_embed_query.return_value = [0.1] * 384
        mock_generate.return_value = "generated answer"

        from vectordb.langchain.namespaces.search.chroma import (
            ChromaNamespaceSearchPipeline,
        )

        pipeline = ChromaNamespaceSearchPipeline(chroma_namespace_config, "arc_train")
        result = pipeline.search("test query", top_k=5)

        assert "answer" in result
        assert result["answer"] == "generated answer"
        mock_generate.assert_called_once_with(llm, "test query", documents)
