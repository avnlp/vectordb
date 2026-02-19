"""Tests for Milvus semantic search pipeline (indexing and search)."""

from unittest.mock import MagicMock, patch

import pytest

from vectordb.haystack.semantic_search.indexing.milvus import (
    MilvusSemanticIndexingPipeline,
)
from vectordb.haystack.semantic_search.search.milvus import (
    MilvusSemanticSearchPipeline,
)


class TestMilvusSemanticSearch:
    """Unit tests for Milvus semantic search pipeline (indexing and search)."""

    # Indexing tests
    @patch("vectordb.haystack.semantic_search.indexing.milvus.MilvusVectorDB")
    @patch(
        "vectordb.haystack.semantic_search.indexing.milvus.EmbedderFactory.create_document_embedder"
    )
    @patch("vectordb.haystack.semantic_search.indexing.milvus.DataloaderCatalog.create")
    def test_indexing_init_loads_config(
        self,
        mock_get_docs: MagicMock,
        mock_embedder: MagicMock,
        mock_db: MagicMock,
        milvus_config: dict,
    ) -> None:
        """Test indexing pipeline initialization loads config correctly."""
        pipeline = MilvusSemanticIndexingPipeline(milvus_config)
        assert pipeline.config == milvus_config
        assert pipeline.collection_name == "test_collection"
        assert pipeline.dimension == 384

    @patch("vectordb.haystack.semantic_search.indexing.milvus.MilvusVectorDB")
    @patch(
        "vectordb.haystack.semantic_search.indexing.milvus.EmbedderFactory.create_document_embedder"
    )
    @patch("vectordb.haystack.semantic_search.indexing.milvus.DataloaderCatalog.create")
    def test_indexing_run_calls_create_collection(
        self,
        mock_get_docs: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        milvus_config: dict,
        sample_documents: list,
    ) -> None:
        """Test indexing run method calls create_collection."""
        # Setup mocks
        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"documents": sample_documents}
        mock_make_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db

        # Run pipeline
        pipeline = MilvusSemanticIndexingPipeline(milvus_config)
        pipeline.run()

        # Assert create_collection was called
        mock_db.create_collection.assert_called_once()
        call_kwargs = mock_db.create_collection.call_args.kwargs
        assert call_kwargs["collection_name"] == "test_collection"
        assert call_kwargs["dimension"] == 384

    @patch("vectordb.haystack.semantic_search.indexing.milvus.MilvusVectorDB")
    @patch(
        "vectordb.haystack.semantic_search.indexing.milvus.EmbedderFactory.create_document_embedder"
    )
    @patch("vectordb.haystack.semantic_search.indexing.milvus.DataloaderCatalog.create")
    def test_indexing_run_calls_insert_documents(
        self,
        mock_get_docs: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        milvus_config: dict,
        sample_documents: list,
    ) -> None:
        """Test indexing run method calls insert_documents."""
        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"documents": sample_documents}
        mock_make_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db

        pipeline = MilvusSemanticIndexingPipeline(milvus_config)
        pipeline.run()

        mock_db.insert_documents.assert_called_once()

    @patch("vectordb.haystack.semantic_search.indexing.milvus.MilvusVectorDB")
    @patch(
        "vectordb.haystack.semantic_search.indexing.milvus.EmbedderFactory.create_document_embedder"
    )
    @patch("vectordb.haystack.semantic_search.indexing.milvus.DataloaderCatalog.create")
    def test_indexing_run_returns_document_count(
        self,
        mock_get_docs: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        milvus_config: dict,
        sample_documents: list,
    ) -> None:
        """Test indexing run returns correct document count."""
        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"documents": sample_documents}
        mock_make_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db

        pipeline = MilvusSemanticIndexingPipeline(milvus_config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)

    @patch("vectordb.haystack.semantic_search.indexing.milvus.MilvusVectorDB")
    @patch(
        "vectordb.haystack.semantic_search.indexing.milvus.EmbedderFactory.create_document_embedder"
    )
    @patch("vectordb.haystack.semantic_search.indexing.milvus.DataloaderCatalog.create")
    def test_indexing_run_no_documents(
        self,
        mock_get_docs: MagicMock,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        milvus_config: dict,
    ) -> None:
        """Test indexing run with no documents."""
        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = []
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader
        mock_embedder = MagicMock()
        mock_make_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db

        pipeline = MilvusSemanticIndexingPipeline(milvus_config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0
        mock_db.create_collection.assert_not_called()

    def test_indexing_init_invalid_config(self) -> None:
        """Test indexing initialization with invalid config."""
        invalid_config = {"embeddings": {"model": "test"}}
        with pytest.raises(ValueError):
            MilvusSemanticIndexingPipeline(invalid_config)

    # Search tests
    @patch("vectordb.haystack.semantic_search.search.milvus.MilvusVectorDB")
    @patch(
        "vectordb.haystack.semantic_search.search.milvus.EmbedderFactory.create_text_embedder"
    )
    def test_search_init_loads_config(
        self,
        mock_embedder: MagicMock,
        mock_db: MagicMock,
        milvus_config: dict,
    ) -> None:
        """Test search pipeline initialization loads config correctly."""
        pipeline = MilvusSemanticSearchPipeline(milvus_config)
        assert pipeline.config == milvus_config
        assert pipeline.collection_name == "test_collection"

    @patch("vectordb.haystack.semantic_search.search.milvus.MilvusVectorDB")
    @patch(
        "vectordb.haystack.semantic_search.search.milvus.EmbedderFactory.create_text_embedder"
    )
    def test_search_calls_embedder(
        self,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        milvus_config: dict,
        sample_documents: list,
    ) -> None:
        """Test search method calls text embedder."""
        # Setup mocks
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_make_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.search.return_value = sample_documents
        mock_db_class.return_value = mock_db

        pipeline = MilvusSemanticSearchPipeline(milvus_config)
        pipeline.search("test query", top_k=5)

        mock_embedder.run.assert_called_once_with(text="test query")

    @patch("vectordb.haystack.semantic_search.search.milvus.MilvusVectorDB")
    @patch(
        "vectordb.haystack.semantic_search.search.milvus.EmbedderFactory.create_text_embedder"
    )
    def test_search_calls_db_search(
        self,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        milvus_config: dict,
        sample_documents: list,
    ) -> None:
        """Test search method calls database search."""
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_make_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.search.return_value = sample_documents
        mock_db_class.return_value = mock_db

        pipeline = MilvusSemanticSearchPipeline(milvus_config)
        pipeline.search("test query", top_k=5)

        mock_db.search.assert_called_once()
        call_kwargs = mock_db.search.call_args.kwargs
        assert call_kwargs["top_k"] == 10  # 5 * 2 for diversification

    @patch("vectordb.haystack.semantic_search.search.milvus.MilvusVectorDB")
    @patch(
        "vectordb.haystack.semantic_search.search.milvus.EmbedderFactory.create_text_embedder"
    )
    def test_search_returns_documents(
        self,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        milvus_config: dict,
        sample_documents: list,
    ) -> None:
        """Test search returns documents."""
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_make_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.search.return_value = sample_documents
        mock_db_class.return_value = mock_db

        pipeline = MilvusSemanticSearchPipeline(milvus_config)
        result = pipeline.search("test query", top_k=5)

        assert "documents" in result
        assert "query" in result
        assert result["query"] == "test query"
        assert len(result["documents"]) <= 5

    @patch("vectordb.haystack.semantic_search.search.milvus.MilvusVectorDB")
    @patch(
        "vectordb.haystack.semantic_search.search.milvus.EmbedderFactory.create_text_embedder"
    )
    def test_search_with_filters(
        self,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        milvus_config: dict,
        sample_documents: list,
    ) -> None:
        """Test search with metadata filters."""
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_make_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.search.return_value = sample_documents
        mock_db_class.return_value = mock_db

        pipeline = MilvusSemanticSearchPipeline(milvus_config)
        pipeline.search(
            "test query",
            top_k=5,
            filters={"source": "wiki"},
        )

        mock_db.search.assert_called_once()
        call_kwargs = mock_db.search.call_args.kwargs
        assert call_kwargs["filter_expr"] is not None

    @patch("vectordb.haystack.semantic_search.search.milvus.MilvusVectorDB")
    @patch(
        "vectordb.haystack.semantic_search.search.milvus.EmbedderFactory.create_text_embedder"
    )
    def test_search_rag_disabled(
        self,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        milvus_config: dict,
        sample_documents: list,
    ) -> None:
        """Test search without RAG."""
        milvus_config["rag"] = {"enabled": False}
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_make_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.search.return_value = sample_documents
        mock_db_class.return_value = mock_db

        pipeline = MilvusSemanticSearchPipeline(milvus_config)
        result = pipeline.search("test query", top_k=5)

        assert "answer" not in result

    @patch("vectordb.haystack.semantic_search.search.milvus.RAGHelper.create_generator")
    @patch("vectordb.haystack.semantic_search.search.milvus.MilvusVectorDB")
    @patch(
        "vectordb.haystack.semantic_search.search.milvus.EmbedderFactory.create_text_embedder"
    )
    def test_search_rag_enabled_generates_answer(
        self,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        mock_create_generator: MagicMock,
        milvus_config: dict,
        sample_documents: list,
    ) -> None:
        """Test search returns RAG answer when enabled."""
        milvus_config["rag"] = {"enabled": True, "generator_model": "test-model"}
        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": [0.1] * 384}
        mock_make_embedder.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.search.return_value = sample_documents
        mock_db_class.return_value = mock_db

        mock_generator = MagicMock()
        mock_generator.run.return_value = {"replies": ["generated answer"]}
        mock_create_generator.return_value = mock_generator

        pipeline = MilvusSemanticSearchPipeline(milvus_config)
        result = pipeline.search("test query", top_k=2)

        assert result["answer"] == "generated answer"
        mock_generator.run.assert_called_once()

    @patch("vectordb.haystack.semantic_search.search.milvus.MilvusVectorDB")
    @patch(
        "vectordb.haystack.semantic_search.search.milvus.EmbedderFactory.create_text_embedder"
    )
    def test_build_milvus_filter_empty_returns_none(
        self,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        milvus_config: dict,
    ) -> None:
        """Test building filters returns None for empty input."""
        pipeline = MilvusSemanticSearchPipeline(milvus_config)
        assert pipeline._build_milvus_filter({}) is None

    @patch("vectordb.haystack.semantic_search.search.milvus.MilvusVectorDB")
    @patch(
        "vectordb.haystack.semantic_search.search.milvus.EmbedderFactory.create_text_embedder"
    )
    def test_build_milvus_filter_builds_expression(
        self,
        mock_make_embedder: MagicMock,
        mock_db_class: MagicMock,
        milvus_config: dict,
    ) -> None:
        """Test filter expression building for mixed filter types."""
        pipeline = MilvusSemanticSearchPipeline(milvus_config)
        filters = {
            "source": {"$eq": "wiki"},
            "views": {"$gte": 10},
            "status": "active",
            "count": 2,
        }

        expression = pipeline._build_milvus_filter(filters)

        assert (
            expression == 'metadata["source"] == "wiki" and '
            'metadata["views"] >= 10 and '
            'metadata["status"] == "active" and '
            'metadata["count"] == 2'
        )

    def test_search_init_invalid_config(self) -> None:
        """Test search initialization with invalid config."""
        invalid_config = {"embeddings": {"model": "test"}}
        with pytest.raises(ValueError):
            MilvusSemanticSearchPipeline(invalid_config)
