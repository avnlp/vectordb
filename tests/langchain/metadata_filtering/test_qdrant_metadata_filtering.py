"""Tests for Qdrant metadata filtering pipelines (LangChain)."""

from unittest.mock import MagicMock, patch

from vectordb.langchain.metadata_filtering.indexing.qdrant import (
    QdrantMetadataFilteringIndexingPipeline,
)
from vectordb.langchain.metadata_filtering.search.qdrant import (
    QdrantMetadataFilteringSearchPipeline,
)


class TestQdrantMetadataFilteringIndexing:
    """Unit tests for Qdrant metadata filtering indexing pipeline."""

    @patch("vectordb.langchain.metadata_filtering.indexing.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.metadata_filtering.indexing.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.indexing.qdrant.DataloaderCatalog.create"
    )
    def test_indexing_initialization(
        self,
        mock_get_docs,
        mock_embedder_helper,
        mock_db,
        qdrant_metadata_filtering_config,
    ):
        """Test pipeline initialization."""
        pipeline = QdrantMetadataFilteringIndexingPipeline(
            qdrant_metadata_filtering_config
        )
        assert pipeline.config == qdrant_metadata_filtering_config
        assert pipeline.collection_name == "test_metadata_filtering"

    @patch("vectordb.langchain.metadata_filtering.indexing.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.metadata_filtering.indexing.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.indexing.qdrant.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.indexing.qdrant.DataloaderCatalog.create"
    )
    def test_indexing_run_with_documents(
        self,
        mock_get_docs,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test indexing with documents."""
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader
        mock_embed_docs.return_value = (sample_documents, [[0.1] * 384] * 5)

        mock_db_inst = MagicMock()
        mock_db_inst.upsert.return_value = len(sample_documents)
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "",
                "collection_name": "test_metadata_filtering",
                "dimension": 384,
            },
        }

        pipeline = QdrantMetadataFilteringIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.upsert.assert_called_once()

    @patch("vectordb.langchain.metadata_filtering.indexing.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.metadata_filtering.indexing.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.indexing.qdrant.DataloaderCatalog.create"
    )
    def test_indexing_run_no_documents(
        self, mock_get_docs, mock_embedder_helper, mock_db
    ):
        """Test indexing with no documents."""
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = []
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "",
                "collection_name": "test_metadata_filtering",
                "dimension": 384,
            },
        }

        pipeline = QdrantMetadataFilteringIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0

    @patch("vectordb.langchain.metadata_filtering.indexing.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.metadata_filtering.indexing.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.indexing.qdrant.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.indexing.qdrant.DataloaderCatalog.create"
    )
    def test_indexing_with_recreate(
        self,
        mock_get_docs,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test indexing with collection recreation."""
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader
        mock_embed_docs.return_value = (sample_documents, [[0.1] * 384] * 5)

        mock_db_inst = MagicMock()
        mock_db_inst.upsert.return_value = len(sample_documents)
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "",
                "collection_name": "test_metadata_filtering",
                "dimension": 384,
                "recreate": True,
            },
        }

        pipeline = QdrantMetadataFilteringIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.delete_collection.assert_called_once_with(
            "test_metadata_filtering"
        )


class TestQdrantMetadataFilteringSearch:
    """Unit tests for Qdrant metadata filtering search pipeline."""

    @patch("vectordb.langchain.metadata_filtering.search.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.metadata_filtering.search.qdrant.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.metadata_filtering.search.qdrant.RAGHelper.create_llm")
    def test_search_initialization(
        self,
        mock_llm_helper,
        mock_embedder_helper,
        mock_db,
        qdrant_metadata_filtering_config,
    ):
        """Test search pipeline initialization."""
        mock_llm_helper.return_value = None

        pipeline = QdrantMetadataFilteringSearchPipeline(
            qdrant_metadata_filtering_config
        )
        assert pipeline.config == qdrant_metadata_filtering_config
        assert pipeline.collection_name == "test_metadata_filtering"

    @patch("vectordb.langchain.metadata_filtering.search.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.metadata_filtering.search.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.search.qdrant.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.metadata_filtering.search.qdrant.RAGHelper.create_llm")
    def test_search_execution(
        self,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test search execution."""
        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "",
                "collection_name": "test_metadata_filtering",
                "dimension": 384,
            },
            "rag": {"enabled": False},
        }

        pipeline = QdrantMetadataFilteringSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert len(result["documents"]) > 0

    @patch("vectordb.langchain.metadata_filtering.search.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.metadata_filtering.search.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.search.qdrant.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.metadata_filtering.search.qdrant.RAGHelper.create_llm")
    def test_search_with_filters(
        self,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test search with metadata filters."""
        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "",
                "collection_name": "test_metadata_filtering",
                "dimension": 384,
            },
            "rag": {"enabled": False},
        }

        pipeline = QdrantMetadataFilteringSearchPipeline(config)
        filters = {"source": "wiki"}
        result = pipeline.search("test query", top_k=5, filters=filters)

        assert result["query"] == "test query"
        mock_db_inst.query.assert_called_once()
        call_kwargs = mock_db_inst.query.call_args.kwargs
        assert call_kwargs["filters"] == filters

    @patch("vectordb.langchain.metadata_filtering.search.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.metadata_filtering.search.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.search.qdrant.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.metadata_filtering.search.qdrant.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.metadata_filtering.search.qdrant.DocumentFilter.filter_by_metadata"
    )
    def test_search_with_configured_filters(
        self,
        mock_filter,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test search with filters configured in config."""
        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None
        mock_filter.return_value = sample_documents[:2]

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "",
                "collection_name": "test_metadata_filtering",
                "dimension": 384,
            },
            "filters": {
                "conditions": [
                    {"field": "source", "value": "wiki", "operator": "equals"}
                ]
            },
            "rag": {"enabled": False},
        }

        pipeline = QdrantMetadataFilteringSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        mock_filter.assert_called_once()

    @patch("vectordb.langchain.metadata_filtering.search.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.metadata_filtering.search.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.search.qdrant.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.metadata_filtering.search.qdrant.RAGHelper.create_llm")
    @patch("vectordb.langchain.metadata_filtering.search.qdrant.RAGHelper.generate")
    def test_search_with_rag(
        self,
        mock_rag_generate,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test search with RAG generation."""
        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst

        mock_llm = MagicMock()
        mock_llm_helper.return_value = mock_llm
        mock_rag_generate.return_value = "Generated answer"

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "",
                "collection_name": "test_metadata_filtering",
                "dimension": 384,
            },
            "rag": {"enabled": True, "model": "test-llm"},
        }

        pipeline = QdrantMetadataFilteringSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert "answer" in result
        assert result["answer"] == "Generated answer"
