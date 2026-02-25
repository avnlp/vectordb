"""Tests for Pinecone metadata filtering pipelines (LangChain)."""

from unittest.mock import MagicMock, patch

from vectordb.langchain.metadata_filtering.indexing.pinecone import (
    PineconeMetadataFilteringIndexingPipeline,
)
from vectordb.langchain.metadata_filtering.search.pinecone import (
    PineconeMetadataFilteringSearchPipeline,
)


class TestPineconeMetadataFilteringIndexing:
    """Unit tests for Pinecone metadata filtering indexing pipeline."""

    @patch("vectordb.langchain.metadata_filtering.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.metadata_filtering.indexing.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.indexing.pinecone.DataloaderCatalog.create"
    )
    def test_indexing_initialization(
        self,
        mock_get_docs,
        mock_embedder_helper,
        mock_db,
        pinecone_metadata_filtering_config,
    ):
        """Test pipeline initialization."""
        pipeline = PineconeMetadataFilteringIndexingPipeline(
            pinecone_metadata_filtering_config
        )
        assert pipeline.config == pinecone_metadata_filtering_config
        assert pipeline.index_name == "test-metadata-filtering"
        assert pipeline.namespace == "test-namespace"

    @patch("vectordb.langchain.metadata_filtering.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.metadata_filtering.indexing.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.indexing.pinecone.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.indexing.pinecone.DataloaderCatalog.create"
    )
    def test_indexing_run_with_documents(
        self,
        mock_get_docs,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
        sample_langchain_documents,
    ):
        """Test indexing with documents."""
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_langchain_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader
        mock_embed_docs.return_value = (
            sample_langchain_documents,
            [[0.1] * 384] * 5,
        )

        mock_db_inst = MagicMock()
        mock_db_inst.upsert.return_value = len(sample_langchain_documents)
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "test-metadata-filtering",
                "namespace": "test-namespace",
                "dimension": 384,
                "metric": "cosine",
            },
        }

        pipeline = PineconeMetadataFilteringIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_langchain_documents)
        mock_db_inst.create_index.assert_called_once()
        mock_db_inst.upsert.assert_called_once()

    @patch("vectordb.langchain.metadata_filtering.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.metadata_filtering.indexing.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.indexing.pinecone.DataloaderCatalog.create"
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
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "test-metadata-filtering",
                "namespace": "test-namespace",
                "dimension": 384,
                "metric": "cosine",
            },
        }

        pipeline = PineconeMetadataFilteringIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0

    @patch("vectordb.langchain.metadata_filtering.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.metadata_filtering.indexing.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.indexing.pinecone.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.indexing.pinecone.DataloaderCatalog.create"
    )
    def test_indexing_with_recreate(
        self,
        mock_get_docs,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
        sample_langchain_documents,
    ):
        """Test indexing with index recreation."""
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_langchain_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader
        mock_embed_docs.return_value = (
            sample_langchain_documents,
            [[0.1] * 384] * 5,
        )

        mock_db_inst = MagicMock()
        mock_db_inst.upsert.return_value = len(sample_langchain_documents)
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "test-metadata-filtering",
                "namespace": "test-namespace",
                "dimension": 384,
                "metric": "cosine",
                "recreate": True,
            },
        }

        pipeline = PineconeMetadataFilteringIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_langchain_documents)
        mock_db_inst.create_index.assert_called_once()
        call_kwargs = mock_db_inst.create_index.call_args.kwargs
        assert call_kwargs["recreate"] is True


class TestPineconeMetadataFilteringSearch:
    """Unit tests for Pinecone metadata filtering search pipeline."""

    @patch("vectordb.langchain.metadata_filtering.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.metadata_filtering.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.metadata_filtering.search.pinecone.RAGHelper.create_llm")
    def test_search_initialization(
        self,
        mock_llm_helper,
        mock_embedder_helper,
        mock_db,
        pinecone_metadata_filtering_config,
    ):
        """Test search pipeline initialization."""
        mock_llm_helper.return_value = None

        pipeline = PineconeMetadataFilteringSearchPipeline(
            pinecone_metadata_filtering_config
        )
        assert pipeline.config == pinecone_metadata_filtering_config
        assert pipeline.index_name == "test-metadata-filtering"
        assert pipeline.namespace == "test-namespace"

    @patch("vectordb.langchain.metadata_filtering.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.metadata_filtering.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.search.pinecone.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.metadata_filtering.search.pinecone.RAGHelper.create_llm")
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
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "test-metadata-filtering",
                "namespace": "test-namespace",
                "dimension": 384,
                "metric": "cosine",
            },
            "rag": {"enabled": False},
        }

        pipeline = PineconeMetadataFilteringSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert len(result["documents"]) > 0

    @patch("vectordb.langchain.metadata_filtering.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.metadata_filtering.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.search.pinecone.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.metadata_filtering.search.pinecone.RAGHelper.create_llm")
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
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "test-metadata-filtering",
                "namespace": "test-namespace",
                "dimension": 384,
                "metric": "cosine",
            },
            "rag": {"enabled": False},
        }

        pipeline = PineconeMetadataFilteringSearchPipeline(config)
        filters = {"source": "wiki"}
        result = pipeline.search("test query", top_k=5, filters=filters)

        assert result["query"] == "test query"
        mock_db_inst.query.assert_called_once()
        call_kwargs = mock_db_inst.query.call_args.kwargs
        assert call_kwargs["filter"] == filters
        assert call_kwargs["namespace"] == "test-namespace"

    @patch("vectordb.langchain.metadata_filtering.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.metadata_filtering.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.search.pinecone.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.metadata_filtering.search.pinecone.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.metadata_filtering.search.pinecone.DocumentFilter.filter_by_metadata"
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
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "test-metadata-filtering",
                "namespace": "test-namespace",
                "dimension": 384,
                "metric": "cosine",
            },
            "filters": {
                "conditions": [
                    {"field": "source", "value": "wiki", "operator": "equals"}
                ]
            },
            "rag": {"enabled": False},
        }

        pipeline = PineconeMetadataFilteringSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        mock_filter.assert_called_once()

    @patch("vectordb.langchain.metadata_filtering.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.metadata_filtering.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.search.pinecone.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.metadata_filtering.search.pinecone.RAGHelper.create_llm")
    @patch("vectordb.langchain.metadata_filtering.search.pinecone.RAGHelper.generate")
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
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "test-metadata-filtering",
                "namespace": "test-namespace",
                "dimension": 384,
                "metric": "cosine",
            },
            "rag": {"enabled": True, "model": "test-llm"},
        }

        pipeline = PineconeMetadataFilteringSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert "answer" in result
        assert result["answer"] == "Generated answer"
