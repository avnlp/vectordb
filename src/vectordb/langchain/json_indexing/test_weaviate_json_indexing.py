"""Tests for Weaviate JSON indexing pipelines (LangChain)."""

from unittest.mock import MagicMock, patch


class TestWeaviateJSONIndexing:
    """Unit tests for Weaviate JSON indexing pipeline."""

    @patch("vectordb.langchain.json_indexing.indexing.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.indexing.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.json_indexing.indexing.weaviate.DataloaderCatalog.create"
    )
    def test_indexing_initialization(
        self, mock_get_docs, mock_embedder_helper, mock_db
    ):
        """Test pipeline initialization."""
        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "",
                "collection_name": "TestJsonIndexing",
            },
        }

        from vectordb.langchain.json_indexing.indexing.weaviate import (
            WeaviateJsonIndexingPipeline,
        )

        pipeline = WeaviateJsonIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.collection_name == "TestJsonIndexing"

    @patch("vectordb.langchain.json_indexing.indexing.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.indexing.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.json_indexing.indexing.weaviate.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.json_indexing.indexing.weaviate.DataloaderCatalog.create"
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
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "",
                "collection_name": "TestJsonIndexing",
            },
        }

        from vectordb.langchain.json_indexing.indexing.weaviate import (
            WeaviateJsonIndexingPipeline,
        )

        pipeline = WeaviateJsonIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.upsert.assert_called_once()

    @patch("vectordb.langchain.json_indexing.indexing.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.indexing.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.json_indexing.indexing.weaviate.DataloaderCatalog.create"
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
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "",
                "collection_name": "TestJsonIndexing",
            },
        }

        from vectordb.langchain.json_indexing.indexing.weaviate import (
            WeaviateJsonIndexingPipeline,
        )

        pipeline = WeaviateJsonIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0

    @patch("vectordb.langchain.json_indexing.indexing.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.indexing.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.json_indexing.indexing.weaviate.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.json_indexing.indexing.weaviate.DataloaderCatalog.create"
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
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "",
                "collection_name": "TestJsonIndexing",
                "recreate": True,
            },
        }

        from vectordb.langchain.json_indexing.indexing.weaviate import (
            WeaviateJsonIndexingPipeline,
        )

        pipeline = WeaviateJsonIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.delete_collection.assert_called_once_with("TestJsonIndexing")
        mock_db_inst.upsert.assert_called_once()


class TestWeaviateJSONSearch:
    """Unit tests for Weaviate JSON search pipeline."""

    @patch("vectordb.langchain.json_indexing.search.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.search.weaviate.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.json_indexing.search.weaviate.RAGHelper.create_llm")
    def test_search_initialization(
        self, mock_llm_helper, mock_embedder_helper, mock_db
    ):
        """Test search pipeline initialization."""
        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "",
                "collection_name": "TestJsonIndexing",
            },
            "rag": {"enabled": False},
        }

        from vectordb.langchain.json_indexing.search.weaviate import (
            WeaviateJsonSearchPipeline,
        )

        pipeline = WeaviateJsonSearchPipeline(config)
        assert pipeline.config == config

    @patch("vectordb.langchain.json_indexing.search.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.search.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.json_indexing.search.weaviate.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.json_indexing.search.weaviate.RAGHelper.create_llm")
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
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "",
                "collection_name": "TestJsonIndexing",
            },
            "rag": {"enabled": False},
        }

        from vectordb.langchain.json_indexing.search.weaviate import (
            WeaviateJsonSearchPipeline,
        )

        pipeline = WeaviateJsonSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert len(result["documents"]) > 0

    @patch("vectordb.langchain.json_indexing.search.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.search.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.json_indexing.search.weaviate.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.json_indexing.search.weaviate.RAGHelper.create_llm")
    @patch("vectordb.langchain.json_indexing.search.weaviate.RAGHelper.generate")
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

        mock_llm_inst = MagicMock()
        mock_llm_helper.return_value = mock_llm_inst
        mock_rag_generate.return_value = "Test answer"

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "",
                "collection_name": "TestJsonIndexing",
            },
            "rag": {"enabled": True},
        }

        from vectordb.langchain.json_indexing.search.weaviate import (
            WeaviateJsonSearchPipeline,
        )

        pipeline = WeaviateJsonSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert result["answer"] == "Test answer"

    @patch("vectordb.langchain.json_indexing.search.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.search.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.json_indexing.search.weaviate.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.json_indexing.search.weaviate.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.json_indexing.search.weaviate.DocumentFilter.filter_by_metadata_json"
    )
    def test_search_with_json_filters(
        self,
        mock_filter,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test search with JSON filters."""
        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None
        mock_filter.return_value = sample_documents[:2]

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "",
                "collection_name": "TestJsonIndexing",
            },
            "rag": {"enabled": False},
            "filters": {
                "conditions": [
                    {
                        "field": "metadata.category",
                        "value": "tech",
                        "operator": "equals",
                    }
                ]
            },
        }

        from vectordb.langchain.json_indexing.search.weaviate import (
            WeaviateJsonSearchPipeline,
        )

        pipeline = WeaviateJsonSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        mock_filter.assert_called()

    @patch("vectordb.langchain.json_indexing.search.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.search.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.json_indexing.search.weaviate.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.json_indexing.search.weaviate.RAGHelper.create_llm")
    def test_search_with_filters_param(
        self,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test search with filters parameter."""
        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "",
                "collection_name": "TestJsonIndexing",
            },
            "rag": {"enabled": False},
        }

        from vectordb.langchain.json_indexing.search.weaviate import (
            WeaviateJsonSearchPipeline,
        )

        pipeline = WeaviateJsonSearchPipeline(config)
        filters = {"source": "wiki"}
        result = pipeline.search("test query", top_k=5, filters=filters)

        assert result["query"] == "test query"
        mock_db_inst.query.assert_called_once()
        call_kwargs = mock_db_inst.query.call_args.kwargs
        assert call_kwargs["filters"] == filters
