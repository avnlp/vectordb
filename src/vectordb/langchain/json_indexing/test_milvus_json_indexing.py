"""Tests for Milvus JSON indexing pipelines (LangChain)."""

from unittest.mock import MagicMock, patch


class TestMilvusJSONIndexing:
    """Unit tests for Milvus JSON indexing pipeline."""

    @patch("vectordb.langchain.json_indexing.indexing.milvus.MilvusVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.indexing.milvus.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.json_indexing.indexing.milvus.DataloaderCatalog.create")
    def test_indexing_initialization(
        self, mock_get_docs, mock_embedder_helper, mock_db
    ):
        """Test pipeline initialization."""
        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "milvus": {
                "host": "localhost",
                "port": 19530,
                "collection_name": "test_json_indexing",
            },
        }

        from vectordb.langchain.json_indexing.indexing.milvus import (
            MilvusJsonIndexingPipeline,
        )

        pipeline = MilvusJsonIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.collection_name == "test_json_indexing"

    @patch("vectordb.langchain.json_indexing.indexing.milvus.MilvusVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.indexing.milvus.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.json_indexing.indexing.milvus.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.json_indexing.indexing.milvus.DataloaderCatalog.create")
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
            "milvus": {
                "host": "localhost",
                "port": 19530,
                "collection_name": "test_json_indexing",
            },
        }

        from vectordb.langchain.json_indexing.indexing.milvus import (
            MilvusJsonIndexingPipeline,
        )

        pipeline = MilvusJsonIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.upsert.assert_called_once()

    @patch("vectordb.langchain.json_indexing.indexing.milvus.MilvusVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.indexing.milvus.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.json_indexing.indexing.milvus.DataloaderCatalog.create")
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
            "milvus": {
                "host": "localhost",
                "port": 19530,
                "collection_name": "test_json_indexing",
            },
        }

        from vectordb.langchain.json_indexing.indexing.milvus import (
            MilvusJsonIndexingPipeline,
        )

        pipeline = MilvusJsonIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0

    @patch("vectordb.langchain.json_indexing.indexing.milvus.MilvusVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.indexing.milvus.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.json_indexing.indexing.milvus.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.json_indexing.indexing.milvus.DataloaderCatalog.create")
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
            "milvus": {
                "host": "localhost",
                "port": 19530,
                "collection_name": "test_json_indexing",
                "recreate": True,
            },
        }

        from vectordb.langchain.json_indexing.indexing.milvus import (
            MilvusJsonIndexingPipeline,
        )

        pipeline = MilvusJsonIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.delete_collection.assert_called_once_with("test_json_indexing")
        mock_db_inst.upsert.assert_called_once()


class TestMilvusJSONSearch:
    """Unit tests for Milvus JSON search pipeline."""

    @patch("vectordb.langchain.json_indexing.search.milvus.MilvusVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.search.milvus.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.json_indexing.search.milvus.RAGHelper.create_llm")
    def test_search_initialization(
        self, mock_llm_helper, mock_embedder_helper, mock_db
    ):
        """Test search pipeline initialization."""
        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "milvus": {
                "host": "localhost",
                "port": 19530,
                "collection_name": "test_json_indexing",
            },
            "rag": {"enabled": False},
        }

        from vectordb.langchain.json_indexing.search.milvus import (
            MilvusJsonSearchPipeline,
        )

        pipeline = MilvusJsonSearchPipeline(config)
        assert pipeline.config == config

    @patch("vectordb.langchain.json_indexing.search.milvus.MilvusVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.search.milvus.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.json_indexing.search.milvus.EmbedderHelper.embed_query")
    @patch("vectordb.langchain.json_indexing.search.milvus.RAGHelper.create_llm")
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
            "milvus": {
                "host": "localhost",
                "port": 19530,
                "collection_name": "test_json_indexing",
            },
            "rag": {"enabled": False},
        }

        from vectordb.langchain.json_indexing.search.milvus import (
            MilvusJsonSearchPipeline,
        )

        pipeline = MilvusJsonSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert len(result["documents"]) > 0

    @patch("vectordb.langchain.json_indexing.search.milvus.MilvusVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.search.milvus.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.json_indexing.search.milvus.EmbedderHelper.embed_query")
    @patch("vectordb.langchain.json_indexing.search.milvus.RAGHelper.create_llm")
    @patch("vectordb.langchain.json_indexing.search.milvus.RAGHelper.generate")
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
            "milvus": {
                "host": "localhost",
                "port": 19530,
                "collection_name": "test_json_indexing",
            },
            "rag": {"enabled": True},
        }

        from vectordb.langchain.json_indexing.search.milvus import (
            MilvusJsonSearchPipeline,
        )

        pipeline = MilvusJsonSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert result["answer"] == "Test answer"

    @patch("vectordb.langchain.json_indexing.search.milvus.MilvusVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.search.milvus.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.json_indexing.search.milvus.EmbedderHelper.embed_query")
    @patch("vectordb.langchain.json_indexing.search.milvus.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.json_indexing.search.milvus.DocumentFilter.filter_by_metadata_json"
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
            "milvus": {
                "host": "localhost",
                "port": 19530,
                "collection_name": "test_json_indexing",
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

        from vectordb.langchain.json_indexing.search.milvus import (
            MilvusJsonSearchPipeline,
        )

        pipeline = MilvusJsonSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        mock_filter.assert_called()

    @patch("vectordb.langchain.json_indexing.search.milvus.MilvusVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.search.milvus.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.json_indexing.search.milvus.EmbedderHelper.embed_query")
    @patch("vectordb.langchain.json_indexing.search.milvus.RAGHelper.create_llm")
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
            "milvus": {
                "host": "localhost",
                "port": 19530,
                "collection_name": "test_json_indexing",
            },
            "rag": {"enabled": False},
        }

        from vectordb.langchain.json_indexing.search.milvus import (
            MilvusJsonSearchPipeline,
        )

        pipeline = MilvusJsonSearchPipeline(config)
        filters = {"source": "wiki"}
        result = pipeline.search("test query", top_k=5, filters=filters)

        assert result["query"] == "test query"
        mock_db_inst.query.assert_called_once()
        call_kwargs = mock_db_inst.query.call_args.kwargs
        assert call_kwargs["filters"] == filters
