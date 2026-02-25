"""Tests for Pinecone JSON indexing pipelines (LangChain)."""

from unittest.mock import MagicMock, patch


class TestPineconeJSONIndexing:
    """Unit tests for Pinecone JSON indexing pipeline."""

    @patch("vectordb.langchain.json_indexing.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.indexing.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.json_indexing.indexing.pinecone.DataloaderCatalog.create"
    )
    def test_indexing_initialization(
        self, mock_get_docs, mock_embedder_helper, mock_db
    ):
        """Test pipeline initialization."""
        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-json-index",
                "namespace": "test",
                "dimension": 384,
                "metric": "cosine",
            },
        }

        from vectordb.langchain.json_indexing.indexing.pinecone import (
            PineconeJsonIndexingPipeline,
        )

        pipeline = PineconeJsonIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.index_name == "test-json-index"
        assert pipeline.namespace == "test"
        assert pipeline.dimension == 384

    @patch("vectordb.langchain.json_indexing.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.indexing.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.json_indexing.indexing.pinecone.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.json_indexing.indexing.pinecone.DataloaderCatalog.create"
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
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-json-index",
                "namespace": "test",
                "dimension": 384,
                "metric": "cosine",
            },
        }

        from vectordb.langchain.json_indexing.indexing.pinecone import (
            PineconeJsonIndexingPipeline,
        )

        pipeline = PineconeJsonIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.create_index.assert_called_once()
        mock_db_inst.upsert.assert_called_once()

    @patch("vectordb.langchain.json_indexing.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.indexing.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.json_indexing.indexing.pinecone.DataloaderCatalog.create"
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
                "api_key": "test-key",
                "index_name": "test-json-index",
                "namespace": "test",
                "dimension": 384,
                "metric": "cosine",
            },
        }

        from vectordb.langchain.json_indexing.indexing.pinecone import (
            PineconeJsonIndexingPipeline,
        )

        pipeline = PineconeJsonIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0

    @patch("vectordb.langchain.json_indexing.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.indexing.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.json_indexing.indexing.pinecone.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.json_indexing.indexing.pinecone.DataloaderCatalog.create"
    )
    def test_indexing_with_recreate(
        self,
        mock_get_docs,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test indexing with index recreation."""
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
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-json-index",
                "namespace": "test",
                "dimension": 384,
                "metric": "cosine",
                "recreate": True,
            },
        }

        from vectordb.langchain.json_indexing.indexing.pinecone import (
            PineconeJsonIndexingPipeline,
        )

        pipeline = PineconeJsonIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.create_index.assert_called_once_with(
            index_name="test-json-index",
            dimension=384,
            metric="cosine",
            recreate=True,
        )
        mock_db_inst.upsert.assert_called_once()


class TestPineconeJSONSearch:
    """Unit tests for Pinecone JSON search pipeline."""

    @patch("vectordb.langchain.json_indexing.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.json_indexing.search.pinecone.RAGHelper.create_llm")
    def test_search_initialization(
        self, mock_llm_helper, mock_embedder_helper, mock_db
    ):
        """Test search pipeline initialization."""
        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-json-index",
                "namespace": "test",
            },
            "rag": {"enabled": False},
        }

        from vectordb.langchain.json_indexing.search.pinecone import (
            PineconeJsonSearchPipeline,
        )

        pipeline = PineconeJsonSearchPipeline(config)
        assert pipeline.config == config
        assert pipeline.namespace == "test"

    @patch("vectordb.langchain.json_indexing.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.json_indexing.search.pinecone.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.json_indexing.search.pinecone.RAGHelper.create_llm")
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
                "api_key": "test-key",
                "index_name": "test-json-index",
                "namespace": "test",
            },
            "rag": {"enabled": False},
        }

        from vectordb.langchain.json_indexing.search.pinecone import (
            PineconeJsonSearchPipeline,
        )

        pipeline = PineconeJsonSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert len(result["documents"]) > 0
        mock_db_inst.query.assert_called_once()
        call_kwargs = mock_db_inst.query.call_args.kwargs
        assert call_kwargs["namespace"] == "test"

    @patch("vectordb.langchain.json_indexing.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.json_indexing.search.pinecone.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.json_indexing.search.pinecone.RAGHelper.create_llm")
    @patch("vectordb.langchain.json_indexing.search.pinecone.RAGHelper.generate")
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
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-json-index",
                "namespace": "test",
            },
            "rag": {"enabled": True},
        }

        from vectordb.langchain.json_indexing.search.pinecone import (
            PineconeJsonSearchPipeline,
        )

        pipeline = PineconeJsonSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert result["answer"] == "Test answer"

    @patch("vectordb.langchain.json_indexing.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.json_indexing.search.pinecone.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.json_indexing.search.pinecone.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.json_indexing.search.pinecone.DocumentFilter.filter_by_metadata_json"
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
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-json-index",
                "namespace": "test",
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

        from vectordb.langchain.json_indexing.search.pinecone import (
            PineconeJsonSearchPipeline,
        )

        pipeline = PineconeJsonSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        mock_filter.assert_called()

    @patch("vectordb.langchain.json_indexing.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.json_indexing.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.json_indexing.search.pinecone.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.json_indexing.search.pinecone.RAGHelper.create_llm")
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
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-json-index",
                "namespace": "test",
            },
            "rag": {"enabled": False},
        }

        from vectordb.langchain.json_indexing.search.pinecone import (
            PineconeJsonSearchPipeline,
        )

        pipeline = PineconeJsonSearchPipeline(config)
        filters = {"source": "wiki"}
        result = pipeline.search("test query", top_k=5, filters=filters)

        assert result["query"] == "test query"
        mock_db_inst.query.assert_called_once()
        call_kwargs = mock_db_inst.query.call_args.kwargs
        assert call_kwargs["filters"] == filters
