"""Tests for Weaviate contextual compression pipelines (LangChain)."""

from unittest.mock import MagicMock, patch

from vectordb.langchain.contextual_compression.indexing.weaviate import (
    WeaviateContextualCompressionIndexingPipeline,
)
from vectordb.langchain.contextual_compression.search.weaviate import (
    WeaviateContextualCompressionSearchPipeline,
)


class TestWeaviateContextualCompressionIndexing:
    """Unit tests for Weaviate contextual compression indexing pipeline."""

    @patch(
        "vectordb.langchain.contextual_compression.indexing.weaviate.WeaviateVectorDB"
    )
    @patch(
        "vectordb.langchain.contextual_compression.indexing.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.contextual_compression.indexing.weaviate.DataloaderCatalog.create"
    )
    def test_indexing_initialization(
        self,
        mock_get_docs,
        mock_embedder_helper,
        mock_db,
        weaviate_contextual_compression_config,
    ):
        """Test pipeline initialization."""
        pipeline = WeaviateContextualCompressionIndexingPipeline(
            weaviate_contextual_compression_config
        )
        assert pipeline.config == weaviate_contextual_compression_config
        assert pipeline.collection_name == "TestContextualCompression"

    @patch(
        "vectordb.langchain.contextual_compression.indexing.weaviate.WeaviateVectorDB"
    )
    @patch(
        "vectordb.langchain.contextual_compression.indexing.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.contextual_compression.indexing.weaviate.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.contextual_compression.indexing.weaviate.DataloaderCatalog.create"
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
                "collection_name": "TestContextualCompression",
            },
        }

        pipeline = WeaviateContextualCompressionIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.upsert.assert_called_once()

    @patch(
        "vectordb.langchain.contextual_compression.indexing.weaviate.WeaviateVectorDB"
    )
    @patch(
        "vectordb.langchain.contextual_compression.indexing.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.contextual_compression.indexing.weaviate.DataloaderCatalog.create"
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
                "collection_name": "TestContextualCompression",
            },
        }

        pipeline = WeaviateContextualCompressionIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0

    @patch(
        "vectordb.langchain.contextual_compression.indexing.weaviate.WeaviateVectorDB"
    )
    @patch(
        "vectordb.langchain.contextual_compression.indexing.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.contextual_compression.indexing.weaviate.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.contextual_compression.indexing.weaviate.DataloaderCatalog.create"
    )
    def test_indexing_with_recreate_option(
        self,
        mock_get_docs,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test indexing with recreate option."""
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
                "collection_name": "TestContextualCompression",
                "recreate": True,
            },
        }

        pipeline = WeaviateContextualCompressionIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.create_collection.assert_called_once()


class TestWeaviateContextualCompressionSearch:
    """Unit tests for Weaviate contextual compression search pipeline."""

    @patch("vectordb.langchain.contextual_compression.search.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.contextual_compression.search.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.contextual_compression.search.weaviate.RAGHelper.create_llm"
    )
    @patch(
        "vectordb.langchain.contextual_compression.search.weaviate.RerankerHelper.create_reranker"
    )
    def test_search_initialization(
        self, mock_reranker_helper, mock_llm_helper, mock_embedder_helper, mock_db
    ):
        """Test search pipeline initialization."""
        mock_llm_helper.return_value = None
        mock_reranker_helper.return_value = MagicMock()

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "",
                "collection_name": "TestContextualCompression",
            },
            "rag": {"enabled": False},
        }

        pipeline = WeaviateContextualCompressionSearchPipeline(config)
        assert pipeline.config == config

    @patch("vectordb.langchain.contextual_compression.search.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.contextual_compression.search.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.contextual_compression.search.weaviate.EmbedderHelper.embed_query"
    )
    @patch(
        "vectordb.langchain.contextual_compression.search.weaviate.RAGHelper.create_llm"
    )
    @patch(
        "vectordb.langchain.contextual_compression.search.weaviate.RerankerHelper.create_reranker"
    )
    def test_search_execution(
        self,
        mock_reranker_helper,
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

        # Mock the reranker to return scores for each document
        mock_reranker = MagicMock()
        mock_reranker.rank.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]
        mock_reranker_helper.return_value = mock_reranker

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "",
                "collection_name": "TestContextualCompression",
            },
            "rag": {"enabled": False},
        }

        pipeline = WeaviateContextualCompressionSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert len(result["documents"]) > 0

    @patch("vectordb.langchain.contextual_compression.search.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.contextual_compression.search.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.contextual_compression.search.weaviate.EmbedderHelper.embed_query"
    )
    @patch(
        "vectordb.langchain.contextual_compression.search.weaviate.RAGHelper.create_llm"
    )
    @patch(
        "vectordb.langchain.contextual_compression.search.weaviate.RerankerHelper.create_reranker"
    )
    @patch(
        "vectordb.langchain.contextual_compression.search.weaviate.RAGHelper.generate"
    )
    def test_search_with_rag(
        self,
        mock_rag_generate,
        mock_reranker_helper,
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

        # Mock the reranker to return scores for each document
        mock_reranker = MagicMock()
        mock_reranker.rank.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]
        mock_reranker_helper.return_value = mock_reranker

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "",
                "collection_name": "TestContextualCompression",
            },
            "rag": {"enabled": True},
        }

        pipeline = WeaviateContextualCompressionSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert result["answer"] == "Test answer"
