"""Tests for Milvus query enhancement pipelines (LangChain)."""

from unittest.mock import MagicMock, patch

from vectordb.langchain.query_enhancement.indexing.milvus import (
    MilvusQueryEnhancementIndexingPipeline,
)
from vectordb.langchain.query_enhancement.search.milvus import (
    MilvusQueryEnhancementSearchPipeline,
)


class TestMilvusQueryEnhancementIndexing:
    """Unit tests for Milvus query enhancement indexing pipeline."""

    @patch("vectordb.langchain.query_enhancement.indexing.milvus.MilvusVectorDB")
    @patch(
        "vectordb.langchain.query_enhancement.indexing.milvus.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.query_enhancement.indexing.milvus.DataloaderCatalog.create"
    )
    def test_indexing_initialization(
        self,
        mock_get_docs,
        mock_embedder_helper,
        mock_db,
        milvus_query_enhancement_config,
    ):
        """Test pipeline initialization."""
        pipeline = MilvusQueryEnhancementIndexingPipeline(
            milvus_query_enhancement_config
        )
        assert pipeline.config == milvus_query_enhancement_config
        assert pipeline.collection_name == "test_query_enhancement"

    @patch("vectordb.langchain.query_enhancement.indexing.milvus.MilvusVectorDB")
    @patch(
        "vectordb.langchain.query_enhancement.indexing.milvus.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.query_enhancement.indexing.milvus.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.query_enhancement.indexing.milvus.DataloaderCatalog.create"
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
            "milvus": {
                "host": "localhost",
                "port": 19530,
                "collection_name": "test_query_enhancement",
                "dimension": 384,
            },
        }

        pipeline = MilvusQueryEnhancementIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.upsert.assert_called_once()

    @patch("vectordb.langchain.query_enhancement.indexing.milvus.MilvusVectorDB")
    @patch(
        "vectordb.langchain.query_enhancement.indexing.milvus.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.query_enhancement.indexing.milvus.DataloaderCatalog.create"
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
            "milvus": {
                "host": "localhost",
                "port": 19530,
                "collection_name": "test_query_enhancement",
                "dimension": 384,
            },
        }

        pipeline = MilvusQueryEnhancementIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0


class TestMilvusQueryEnhancementSearch:
    """Unit tests for Milvus query enhancement search pipeline."""

    @patch("langchain_groq.ChatGroq")
    @patch("vectordb.langchain.query_enhancement.search.milvus.MilvusVectorDB")
    @patch(
        "vectordb.langchain.query_enhancement.search.milvus.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.query_enhancement.search.milvus.RAGHelper.create_llm")
    @patch("vectordb.langchain.query_enhancement.search.milvus.QueryEnhancer")
    def test_search_initialization(
        self, mock_enhancer, mock_llm_helper, mock_embedder_helper, mock_db, mock_llm
    ):
        """Test search pipeline initialization."""
        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "milvus": {
                "host": "localhost",
                "port": 19530,
                "collection_name": "test_query_enhancement",
                "dimension": 384,
            },
            "rag": {"enabled": False},
        }

        pipeline = MilvusQueryEnhancementSearchPipeline(config)
        assert pipeline.config == config

    @patch("langchain_groq.ChatGroq")
    @patch("vectordb.langchain.query_enhancement.search.milvus.MilvusVectorDB")
    @patch(
        "vectordb.langchain.query_enhancement.search.milvus.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.query_enhancement.search.milvus.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.query_enhancement.search.milvus.RAGHelper.create_llm")
    @patch("vectordb.langchain.query_enhancement.search.milvus.QueryEnhancer")
    def test_search_execution(
        self,
        mock_enhancer,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
        mock_llm,
        sample_documents,
    ):
        """Test search execution."""
        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        mock_enhancer_inst = MagicMock()
        mock_enhancer_inst.generate_queries.return_value = ["test query"]
        mock_enhancer.return_value = mock_enhancer_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "milvus": {
                "host": "localhost",
                "port": 19530,
                "collection_name": "test_query_enhancement",
                "dimension": 384,
            },
            "rag": {"enabled": False},
        }

        pipeline = MilvusQueryEnhancementSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert len(result["documents"]) > 0

    @patch("langchain_groq.ChatGroq")
    @patch("vectordb.langchain.query_enhancement.search.milvus.MilvusVectorDB")
    @patch(
        "vectordb.langchain.query_enhancement.search.milvus.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.query_enhancement.search.milvus.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.query_enhancement.search.milvus.RAGHelper.create_llm")
    @patch("vectordb.langchain.query_enhancement.search.milvus.QueryEnhancer")
    def test_search_with_different_modes(
        self,
        mock_enhancer,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
        mock_llm,
        sample_documents,
    ):
        """Test search with different query enhancement modes."""
        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        mock_enhancer_inst = MagicMock()
        mock_enhancer_inst.generate_queries.return_value = ["query 1", "query 2"]
        mock_enhancer.return_value = mock_enhancer_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "milvus": {
                "host": "localhost",
                "port": 19530,
                "collection_name": "test_query_enhancement",
                "dimension": 384,
            },
            "rag": {"enabled": False},
        }

        pipeline = MilvusQueryEnhancementSearchPipeline(config)

        # Test multi_query mode
        result = pipeline.search("test query", top_k=5, mode="multi_query")
        assert "documents" in result
        assert "enhanced_queries" in result

        # Test hyde mode
        result = pipeline.search("test query", top_k=5, mode="hyde")
        assert "documents" in result

        # Test step_back mode
        result = pipeline.search("test query", top_k=5, mode="step_back")
        assert "documents" in result
