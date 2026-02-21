"""Tests for Chroma hybrid indexing pipelines (LangChain)."""

from unittest.mock import MagicMock, patch

from vectordb.langchain.hybrid_indexing.indexing.chroma import (
    ChromaHybridIndexingPipeline,
)
from vectordb.langchain.hybrid_indexing.search.chroma import (
    ChromaHybridSearchPipeline,
)


class TestChromaHybridIndexing:
    """Unit tests for Chroma hybrid indexing pipeline."""

    @patch("vectordb.langchain.hybrid_indexing.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.hybrid_indexing.indexing.chroma.SparseEmbedder")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.chroma.DataloaderCatalog.create"
    )
    def test_indexing_initialization(
        self, mock_get_docs, mock_sparse_embedder_cls, mock_embedder_helper, mock_db
    ):
        """Test pipeline initialization."""
        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_hybrid_indexing",
            },
            "sparse": {
                "model": "bm25",
            },
        }

        pipeline = ChromaHybridIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.collection_name == "test_hybrid_indexing"

    @patch("vectordb.langchain.hybrid_indexing.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.chroma.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.hybrid_indexing.indexing.chroma.SparseEmbedder")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.chroma.DataloaderCatalog.create"
    )
    def test_indexing_run_with_documents(
        self,
        mock_get_docs,
        mock_sparse_embedder_cls,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test indexing with documents."""
        # Mock chain: create() -> load() -> to_langchain()
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader

        mock_embed_docs.return_value = (sample_documents, [[0.1] * 384] * 5)

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder.embed_documents.return_value = [
            [0.5, 0.3, 0.2] for _ in range(5)
        ]
        mock_sparse_embedder_cls.return_value = mock_sparse_embedder

        mock_db_inst = MagicMock()
        mock_db_inst.upsert.return_value = len(sample_documents)
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_hybrid_indexing",
            },
            "sparse": {
                "model": "bm25",
            },
        }

        pipeline = ChromaHybridIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)

    @patch("vectordb.langchain.hybrid_indexing.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.hybrid_indexing.indexing.chroma.SparseEmbedder")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.chroma.DataloaderCatalog.create"
    )
    def test_indexing_run_no_documents(
        self, mock_get_docs, mock_sparse_embedder_cls, mock_embedder_helper, mock_db
    ):
        """Test indexing with no documents."""
        # Mock chain: create() -> load() -> to_langchain()
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = []
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_hybrid_indexing",
            },
        }

        pipeline = ChromaHybridIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0


class TestChromaHybridSearch:
    """Unit tests for Chroma hybrid search pipeline."""

    @patch("vectordb.langchain.hybrid_indexing.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.hybrid_indexing.search.chroma.RAGHelper.create_llm")
    @patch("vectordb.langchain.hybrid_indexing.search.chroma.SparseEmbedder")
    def test_search_initialization(
        self, mock_sparse_embedder, mock_llm_helper, mock_embedder_helper, mock_db
    ):
        """Test search pipeline initialization."""
        mock_sparse_embedder.return_value = MagicMock()
        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_hybrid_indexing",
            },
            "rag": {"enabled": False},
        }

        pipeline = ChromaHybridSearchPipeline(config)
        assert pipeline.config == config

    @patch("vectordb.langchain.hybrid_indexing.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.hybrid_indexing.search.chroma.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.hybrid_indexing.search.chroma.RAGHelper.create_llm")
    @patch("vectordb.langchain.hybrid_indexing.search.chroma.SparseEmbedder")
    def test_search_execution(
        self,
        mock_sparse_embedder,
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

        mock_sparse_embedder.return_value = MagicMock()

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_hybrid_indexing",
            },
            "rag": {"enabled": False},
        }

        pipeline = ChromaHybridSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert len(result["documents"]) > 0
