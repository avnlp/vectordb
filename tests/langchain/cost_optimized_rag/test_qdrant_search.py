"""Tests for Qdrant cost-optimized RAG search pipeline (LangChain)."""

from unittest.mock import MagicMock, patch

from vectordb.langchain.cost_optimized_rag.search.qdrant import (
    QdrantCostOptimizedRAGSearchPipeline,
)


class TestQdrantCostOptimizedRAGSearchPipeline:
    """Tests for QdrantCostOptimizedRAGSearchPipeline."""

    @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.qdrant.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.SparseEmbedder")
    def test_init_with_valid_config(
        self, mock_sparse_cls, mock_embedder_helper, mock_db_cls, qdrant_config
    ):
        """Test initialization with valid config dict."""
        mock_embedder = MagicMock()
        mock_embedder_helper.return_value = mock_embedder

        mock_sparse_embedder = MagicMock()
        mock_sparse_cls.return_value = mock_sparse_embedder

        mock_db_instance = MagicMock()
        mock_db_cls.return_value = mock_db_instance

        pipeline = QdrantCostOptimizedRAGSearchPipeline(qdrant_config)
        assert pipeline is not None
        assert pipeline.collection_name == "test_cost_optimized_rag"
        mock_db_cls.assert_called_once()

    @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.qdrant.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.SparseEmbedder")
    def test_init_with_config_path(
        self, mock_sparse_cls, mock_embedder_helper, mock_db_cls, tmp_path
    ):
        """Test initialization with config file path."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
qdrant:
  url: http://localhost:6333
  api_key: ""
  collection_name: test_cost_optimized_rag
  dimension: 384
embeddings:
  model: sentence-transformers/all-MiniLM-L6-v2
  device: cpu
dataloader:
  type: arc
  split: test
  limit: 10
search:
  top_k: 5
  rrf_k: 60
            """
        )

        mock_embedder = MagicMock()
        mock_embedder_helper.return_value = mock_embedder

        mock_sparse_embedder = MagicMock()
        mock_sparse_cls.return_value = mock_sparse_embedder

        mock_db_instance = MagicMock()
        mock_db_cls.return_value = mock_db_instance

        pipeline = QdrantCostOptimizedRAGSearchPipeline(str(config_file))
        assert pipeline is not None

    class TestSearch:
        """Tests for search method."""

        @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.QdrantVectorDB")
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.qdrant.EmbedderHelper.create_embedder"
        )
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.qdrant.EmbedderHelper.embed_query"
        )
        @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.SparseEmbedder")
        def test_search_returns_documents(
            self,
            mock_sparse_cls,
            mock_embed_query,
            mock_embedder_helper,
            mock_db_cls,
            qdrant_config,
        ):
            """Test that search returns documents."""
            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_sparse_embedder = MagicMock()
            mock_sparse_embedder.embed_query.return_value = {
                "indices": [0, 1, 2],
                "values": [0.5, 0.3, 0.2],
            }
            mock_sparse_cls.return_value = mock_sparse_embedder

            mock_db_instance = MagicMock()
            mock_db_instance.search.return_value = []
            mock_db_cls.return_value = mock_db_instance

            mock_embed_query.return_value = [0.1] * 384

            pipeline = QdrantCostOptimizedRAGSearchPipeline(qdrant_config)
            result = pipeline.search("test query")

            assert "documents" in result
            assert "query" in result

        @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.QdrantVectorDB")
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.qdrant.EmbedderHelper.create_embedder"
        )
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.qdrant.EmbedderHelper.embed_query"
        )
        @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.SparseEmbedder")
        def test_search_with_filters(
            self,
            mock_sparse_cls,
            mock_embed_query,
            mock_embedder_helper,
            mock_db_cls,
            qdrant_config,
        ):
            """Test search with filters."""
            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_sparse_embedder = MagicMock()
            mock_sparse_embedder.embed_query.return_value = {
                "indices": [0, 1, 2],
                "values": [0.5, 0.3, 0.2],
            }
            mock_sparse_cls.return_value = mock_sparse_embedder

            mock_db_instance = MagicMock()
            mock_db_instance.search.return_value = []
            mock_db_cls.return_value = mock_db_instance

            mock_embed_query.return_value = [0.1] * 384

            pipeline = QdrantCostOptimizedRAGSearchPipeline(qdrant_config)
            filters = {"metadata.field": "value"}
            pipeline.search("test query", filters=filters)

            mock_db_instance.search.assert_called_once()

        @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.QdrantVectorDB")
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.qdrant.EmbedderHelper.create_embedder"
        )
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.qdrant.EmbedderHelper.embed_query"
        )
        @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.SparseEmbedder")
        def test_search_with_top_k(
            self,
            mock_sparse_cls,
            mock_embed_query,
            mock_embedder_helper,
            mock_db_cls,
            qdrant_config,
        ):
            """Test search with custom top_k."""
            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_sparse_embedder = MagicMock()
            mock_sparse_embedder.embed_query.return_value = {
                "indices": [0, 1, 2],
                "values": [0.5, 0.3, 0.2],
            }
            mock_sparse_cls.return_value = mock_sparse_embedder

            mock_db_instance = MagicMock()
            mock_db_instance.search.return_value = []
            mock_db_cls.return_value = mock_db_instance

            mock_embed_query.return_value = [0.1] * 384

            pipeline = QdrantCostOptimizedRAGSearchPipeline(qdrant_config)
            result = pipeline.search("test query", top_k=5)

            assert result is not None

        @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.QdrantVectorDB")
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.qdrant.EmbedderHelper.create_embedder"
        )
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.qdrant.EmbedderHelper.embed_query"
        )
        @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.SparseEmbedder")
        def test_search_fuses_results(
            self,
            mock_sparse_cls,
            mock_embed_query,
            mock_embedder_helper,
            mock_db_cls,
            qdrant_config,
        ):
            """Test that search uses native hybrid search."""
            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_sparse_embedder = MagicMock()
            mock_sparse_embedder.embed_query.return_value = {
                "indices": [0, 1, 2],
                "values": [0.5, 0.3, 0.2],
            }
            mock_sparse_cls.return_value = mock_sparse_embedder

            # Simulate hybrid search results
            hybrid_docs = [
                {"id": "1", "text": "doc1", "score": 0.9},
                {"id": "2", "text": "doc2", "score": 0.8},
            ]

            mock_db_instance = MagicMock()
            mock_db_instance.search.return_value = hybrid_docs
            mock_db_cls.return_value = mock_db_instance

            mock_embed_query.return_value = [0.1] * 384

            pipeline = QdrantCostOptimizedRAGSearchPipeline(qdrant_config)
            result = pipeline.search("test query")

            assert "documents" in result
            mock_db_instance.search.assert_called_once()
