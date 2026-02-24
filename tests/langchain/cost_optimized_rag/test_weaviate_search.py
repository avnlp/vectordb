"""Tests for Weaviate cost-optimized RAG search pipeline (LangChain)."""

from unittest.mock import MagicMock, patch

from vectordb.langchain.cost_optimized_rag.search.weaviate import (
    WeaviateCostOptimizedRAGSearchPipeline,
)


class TestWeaviateCostOptimizedRAGSearchPipeline:
    """Tests for WeaviateCostOptimizedRAGSearchPipeline."""

    @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.weaviate.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.SparseEmbedder")
    def test_init_with_valid_config(
        self, mock_sparse_cls, mock_embedder_helper, mock_db_cls, weaviate_config
    ):
        """Test initialization with valid config dict."""
        mock_embedder = MagicMock()
        mock_embedder_helper.return_value = mock_embedder

        mock_sparse_embedder = MagicMock()
        mock_sparse_cls.return_value = mock_sparse_embedder

        mock_db_instance = MagicMock()
        mock_db_cls.return_value = mock_db_instance

        pipeline = WeaviateCostOptimizedRAGSearchPipeline(weaviate_config)
        assert pipeline is not None
        assert pipeline.collection_name == "TestCostOptimizedRAG"
        assert pipeline.alpha == 0.5  # Weaviate hybrid parameter
        mock_db_cls.assert_called_once()

    @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.weaviate.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.SparseEmbedder")
    def test_init_with_config_path(
        self, mock_sparse_cls, mock_embedder_helper, mock_db_cls, tmp_path
    ):
        """Test initialization with config file path."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
weaviate:
  url: http://localhost:8080
  api_key: ""
  collection_name: TestCostOptimizedRAG
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
  alpha: 0.5
            """
        )

        mock_embedder = MagicMock()
        mock_embedder_helper.return_value = mock_embedder

        mock_sparse_embedder = MagicMock()
        mock_sparse_cls.return_value = mock_sparse_embedder

        mock_db_instance = MagicMock()
        mock_db_cls.return_value = mock_db_instance

        pipeline = WeaviateCostOptimizedRAGSearchPipeline(str(config_file))
        assert pipeline is not None

    class TestSearch:
        """Tests for search method."""

        @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.WeaviateVectorDB")
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.weaviate.EmbedderHelper.create_embedder"
        )
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.weaviate.EmbedderHelper.embed_query"
        )
        @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.SparseEmbedder")
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.weaviate.ResultMerger.merge_and_deduplicate"
        )
        def test_search_returns_documents(
            self,
            mock_merger,
            mock_sparse_cls,
            mock_embed_query,
            mock_embedder_helper,
            mock_db_cls,
            weaviate_config,
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
            mock_db_instance.query.return_value = []
            mock_db_instance.query_with_sparse.return_value = []
            mock_db_cls.return_value = mock_db_instance

            mock_embed_query.return_value = [0.1] * 384

            mock_merger.return_value = []

            pipeline = WeaviateCostOptimizedRAGSearchPipeline(weaviate_config)
            result = pipeline.search("test query")

            assert "documents" in result
            assert "query" in result

        @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.WeaviateVectorDB")
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.weaviate.EmbedderHelper.create_embedder"
        )
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.weaviate.EmbedderHelper.embed_query"
        )
        @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.SparseEmbedder")
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.weaviate.ResultMerger.merge_and_deduplicate"
        )
        def test_search_with_filters(
            self,
            mock_merger,
            mock_sparse_cls,
            mock_embed_query,
            mock_embedder_helper,
            mock_db_cls,
            weaviate_config,
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
            mock_db_instance.query.return_value = []
            mock_db_instance.query_with_sparse.return_value = []
            mock_db_cls.return_value = mock_db_instance

            mock_embed_query.return_value = [0.1] * 384

            mock_merger.return_value = []

            pipeline = WeaviateCostOptimizedRAGSearchPipeline(weaviate_config)
            filters = {"metadata.field": "value"}
            pipeline.search("test query", filters=filters)

            mock_db_instance.query.assert_called_once()
            mock_db_instance.query_with_sparse.assert_called_once()

        @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.WeaviateVectorDB")
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.weaviate.EmbedderHelper.create_embedder"
        )
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.weaviate.EmbedderHelper.embed_query"
        )
        @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.SparseEmbedder")
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.weaviate.ResultMerger.merge_and_deduplicate"
        )
        def test_search_with_top_k(
            self,
            mock_merger,
            mock_sparse_cls,
            mock_embed_query,
            mock_embedder_helper,
            mock_db_cls,
            weaviate_config,
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
            mock_db_instance.query.return_value = []
            mock_db_instance.query_with_sparse.return_value = []
            mock_db_cls.return_value = mock_db_instance

            mock_embed_query.return_value = [0.1] * 384

            mock_merger.return_value = []

            pipeline = WeaviateCostOptimizedRAGSearchPipeline(weaviate_config)
            result = pipeline.search("test query", top_k=5)

            assert result is not None

        @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.WeaviateVectorDB")
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.weaviate.EmbedderHelper.create_embedder"
        )
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.weaviate.EmbedderHelper.embed_query"
        )
        @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.SparseEmbedder")
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.weaviate.ResultMerger.merge_and_deduplicate"
        )
        def test_search_fuses_results(
            self,
            mock_merger,
            mock_sparse_cls,
            mock_embed_query,
            mock_embedder_helper,
            mock_db_cls,
            weaviate_config,
        ):
            """Test that search fuses results from dense and sparse search."""
            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_sparse_embedder = MagicMock()
            mock_sparse_embedder.embed_query.return_value = {
                "indices": [0, 1, 2],
                "values": [0.5, 0.3, 0.2],
            }
            mock_sparse_cls.return_value = mock_sparse_embedder

            # Simulate some documents returned
            dense_docs = [
                {"id": "1", "text": "doc1", "score": 0.9},
                {"id": "2", "text": "doc2", "score": 0.8},
            ]
            sparse_docs = [
                {"id": "2", "text": "doc2", "score": 0.85},
                {"id": "3", "text": "doc3", "score": 0.7},
            ]

            mock_db_instance = MagicMock()
            mock_db_instance.query.return_value = dense_docs
            mock_db_instance.query_with_sparse.return_value = sparse_docs
            mock_db_cls.return_value = mock_db_instance

            mock_embed_query.return_value = [0.1] * 384

            mock_merger.return_value = dense_docs + sparse_docs

            pipeline = WeaviateCostOptimizedRAGSearchPipeline(weaviate_config)
            result = pipeline.search("test query")

            assert "documents" in result

    class TestAlphaParameter:
        """Tests for Weaviate alpha parameter (hybrid search balance)."""

        @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.WeaviateVectorDB")
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.weaviate.EmbedderHelper.create_embedder"
        )
        @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.SparseEmbedder")
        def test_custom_alpha(self, mock_sparse_cls, mock_embedder_helper, mock_db_cls):
            """Test initialization with custom alpha parameter."""
            config = {
                "dataloader": {"type": "arc", "limit": 10},
                "embeddings": {"model": "test-model", "device": "cpu"},
                "weaviate": {
                    "url": "http://localhost:8080",
                    "collection_name": "TestCostOptimizedRAG",
                    "dimension": 384,
                },
                "search": {
                    "top_k": 5,
                    "rrf_k": 60,
                    "alpha": 0.75,  # Custom alpha for more dense weighting
                },
            }

            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_sparse_embedder = MagicMock()
            mock_sparse_cls.return_value = mock_sparse_embedder

            mock_db_instance = MagicMock()
            mock_db_cls.return_value = mock_db_instance

            pipeline = WeaviateCostOptimizedRAGSearchPipeline(config)
            assert pipeline.alpha == 0.75

        @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.WeaviateVectorDB")
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.weaviate.EmbedderHelper.create_embedder"
        )
        @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.SparseEmbedder")
        def test_default_alpha(
            self, mock_sparse_cls, mock_embedder_helper, mock_db_cls, weaviate_config
        ):
            """Test initialization with default alpha parameter."""
            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_sparse_embedder = MagicMock()
            mock_sparse_cls.return_value = mock_sparse_embedder

            mock_db_instance = MagicMock()
            mock_db_cls.return_value = mock_db_instance

            pipeline = WeaviateCostOptimizedRAGSearchPipeline(weaviate_config)
            assert pipeline.alpha == 0.5
