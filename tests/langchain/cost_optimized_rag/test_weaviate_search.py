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
    def test_init_with_valid_config(
        self, mock_embedder_helper, mock_db_cls, weaviate_config
    ):
        """Test initialization with valid config dict."""
        mock_embedder = MagicMock()
        mock_embedder_helper.return_value = mock_embedder

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
    def test_init_with_config_path(self, mock_embedder_helper, mock_db_cls, tmp_path):
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
  alpha: 0.5
            """
        )

        mock_embedder = MagicMock()
        mock_embedder_helper.return_value = mock_embedder

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
        def test_search_returns_documents(
            self,
            mock_embed_query,
            mock_embedder_helper,
            mock_db_cls,
            weaviate_config,
        ):
            """Test that search returns documents."""
            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_db_instance = MagicMock()
            mock_db_instance.hybrid_search.return_value = []
            mock_db_cls.return_value = mock_db_instance

            mock_embed_query.return_value = [0.1] * 384

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
        def test_search_with_filters(
            self,
            mock_embed_query,
            mock_embedder_helper,
            mock_db_cls,
            weaviate_config,
        ):
            """Test search with filters."""
            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_db_instance = MagicMock()
            mock_db_instance.hybrid_search.return_value = []
            mock_db_cls.return_value = mock_db_instance

            mock_embed_query.return_value = [0.1] * 384

            pipeline = WeaviateCostOptimizedRAGSearchPipeline(weaviate_config)
            filters = {"metadata.field": "value"}
            pipeline.search("test query", filters=filters)

            mock_db_instance.hybrid_search.assert_called_once()

        @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.WeaviateVectorDB")
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.weaviate.EmbedderHelper.create_embedder"
        )
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.weaviate.EmbedderHelper.embed_query"
        )
        def test_search_with_top_k(
            self,
            mock_embed_query,
            mock_embedder_helper,
            mock_db_cls,
            weaviate_config,
        ):
            """Test search with custom top_k."""
            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_db_instance = MagicMock()
            mock_db_instance.hybrid_search.return_value = []
            mock_db_cls.return_value = mock_db_instance

            mock_embed_query.return_value = [0.1] * 384

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
        def test_search_uses_hybrid_search(
            self,
            mock_embed_query,
            mock_embedder_helper,
            mock_db_cls,
            weaviate_config,
        ):
            """Test that search uses Weaviate native hybrid search."""
            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            # Simulate some documents returned
            hybrid_docs = [
                {"id": "1", "text": "doc1", "score": 0.9},
                {"id": "2", "text": "doc2", "score": 0.8},
            ]

            mock_db_instance = MagicMock()
            mock_db_instance.hybrid_search.return_value = hybrid_docs
            mock_db_cls.return_value = mock_db_instance

            mock_embed_query.return_value = [0.1] * 384

            pipeline = WeaviateCostOptimizedRAGSearchPipeline(weaviate_config)
            result = pipeline.search("test query")

            assert "documents" in result
            mock_db_instance.hybrid_search.assert_called_once()

    class TestAlphaParameter:
        """Tests for Weaviate alpha parameter (hybrid search balance)."""

        @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.WeaviateVectorDB")
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.weaviate.EmbedderHelper.create_embedder"
        )
        def test_custom_alpha(self, mock_embedder_helper, mock_db_cls):
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
                    "alpha": 0.75,  # Custom alpha for more dense weighting
                },
            }

            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_db_instance = MagicMock()
            mock_db_cls.return_value = mock_db_instance

            pipeline = WeaviateCostOptimizedRAGSearchPipeline(config)
            assert pipeline.alpha == 0.75

        @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.WeaviateVectorDB")
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.weaviate.EmbedderHelper.create_embedder"
        )
        def test_default_alpha(
            self, mock_embedder_helper, mock_db_cls, weaviate_config
        ):
            """Test initialization with default alpha parameter."""
            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_db_instance = MagicMock()
            mock_db_cls.return_value = mock_db_instance

            pipeline = WeaviateCostOptimizedRAGSearchPipeline(weaviate_config)
            assert pipeline.alpha == 0.5
