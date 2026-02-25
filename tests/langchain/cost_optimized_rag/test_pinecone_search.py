"""Tests for Pinecone cost-optimized RAG search pipeline (LangChain)."""

from unittest.mock import MagicMock, patch

from haystack.dataclasses import Document as HaystackDocument

from vectordb.langchain.cost_optimized_rag.search.pinecone import (
    PineconeCostOptimizedRAGSearchPipeline,
)


class TestPineconeCostOptimizedRAGSearchPipeline:
    """Tests for PineconeCostOptimizedRAGSearchPipeline."""

    @patch("vectordb.langchain.cost_optimized_rag.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.cost_optimized_rag.search.pinecone.SparseEmbedder")
    def test_init_with_valid_config(
        self, mock_sparse_cls, mock_embedder_helper, mock_db_cls, pinecone_config
    ):
        """Test initialization with valid config dict."""
        mock_embedder = MagicMock()
        mock_embedder_helper.return_value = mock_embedder

        mock_sparse_embedder = MagicMock()
        mock_sparse_cls.return_value = mock_sparse_embedder

        mock_db_instance = MagicMock()
        mock_db_cls.return_value = mock_db_instance

        pipeline = PineconeCostOptimizedRAGSearchPipeline(pinecone_config)
        assert pipeline is not None
        assert pipeline.index_name == "test-index"
        assert pipeline.namespace == "test"
        mock_db_cls.assert_called_once()

    @patch("vectordb.langchain.cost_optimized_rag.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.cost_optimized_rag.search.pinecone.SparseEmbedder")
    def test_init_with_config_path(
        self, mock_sparse_cls, mock_embedder_helper, mock_db_cls, tmp_path
    ):
        """Test initialization with config file path."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
pinecone:
  api_key: test-key
  index_name: test-index
  namespace: test
  dimension: 384
  metric: cosine
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

        pipeline = PineconeCostOptimizedRAGSearchPipeline(str(config_file))
        assert pipeline is not None

    class TestSearch:
        """Tests for search method."""

        @patch("vectordb.langchain.cost_optimized_rag.search.pinecone.PineconeVectorDB")
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.pinecone.EmbedderHelper.create_embedder"
        )
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.pinecone.EmbedderHelper.embed_query"
        )
        @patch("vectordb.langchain.cost_optimized_rag.search.pinecone.SparseEmbedder")
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.pinecone.ResultMerger.merge_and_deduplicate"
        )
        def test_search_returns_documents(
            self,
            mock_merger,
            mock_sparse_cls,
            mock_embed_query,
            mock_embedder_helper,
            mock_db_cls,
            pinecone_config,
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

            pipeline = PineconeCostOptimizedRAGSearchPipeline(pinecone_config)
            result = pipeline.search("test query")

            assert "documents" in result
            assert "query" in result

        @patch("vectordb.langchain.cost_optimized_rag.search.pinecone.PineconeVectorDB")
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.pinecone.EmbedderHelper.create_embedder"
        )
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.pinecone.EmbedderHelper.embed_query"
        )
        @patch("vectordb.langchain.cost_optimized_rag.search.pinecone.SparseEmbedder")
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.pinecone.ResultMerger.merge_and_deduplicate"
        )
        def test_search_with_filters(
            self,
            mock_merger,
            mock_sparse_cls,
            mock_embed_query,
            mock_embedder_helper,
            mock_db_cls,
            pinecone_config,
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

            pipeline = PineconeCostOptimizedRAGSearchPipeline(pinecone_config)
            filters = {"metadata.field": "value"}
            pipeline.search("test query", filters=filters)

            mock_db_instance.query.assert_called_once()
            mock_db_instance.query_with_sparse.assert_called_once()

        @patch("vectordb.langchain.cost_optimized_rag.search.pinecone.PineconeVectorDB")
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.pinecone.EmbedderHelper.create_embedder"
        )
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.pinecone.EmbedderHelper.embed_query"
        )
        @patch("vectordb.langchain.cost_optimized_rag.search.pinecone.SparseEmbedder")
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.pinecone.ResultMerger.merge_and_deduplicate"
        )
        def test_search_with_top_k(
            self,
            mock_merger,
            mock_sparse_cls,
            mock_embed_query,
            mock_embedder_helper,
            mock_db_cls,
            pinecone_config,
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

            pipeline = PineconeCostOptimizedRAGSearchPipeline(pinecone_config)
            result = pipeline.search("test query", top_k=5)

            assert result is not None

        @patch("vectordb.langchain.cost_optimized_rag.search.pinecone.PineconeVectorDB")
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.pinecone.EmbedderHelper.create_embedder"
        )
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.pinecone.EmbedderHelper.embed_query"
        )
        @patch("vectordb.langchain.cost_optimized_rag.search.pinecone.SparseEmbedder")
        @patch(
            "vectordb.langchain.cost_optimized_rag.search.pinecone.ResultMerger.merge_and_deduplicate"
        )
        def test_search_fuses_results(
            self,
            mock_merger,
            mock_sparse_cls,
            mock_embed_query,
            mock_embedder_helper,
            mock_db_cls,
            pinecone_config,
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
                HaystackDocument(content="doc1", meta={}, id="1"),
                HaystackDocument(content="doc2", meta={}, id="2"),
            ]
            sparse_docs = [
                HaystackDocument(content="doc2", meta={}, id="2"),
                HaystackDocument(content="doc3", meta={}, id="3"),
            ]

            mock_db_instance = MagicMock()
            mock_db_instance.query.return_value = dense_docs
            mock_db_instance.query_with_sparse.return_value = sparse_docs
            mock_db_cls.return_value = mock_db_instance

            mock_embed_query.return_value = [0.1] * 384

            from langchain_core.documents import Document as LangchainDocument

            mock_merger.return_value = [
                LangchainDocument(page_content="doc1", metadata={"id": "1"}),
                LangchainDocument(page_content="doc2", metadata={"id": "2"}),
                LangchainDocument(page_content="doc3", metadata={"id": "3"}),
            ]

            pipeline = PineconeCostOptimizedRAGSearchPipeline(pinecone_config)
            result = pipeline.search("test query")

            assert "documents" in result
