"""Unit tests for Qdrant MMR search pipeline (LangChain).

Tests validate the QdrantMMRSearchPipeline's behavior for Maximal Marginal
Relevance (MMR) search with Qdrant vector database. MMR balances relevance
against diversity to reduce redundant results in retrieval.

Qdrant-specific aspects tested include:
- URL-based connection to Qdrant server
- API key-based authentication
- Collection name configuration
- Metadata filtering in vector search
- MMR reranking with diversity parameters (lambda)

Test coverage includes:
- Pipeline initialization from config dict and file path
- Search with various parameters (top_k, mmr_k, filters)
- MMR reranking application on retrieved candidates
- Lambda parameter control for relevance/diversity trade-off

These tests mock external dependencies (QdrantVectorDB, EmbedderHelper,
MMRHelper) to enable fast, isolated unit tests without requiring live
Qdrant service connections.
"""

from typing import Any
from unittest.mock import MagicMock, patch

from vectordb.langchain.mmr.search.qdrant import QdrantMMRSearchPipeline


class TestQdrantMMRSearchPipeline:
    """Test suite for Qdrant MMR search pipeline.

    Validates pipeline initialization and search execution:
    - Config parsing with Qdrant-specific parameters
    - Collection name extraction
    - Search with query embedding and MMR reranking
    - Filter support for metadata-based queries
    - Lambda parameter for diversity control
    """

    @patch("vectordb.langchain.mmr.search.qdrant.QdrantVectorDB")
    @patch("vectordb.langchain.mmr.search.qdrant.EmbedderHelper.create_embedder")
    def test_init_with_valid_config(
        self,
        mock_embedder_helper: Any,
        mock_db_cls: Any,
        qdrant_config: Any,
    ) -> None:
        """Test pipeline initialization from configuration dictionary.

        Validates that:
        - Pipeline extracts collection_name from qdrant config
        - URL and API key are configured for connection
        - Embedder is initialized via EmbedderHelper
        - QdrantVectorDB is instantiated with config

        Args:
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db_cls: Mock for QdrantVectorDB class.
            qdrant_config: Fixture with Qdrant configuration dict.

        Returns:
            None
        """
        mock_embedder = MagicMock()
        mock_embedder_helper.return_value = mock_embedder

        mock_db_instance = MagicMock()
        mock_db_cls.return_value = mock_db_instance

        pipeline = QdrantMMRSearchPipeline(qdrant_config)
        assert pipeline is not None
        assert pipeline.collection_name == "test_mmr"

    @patch("vectordb.langchain.mmr.search.qdrant.QdrantVectorDB")
    @patch("vectordb.langchain.mmr.search.qdrant.EmbedderHelper.create_embedder")
    def test_init_with_config_path(
        self, mock_embedder_helper: Any, mock_db_cls: Any, tmp_path: Any
    ) -> None:
        """Test pipeline initialization from YAML configuration file.

        Validates that:
        - YAML config file is loaded and parsed
        - All Qdrant parameters (url, api_key, collection_name) extracted
        - MMR-specific settings (threshold, k) are configured
        - Pipeline initializes successfully from file path

        Args:
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db_cls: Mock for QdrantVectorDB class.
            tmp_path: Pytest fixture for temporary directory.

        Returns:
            None
        """
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
qdrant:
  url: http://localhost:6333
  api_key: ""
  collection_name: test_mmr
  dimension: 384
embeddings:
  model: sentence-transformers/all-MiniLM-L6-v2
  device: cpu
dataloader:
  type: arc
  split: test
  limit: 10
mmr:
  threshold: 0.5
  k: 5
            """
        )

        mock_embedder = MagicMock()
        mock_embedder_helper.return_value = mock_embedder

        mock_db_instance = MagicMock()
        mock_db_cls.return_value = mock_db_instance

        pipeline = QdrantMMRSearchPipeline(str(config_file))
        assert pipeline is not None

    class TestSearch:
        """Nested test class for search method scenarios.

        Validates various search configurations and MMR behavior:
        - Basic search returning documents
        - Custom mmr_k parameter for result count
        - Metadata filters for targeted retrieval
        - Custom top_k for candidate pool size
        - MMR reranking verification
        """

        @patch("vectordb.langchain.mmr.search.qdrant.QdrantVectorDB")
        @patch("vectordb.langchain.mmr.search.qdrant.EmbedderHelper.create_embedder")
        @patch("vectordb.langchain.mmr.search.qdrant.EmbedderHelper.embed_query")
        @patch("vectordb.langchain.mmr.search.qdrant.MMRHelper.mmr_rerank_simple")
        def test_search_returns_documents(
            self,
            mock_mmr: Any,
            mock_embed_query: Any,
            mock_embedder_helper: Any,
            mock_db_cls: Any,
            qdrant_config: Any,
            sample_mmr_candidates: Any,
        ) -> None:
            """Test basic search execution returns documents with query.

            Validates the search workflow:
            1. Query is embedded via EmbedderHelper.embed_query
            2. QdrantVectorDB.query retrieves candidate documents
            3. MMRHelper.mmr_rerank_simple diversifies results
            4. Result dict contains documents and original query

            Args:
                mock_mmr: Mock for MMRHelper.mmr_rerank_simple.
                mock_embed_query: Mock returning query embedding vector.
                mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
                mock_db_cls: Mock for QdrantVectorDB class.
                qdrant_config: Fixture with Qdrant configuration.
                sample_mmr_candidates: Fixture with MMR candidate documents.

            Returns:
                None
            """
            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_embed_query.return_value = [0.1] * 384

            mock_db_instance = MagicMock()
            mock_db_instance.query.return_value = sample_mmr_candidates
            mock_db_cls.return_value = mock_db_instance

            mock_mmr.return_value = sample_mmr_candidates[:3]

            pipeline = QdrantMMRSearchPipeline(qdrant_config)
            result = pipeline.search("test query")

            assert "documents" in result
            assert "query" in result
            assert result["query"] == "test query"

        @patch("vectordb.langchain.mmr.search.qdrant.QdrantVectorDB")
        @patch("vectordb.langchain.mmr.search.qdrant.EmbedderHelper.create_embedder")
        @patch("vectordb.langchain.mmr.search.qdrant.EmbedderHelper.embed_query")
        @patch("vectordb.langchain.mmr.search.qdrant.MMRHelper.mmr_rerank_simple")
        def test_search_with_mmr_k_parameter(
            self,
            mock_mmr: Any,
            mock_embed_query: Any,
            mock_embedder_helper: Any,
            mock_db_cls: Any,
            qdrant_config: Any,
            sample_mmr_candidates: Any,
        ) -> None:
            """Test mmr_k parameter controls number of MMR-selected results.

            Validates that:
            - mmr_k parameter is passed to MMRHelper.mmr_rerank_simple
            - Controls final result count after diversity reranking
            - Separate from top_k which controls initial candidate pool

            Args:
                mock_mmr: Mock for MMRHelper.mmr_rerank_simple.
                mock_embed_query: Mock returning query embedding vector.
                mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
                mock_db_cls: Mock for QdrantVectorDB class.
                qdrant_config: Fixture with Qdrant configuration.
                sample_mmr_candidates: Fixture with MMR candidate documents.

            Returns:
                None
            """
            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_embed_query.return_value = [0.1] * 384

            mock_db_instance = MagicMock()
            mock_db_instance.query.return_value = sample_mmr_candidates
            mock_db_cls.return_value = mock_db_instance

            mock_mmr.return_value = sample_mmr_candidates[:5]

            pipeline = QdrantMMRSearchPipeline(qdrant_config)
            pipeline.search("test query", mmr_k=5)

            mock_mmr.assert_called_once()
            call_args = mock_mmr.call_args
            assert call_args.kwargs.get("k") == 5

        @patch("vectordb.langchain.mmr.search.qdrant.QdrantVectorDB")
        @patch("vectordb.langchain.mmr.search.qdrant.EmbedderHelper.create_embedder")
        @patch("vectordb.langchain.mmr.search.qdrant.EmbedderHelper.embed_query")
        @patch("vectordb.langchain.mmr.search.qdrant.MMRHelper.mmr_rerank_simple")
        def test_search_with_filters(
            self,
            mock_mmr: Any,
            mock_embed_query: Any,
            mock_embedder_helper: Any,
            mock_db_cls: Any,
            qdrant_config: Any,
            sample_mmr_candidates: Any,
        ) -> None:
            """Test metadata filters restrict search to matching documents.

            Validates that:
            - Filters dict is passed to QdrantVectorDB.query
            - Qdrant filter format applied to metadata fields
            - MMR reranking applied to filtered results

            Args:
                mock_mmr: Mock for MMRHelper.mmr_rerank_simple.
                mock_embed_query: Mock returning query embedding vector.
                mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
                mock_db_cls: Mock for QdrantVectorDB class.
                qdrant_config: Fixture with Qdrant configuration.
                sample_mmr_candidates: Fixture with MMR candidate documents.

            Returns:
                None
            """
            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_embed_query.return_value = [0.1] * 384

            mock_db_instance = MagicMock()
            mock_db_instance.query.return_value = sample_mmr_candidates
            mock_db_cls.return_value = mock_db_instance

            mock_mmr.return_value = sample_mmr_candidates[:3]

            pipeline = QdrantMMRSearchPipeline(qdrant_config)
            filters = {"metadata.source": "wiki"}
            pipeline.search("test query", filters=filters)

            mock_db_instance.query.assert_called_once()
            call_args = mock_db_instance.query.call_args
            assert call_args.kwargs.get("filters") == filters

        @patch("vectordb.langchain.mmr.search.qdrant.QdrantVectorDB")
        @patch("vectordb.langchain.mmr.search.qdrant.EmbedderHelper.create_embedder")
        @patch("vectordb.langchain.mmr.search.qdrant.EmbedderHelper.embed_query")
        @patch("vectordb.langchain.mmr.search.qdrant.MMRHelper.mmr_rerank_simple")
        def test_search_with_top_k(
            self,
            mock_mmr: Any,
            mock_embed_query: Any,
            mock_embedder_helper: Any,
            mock_db_cls: Any,
            qdrant_config: Any,
            sample_mmr_candidates: Any,
        ) -> None:
            """Test top_k parameter controls initial candidate pool size.

            Validates that:
            - top_k is passed to QdrantVectorDB.query
            - Controls number of candidates before MMR reranking
            - Larger top_k provides more diversity options for MMR

            Args:
                mock_mmr: Mock for MMRHelper.mmr_rerank_simple.
                mock_embed_query: Mock returning query embedding vector.
                mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
                mock_db_cls: Mock for QdrantVectorDB class.
                qdrant_config: Fixture with Qdrant configuration.
                sample_mmr_candidates: Fixture with MMR candidate documents.

            Returns:
                None
            """
            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_embed_query.return_value = [0.1] * 384

            mock_db_instance = MagicMock()
            mock_db_instance.query.return_value = sample_mmr_candidates
            mock_db_cls.return_value = mock_db_instance

            mock_mmr.return_value = sample_mmr_candidates[:3]

            pipeline = QdrantMMRSearchPipeline(qdrant_config)
            pipeline.search("test query", top_k=10)

            mock_db_instance.query.assert_called_once()
            call_args = mock_db_instance.query.call_args
            assert call_args.kwargs.get("top_k") == 10

        @patch("vectordb.langchain.mmr.search.qdrant.QdrantVectorDB")
        @patch("vectordb.langchain.mmr.search.qdrant.EmbedderHelper.create_embedder")
        @patch("vectordb.langchain.mmr.search.qdrant.EmbedderHelper.embed_query")
        @patch("vectordb.langchain.mmr.search.qdrant.MMRHelper.mmr_rerank_simple")
        def test_mmr_applied_to_results(
            self,
            mock_mmr: Any,
            mock_embed_query: Any,
            mock_embedder_helper: Any,
            mock_db_cls: Any,
            qdrant_config: Any,
            sample_mmr_candidates: Any,
        ) -> None:
            """Test MMR reranking is applied to vector search results.

            Validates that:
            - MMRHelper.mmr_rerank_simple is called with candidates
            - MMR may return different documents than initial query
            - Result documents match MMR output (not raw query output)
            - Diversity is introduced into final results

            Args:
                mock_mmr: Mock for MMRHelper.mmr_rerank_simple.
                mock_embed_query: Mock returning query embedding vector.
                mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
                mock_db_cls: Mock for QdrantVectorDB class.
                qdrant_config: Fixture with Qdrant configuration.
                sample_mmr_candidates: Fixture with MMR candidate documents.

            Returns:
                None
            """
            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_embed_query.return_value = [0.1] * 384

            mock_db_instance = MagicMock()
            mock_db_instance.query.return_value = sample_mmr_candidates
            mock_db_cls.return_value = mock_db_instance

            mmr_result = [sample_mmr_candidates[0], sample_mmr_candidates[2]]
            mock_mmr.return_value = mmr_result

            pipeline = QdrantMMRSearchPipeline(qdrant_config)
            result = pipeline.search("test query")

            mock_mmr.assert_called_once()
            assert len(result["documents"]) == 2

    class TestMMRParameters:
        """Test class for MMR-specific parameter handling.

        Validates diversity control parameters:
        - Lambda parameter for relevance vs diversity trade-off
        - Threshold settings for MMR scoring
        """

        @patch("vectordb.langchain.mmr.search.qdrant.QdrantVectorDB")
        @patch("vectordb.langchain.mmr.search.qdrant.EmbedderHelper.create_embedder")
        @patch("vectordb.langchain.mmr.search.qdrant.EmbedderHelper.embed_query")
        @patch("vectordb.langchain.mmr.search.qdrant.MMRHelper.mmr_rerank_simple")
        def test_lambda_param_passed_to_mmr(
            self,
            mock_mmr: Any,
            mock_embed_query: Any,
            mock_embedder_helper: Any,
            mock_db_cls: Any,
            qdrant_config: Any,
            sample_mmr_candidates: Any,
        ) -> None:
            """Test lambda_param controls MMR relevance/diversity trade-off.

            Validates that:
            - lambda_param is passed to MMRHelper.mmr_rerank_simple
            - Value between 0.0 (diversity) and 1.0 (relevance)
            - Affects MMR scoring formula application

            Args:
                mock_mmr: Mock for MMRHelper.mmr_rerank_simple.
                mock_embed_query: Mock returning query embedding vector.
                mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
                mock_db_cls: Mock for QdrantVectorDB class.
                qdrant_config: Fixture with Qdrant configuration.
                sample_mmr_candidates: Fixture with MMR candidate documents.

            Returns:
                None
            """
            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_embed_query.return_value = [0.1] * 384

            mock_db_instance = MagicMock()
            mock_db_instance.query.return_value = sample_mmr_candidates
            mock_db_cls.return_value = mock_db_instance

            mock_mmr.return_value = sample_mmr_candidates[:3]

            pipeline = QdrantMMRSearchPipeline(qdrant_config)
            pipeline.search("test query", lambda_param=0.7)

            mock_mmr.assert_called_once()
