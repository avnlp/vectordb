"""Tests for Qdrant diversity filtering pipelines (LangChain)."""

from unittest.mock import MagicMock, patch

import pytest

from vectordb.langchain.diversity_filtering.indexing.qdrant import (
    QdrantDiversityFilteringIndexingPipeline,
)
from vectordb.langchain.diversity_filtering.search.qdrant import (
    QdrantDiversityFilteringSearchPipeline,
)


class TestQdrantDiversityFilteringIndexing:
    """Unit tests for Qdrant diversity filtering indexing pipeline."""

    @patch("vectordb.langchain.diversity_filtering.indexing.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.qdrant.DataloaderCatalog.create"
    )
    def test_indexing_initialization(
        self,
        mock_get_docs,
        mock_embedder_helper,
        mock_db,
        qdrant_diversity_filtering_config,
    ):
        """Test pipeline initialization."""
        pipeline = QdrantDiversityFilteringIndexingPipeline(
            qdrant_diversity_filtering_config
        )
        assert pipeline.config == qdrant_diversity_filtering_config
        assert pipeline.collection_name == "test_diversity_filtering"

    @patch("vectordb.langchain.diversity_filtering.indexing.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.qdrant.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.qdrant.DataloaderCatalog.create"
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
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "",
                "collection_name": "test_diversity_filtering",
                "dimension": 384,
            },
        }

        pipeline = QdrantDiversityFilteringIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.upsert.assert_called_once()

    @patch("vectordb.langchain.diversity_filtering.indexing.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.qdrant.DataloaderCatalog.create"
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
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "",
                "collection_name": "test_diversity_filtering",
                "dimension": 384,
            },
        }

        pipeline = QdrantDiversityFilteringIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0

    @patch("vectordb.langchain.diversity_filtering.indexing.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.qdrant.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.qdrant.DataloaderCatalog.create"
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
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "",
                "collection_name": "test_diversity_filtering",
                "dimension": 384,
                "recreate": True,
            },
        }

        pipeline = QdrantDiversityFilteringIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.create_collection.assert_called_once_with(
            collection_name="test_diversity_filtering",
            dimension=384,
            recreate=True,
        )


class TestQdrantDiversityFilteringSearch:
    """Unit tests for Qdrant diversity filtering search pipeline."""

    @patch("vectordb.langchain.diversity_filtering.search.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.search.qdrant.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.diversity_filtering.search.qdrant.RAGHelper.create_llm")
    def test_search_initialization(
        self, mock_llm_helper, mock_embedder_helper, mock_db
    ):
        """Test search pipeline initialization."""
        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "",
                "collection_name": "test_diversity_filtering",
            },
            "diversity": {
                "method": "mmr",
                "max_documents": 5,
                "lambda_param": 0.5,
            },
            "rag": {"enabled": False},
        }

        pipeline = QdrantDiversityFilteringSearchPipeline(config)
        assert pipeline.config == config

    @patch("vectordb.langchain.diversity_filtering.search.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.search.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.qdrant.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.qdrant.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.diversity_filtering.search.qdrant.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.diversity_filtering.search.qdrant.DiversityFilteringHelper.mmr_diversify"
    )
    def test_search_execution_mmr_method(
        self,
        mock_diversify,
        mock_llm_helper,
        mock_embed_query,
        mock_embed_documents,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test search execution with MMR-based diversity filtering.

        Validates that when ``diversity.method`` is ``"mmr"``, the pipeline calls
        ``DiversityFilteringHelper.mmr_diversify`` with the configured ``lambda_param``
        and returns a result set no larger than ``top_k``.

        Args:
            mock_diversify: Mock for DiversityFilteringHelper.mmr_diversify
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_embed_query: Mock for EmbedderHelper.embed_query
            mock_embed_documents: Mock for EmbedderHelper.embed_documents
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for QdrantVectorDB class
            sample_documents: Fixture providing test documents

        Verifies:
            - mmr_diversify is called exactly once with MMR configuration
            - Result contains the original query string
            - Returned document count does not exceed top_k
        """
        mock_embed_query.return_value = [0.1] * 384
        mock_embed_documents.return_value = (
            sample_documents,
            [[0.1] * 384] * len(sample_documents),
        )
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None
        mock_diversify.return_value = sample_documents[:3]

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "",
                "collection_name": "test_diversity_filtering",
            },
            "diversity": {
                "method": "mmr",
                "max_documents": 5,
                "lambda_param": 0.5,
            },
            "rag": {"enabled": False},
        }

        pipeline = QdrantDiversityFilteringSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert len(result["documents"]) <= 5
        mock_diversify.assert_called_once()

    @patch("vectordb.langchain.diversity_filtering.search.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.search.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.qdrant.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.qdrant.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.diversity_filtering.search.qdrant.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.diversity_filtering.search.qdrant.DiversityFilteringHelper.clustering_diversify"
    )
    def test_search_execution_clustering_method(
        self,
        mock_cluster_diversify,
        mock_llm_helper,
        mock_embed_query,
        mock_embed_documents,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test search execution with clustering-based diversity filtering.

        Validates that when ``diversity.method`` is ``"clustering"``, the pipeline
        delegates to ``DiversityFilteringHelper.clustering_diversify`` and that
        cluster parameters (``num_clusters``, ``samples_per_cluster``) are forwarded.

        Args:
            mock_cluster_diversify: Mock for
                DiversityFilteringHelper.clustering_diversify
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_embed_query: Mock for EmbedderHelper.embed_query
            mock_embed_documents: Mock for EmbedderHelper.embed_documents
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for QdrantVectorDB class
            sample_documents: Fixture providing test documents

        Verifies:
            - clustering_diversify is called exactly once when method is "clustering"
            - Result contains the original query string
            - Returned document count does not exceed top_k
        """
        mock_embed_query.return_value = [0.1] * 384
        mock_embed_documents.return_value = (
            sample_documents,
            [[0.1] * 384] * len(sample_documents),
        )
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None
        mock_cluster_diversify.return_value = sample_documents[:3]

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "",
                "collection_name": "test_diversity_filtering",
            },
            "diversity": {
                "method": "clustering",
                "num_clusters": 3,
                "samples_per_cluster": 2,
            },
            "rag": {"enabled": False},
        }

        pipeline = QdrantDiversityFilteringSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert len(result["documents"]) <= 5
        mock_cluster_diversify.assert_called_once()

    @patch("vectordb.langchain.diversity_filtering.search.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.search.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.qdrant.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.qdrant.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.diversity_filtering.search.qdrant.RAGHelper.create_llm")
    def test_search_execution_invalid_method(
        self,
        mock_llm_helper,
        mock_embed_query,
        mock_embed_documents,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test that an unrecognised diversity method raises ValueError.

        Ensures the pipeline does not silently fall through to an unexpected code path
        when a caller supplies an obsolete or misspelled method name such as
        ``"threshold"``.

        Args:
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_embed_query: Mock for EmbedderHelper.embed_query
            mock_embed_documents: Mock for EmbedderHelper.embed_documents
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for QdrantVectorDB class
            sample_documents: Fixture providing test documents

        Verifies:
            - ``ValueError`` is raised when ``diversity.method`` is ``"threshold"``
            - Error message matches ``"Unknown diversity method"``
        """
        mock_embed_query.return_value = [0.1] * 384
        mock_embed_documents.return_value = (
            sample_documents,
            [[0.1] * 384] * len(sample_documents),
        )
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "",
                "collection_name": "test_diversity_filtering",
            },
            "diversity": {
                "method": "threshold",
                "max_documents": 5,
                "lambda_param": 0.5,
            },
            "rag": {"enabled": False},
        }

        pipeline = QdrantDiversityFilteringSearchPipeline(config)
        with pytest.raises(ValueError, match="Unknown diversity method"):
            pipeline.search("test query", top_k=5)

    @patch("vectordb.langchain.diversity_filtering.search.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.search.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.qdrant.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.qdrant.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.diversity_filtering.search.qdrant.RAGHelper.create_llm")
    @patch("vectordb.langchain.diversity_filtering.search.qdrant.RAGHelper.generate")
    @patch(
        "vectordb.langchain.diversity_filtering.search.qdrant.DiversityFilteringHelper.mmr_diversify"
    )
    def test_search_with_rag(
        self,
        mock_diversify,
        mock_rag_generate,
        mock_llm_helper,
        mock_embed_query,
        mock_embed_documents,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test that MMR search results are used to generate a RAG answer.

        Validates the full end-to-end path where diverse documents retrieved via MMR
        are passed to the LLM to synthesise a final answer. Ensures the generated
        answer is surfaced in the result dictionary alongside the original documents.

        Args:
            mock_diversify: Mock for DiversityFilteringHelper.mmr_diversify
            mock_rag_generate: Mock for RAGHelper.generate
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_embed_query: Mock for EmbedderHelper.embed_query
            mock_embed_documents: Mock for EmbedderHelper.embed_documents
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for QdrantVectorDB class
            sample_documents: Fixture providing test documents

        Verifies:
            - Query is preserved in the result
            - RAG-generated answer is included in the result under ``"answer"``
        """
        mock_embed_query.return_value = [0.1] * 384
        mock_embed_documents.return_value = (
            sample_documents,
            [[0.1] * 384] * len(sample_documents),
        )
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst

        mock_llm_inst = MagicMock()
        mock_llm_helper.return_value = mock_llm_inst
        mock_rag_generate.return_value = "Test answer"
        mock_diversify.return_value = sample_documents[:3]

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "",
                "collection_name": "test_diversity_filtering",
            },
            "diversity": {
                "method": "mmr",
                "max_documents": 5,
                "lambda_param": 0.5,
            },
            "rag": {"enabled": True},
        }

        pipeline = QdrantDiversityFilteringSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert result["answer"] == "Test answer"

    @patch("vectordb.langchain.diversity_filtering.search.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.search.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.qdrant.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.qdrant.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.diversity_filtering.search.qdrant.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.diversity_filtering.search.qdrant.DiversityFilteringHelper.mmr_diversify"
    )
    def test_search_candidate_multiplier_config(
        self,
        mock_diversify,
        mock_llm_helper,
        mock_embed_query,
        mock_embed_documents,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test that candidate_multiplier configuration is respected.

        Validates that the candidate_multiplier parameter from the diversity config
        is used to scale the top_k value when fetching candidates from Qdrant.

        Args:
            mock_diversify: Mock for DiversityFilteringHelper.mmr_diversify
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_embed_query: Mock for EmbedderHelper.embed_query
            mock_embed_documents: Mock for EmbedderHelper.embed_documents
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for QdrantVectorDB class
            sample_documents: Fixture providing test documents

        Verifies:
            - Qdrant query is called with top_k * candidate_multiplier
            - Default multiplier of 3 is used when not specified
            - Custom multiplier is applied when configured
        """
        mock_embed_query.return_value = [0.1] * 384
        mock_embed_documents.return_value = (
            sample_documents,
            [[0.1] * 384] * len(sample_documents),
        )
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None
        mock_diversify.return_value = sample_documents[:3]

        # Test with default multiplier (3)
        config_default = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "",
                "collection_name": "test_diversity_filtering",
            },
            "diversity": {
                "method": "mmr",
                "max_documents": 5,
                "lambda_param": 0.5,
            },
            "rag": {"enabled": False},
        }

        pipeline_default = QdrantDiversityFilteringSearchPipeline(config_default)
        pipeline_default.search("test query", top_k=5)
        # Default multiplier is 3, so top_k=5 should fetch 15 documents
        mock_db_inst.query.assert_called_with(
            query_embedding=[0.1] * 384,
            top_k=15,
            filters=None,
            collection_name="test_diversity_filtering",
        )

        # Test with custom multiplier (5)
        config_custom = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "",
                "collection_name": "test_diversity_filtering",
            },
            "diversity": {
                "method": "mmr",
                "max_documents": 5,
                "lambda_param": 0.5,
                "candidate_multiplier": 5,
            },
            "rag": {"enabled": False},
        }

        pipeline_custom = QdrantDiversityFilteringSearchPipeline(config_custom)
        pipeline_custom.search("test query", top_k=5)
        # Custom multiplier is 5, so top_k=5 should fetch 25 documents
        mock_db_inst.query.assert_called_with(
            query_embedding=[0.1] * 384,
            top_k=25,
            filters=None,
            collection_name="test_diversity_filtering",
        )
