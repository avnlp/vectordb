"""Tests for Pinecone diversity filtering pipelines (LangChain)."""

from unittest.mock import MagicMock, patch

from vectordb.langchain.diversity_filtering.indexing.pinecone import (
    PineconeDiversityFilteringIndexingPipeline,
)
from vectordb.langchain.diversity_filtering.search.pinecone import (
    PineconeDiversityFilteringSearchPipeline,
)


class TestPineconeDiversityFilteringIndexing:
    """Unit tests for Pinecone diversity filtering indexing pipeline."""

    @patch("vectordb.langchain.diversity_filtering.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.pinecone.DataloaderCatalog.create"
    )
    def test_indexing_initialization(
        self,
        mock_get_docs,
        mock_embedder_helper,
        mock_db,
        pinecone_diversity_filtering_config,
    ):
        """Test pipeline initialization."""
        pipeline = PineconeDiversityFilteringIndexingPipeline(
            pinecone_diversity_filtering_config
        )
        assert pipeline.config == pinecone_diversity_filtering_config
        assert pipeline.index_name == "test-diversity-filtering"
        assert pipeline.namespace == "test-namespace"

    @patch("vectordb.langchain.diversity_filtering.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.pinecone.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.pinecone.DataloaderCatalog.create"
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
                "api_key": "test-api-key",
                "index_name": "test-diversity-filtering",
                "namespace": "test-namespace",
                "dimension": 384,
                "metric": "cosine",
            },
        }

        pipeline = PineconeDiversityFilteringIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.upsert.assert_called_once()

    @patch("vectordb.langchain.diversity_filtering.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.pinecone.DataloaderCatalog.create"
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
                "api_key": "test-api-key",
                "index_name": "test-diversity-filtering",
                "namespace": "test-namespace",
                "dimension": 384,
            },
        }

        pipeline = PineconeDiversityFilteringIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0

    @patch("vectordb.langchain.diversity_filtering.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.pinecone.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.pinecone.DataloaderCatalog.create"
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
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "test-diversity-filtering",
                "namespace": "test-namespace",
                "dimension": 384,
                "recreate": True,
            },
        }

        pipeline = PineconeDiversityFilteringIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.create_index.assert_called_once()


class TestPineconeDiversityFilteringSearch:
    """Unit tests for Pinecone diversity filtering search pipeline."""

    @patch("vectordb.langchain.diversity_filtering.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.pinecone.RAGHelper.create_llm"
    )
    def test_search_initialization(
        self, mock_llm_helper, mock_embedder_helper, mock_db
    ):
        """Test search pipeline initialization."""
        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "test-diversity-filtering",
                "namespace": "test-namespace",
            },
            "diversity": {
                "method": "threshold",
                "max_documents": 5,
                "similarity_threshold": 0.7,
            },
            "rag": {"enabled": False},
        }

        pipeline = PineconeDiversityFilteringSearchPipeline(config)
        assert pipeline.config == config

    @patch("vectordb.langchain.diversity_filtering.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.pinecone.EmbedderHelper.embed_query"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.pinecone.RAGHelper.create_llm"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.pinecone.DiversificationHelper.diversify"
    )
    def test_search_execution_threshold_method(
        self,
        mock_diversify,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test search execution with threshold method."""
        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None
        mock_diversify.return_value = sample_documents[:3]

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "test-diversity-filtering",
                "namespace": "test-namespace",
            },
            "diversity": {
                "method": "threshold",
                "max_documents": 5,
                "similarity_threshold": 0.7,
            },
            "rag": {"enabled": False},
        }

        pipeline = PineconeDiversityFilteringSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert len(result["documents"]) <= 5
        mock_diversify.assert_called_once()

    @patch("vectordb.langchain.diversity_filtering.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.pinecone.EmbedderHelper.embed_query"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.pinecone.RAGHelper.create_llm"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.pinecone.DiversificationHelper.clustering_based_diversity"
    )
    def test_search_execution_clustering_method(
        self,
        mock_cluster_diversify,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test search execution with clustering method."""
        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None
        mock_cluster_diversify.return_value = sample_documents[:3]

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "test-diversity-filtering",
                "namespace": "test-namespace",
            },
            "diversity": {
                "method": "clustering",
                "num_clusters": 3,
                "samples_per_cluster": 2,
            },
            "rag": {"enabled": False},
        }

        pipeline = PineconeDiversityFilteringSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert len(result["documents"]) <= 5
        mock_cluster_diversify.assert_called_once()

    @patch("vectordb.langchain.diversity_filtering.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.pinecone.EmbedderHelper.embed_query"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.pinecone.RAGHelper.create_llm"
    )
    @patch("vectordb.langchain.diversity_filtering.search.pinecone.RAGHelper.generate")
    @patch(
        "vectordb.langchain.diversity_filtering.search.pinecone.DiversificationHelper.diversify"
    )
    def test_search_with_rag(
        self,
        mock_diversify,
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
        mock_diversify.return_value = sample_documents[:3]

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "test-diversity-filtering",
                "namespace": "test-namespace",
            },
            "diversity": {
                "method": "threshold",
                "max_documents": 5,
                "similarity_threshold": 0.7,
            },
            "rag": {"enabled": True},
        }

        pipeline = PineconeDiversityFilteringSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert result["answer"] == "Test answer"
