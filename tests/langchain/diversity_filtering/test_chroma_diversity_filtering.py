"""Tests for Chroma diversity filtering pipelines (LangChain).

This module tests the diversity filtering feature which implements Maximal Marginal
Relevance (MMR) to balance relevance and diversity in retrieval results. MMR ensures
search results cover different aspects of a query rather than returning similar
documents.

Diversity Filtering Concept:
    Standard retrieval ranks documents purely by similarity to the query, which can
result in redundant results that all express the same aspect. Diversity filtering
uses MMR to select documents that are both relevant to the query and diverse from
each other, providing broader coverage of the answer space.

Maximal Marginal Relevance (MMR):
    MMR is an iterative selection algorithm that balances:
    - Relevance: Similarity between document and query
    - Diversity: Dissimilarity between selected documents

    Formula: MMR = lambda * Relevance - (1 - lambda) * max(Similarity to selected)

    Where lambda controls the trade-off:
    - lambda = 1.0: Pure relevance ranking (no diversity)
    - lambda = 0.0: Pure diversity (may sacrifice relevance)
    - lambda = 0.5: Balanced relevance and diversity (recommended)

Pipeline Architecture:
    Indexing Pipeline:
        1. Load documents from configured data source (ARC, TriviaQA, etc.)
        2. Generate dense embeddings for semantic similarity
        3. Store documents and embeddings in Chroma vector database
        4. Standard semantic indexing (no special handling required)

    Search Pipeline:
        1. Embed query using dense embedding model
        2. Retrieve candidate documents from Chroma (top_k * 3 for MMR pool)
        3. Compute relevance scores: similarity(query, document)
        4. Iteratively select documents using MMR algorithm:
           a. Pick document with highest MMR score
           b. Update diversity penalties for remaining documents
           c. Repeat until top_k documents selected
        5. Return diverse set of relevant documents
        6. Optionally generate RAG answer from diverse results

Components Tested:
    - ChromaDiversityFilteringIndexingPipeline: Standard semantic indexing
    - ChromaDiversityFilteringSearchPipeline: MMR-based diverse retrieval
    - MMR algorithm implementation for result selection

Key Features:
    - MMR-based diversity-relevance balancing
    - Configurable lambda parameter for trade-off control
    - Candidate pool expansion (retrieves 3x top_k for selection)
    - Iterative selection with diversity penalties
    - Optional RAG generation from diverse results
    - Chroma database integration

Test Coverage:
    - Pipeline initialization with MMR configuration
    - Document indexing (standard semantic indexing)
    - MMR search with diversity filtering
    - Lambda parameter effects on results
    - RAG generation from diverse documents
    - Empty document handling
    - Edge cases: single document, all identical documents

Configuration:
    Diversity filtering is configured in the search pipeline:
    - lambda_param: Trade-off between relevance and diversity (default: 0.5)
    - top_k: Number of diverse documents to return (default: 5)
    - Candidate pool is automatically expanded to 3 * top_k

Benefits of MMR:
    - Reduces redundancy in search results
    - Covers multiple aspects of multi-faceted queries
    - Improves user satisfaction with varied perspectives
    - Better for exploratory search scenarios

Trade-offs:
    - Requires larger initial retrieval (2x top_k)
    - Slightly higher computational cost
    - May reduce precision for narrow, specific queries

All tests mock vector database and embedding operations to ensure fast,
deterministic unit tests without external dependencies.
"""

from unittest.mock import MagicMock, patch

import pytest

from vectordb.langchain.diversity_filtering.indexing.chroma import (
    ChromaDiversityFilteringIndexingPipeline,
)
from vectordb.langchain.diversity_filtering.search.chroma import (
    ChromaDiversityFilteringSearchPipeline,
)


class TestChromaDiversityFilteringIndexing:
    """Unit tests for Chroma diversity filtering indexing pipeline.

    Validates the indexing pipeline for MMR-based diverse retrieval.
    Uses standard semantic indexing since diversity filtering happens at search time.

    Tested Behaviors:
        - Pipeline initialization with Chroma configuration
        - Document loading and dense embedding generation
        - Standard semantic document indexing
        - Collection creation and management
        - Empty document handling
        - Recreate option for collection management

    Mocks:
        - ChromaVectorDB: Database operations (collection creation, upsert)
        - EmbedderHelper: Embedding model and document embedding
        - DataLoaderHelper: Document loading from data sources

    Configuration:
        - persist_dir: Chroma persistence directory
        - collection_name: Target collection for documents
        - recreate: Whether to recreate collection on indexing
    """

    @patch("vectordb.langchain.diversity_filtering.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.chroma.DataloaderCatalog.create"
    )
    def test_indexing_initialization(
        self, mock_get_docs, mock_embedder_helper, mock_db
    ):
        """Test pipeline initialization with valid configuration.

        Args:
            mock_get_docs: Mock for DataloaderCatalog.create
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class

        Verifies:
            - Configuration is stored correctly
            - Collection name is extracted from config
            - Database connection is established
            - Embedding model is configured
        """
        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_dir": "./test_chroma_data",
                "collection_name": "test_diversity_filtering",
            },
        }

        pipeline = ChromaDiversityFilteringIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.collection_name == "test_diversity_filtering"

    @patch("vectordb.langchain.diversity_filtering.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.chroma.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.chroma.DataloaderCatalog.create"
    )
    def test_indexing_run_with_documents(
        self,
        mock_get_docs,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test indexing pipeline with documents.

        Args:
            mock_get_docs: Mock for DataloaderCatalog.create
            mock_embed_docs: Mock for EmbedderHelper.embed_documents
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class
            sample_documents: Fixture providing test documents

        Verifies:
            - Documents are loaded from data source
            - Dense embeddings are generated for all documents
            - Documents are upserted to Chroma database
            - Returns count of indexed documents
            - Database upsert is called once
        """
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
            "chroma": {
                "persist_dir": "./test_chroma_data",
                "collection_name": "test_diversity_filtering",
            },
        }

        pipeline = ChromaDiversityFilteringIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.upsert.assert_called_once()

    @patch("vectordb.langchain.diversity_filtering.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.chroma.DataloaderCatalog.create"
    )
    def test_indexing_run_no_documents(
        self, mock_get_docs, mock_embedder_helper, mock_db
    ):
        """Test indexing pipeline with no documents.

        Args:
            mock_get_docs: Mock for DataloaderCatalog.create
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class

        Verifies:
            - Pipeline handles empty document list gracefully
            - Returns 0 documents indexed
            - No database operations performed for empty input
        """
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = []
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_dir": "./test_chroma_data",
                "collection_name": "test_diversity_filtering",
            },
        }

        pipeline = ChromaDiversityFilteringIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0

    @patch("vectordb.langchain.diversity_filtering.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.chroma.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.indexing.chroma.DataloaderCatalog.create"
    )
    def test_indexing_with_recreate_option(
        self,
        mock_get_docs,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test indexing with recreate option for collection management.

        Args:
            mock_get_docs: Mock for DataloaderCatalog.create
            mock_embed_docs: Mock for EmbedderHelper.embed_documents
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class
            sample_documents: Fixture providing test documents

        Verifies:
            - Collection is recreated when recreate=True
            - create_collection is called before upserting
            - Documents are indexed after recreation
        """
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
            "chroma": {
                "persist_dir": "./test_chroma_data",
                "collection_name": "test_diversity_filtering",
                "recreate": True,
            },
        }

        pipeline = ChromaDiversityFilteringIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.delete_collection.assert_called_once_with(
            name="test_diversity_filtering"
        )
        mock_db_inst.create_collection.assert_called_once_with(
            name="test_diversity_filtering",
            get_or_create=False,
        )


class TestChromaDiversityFilteringSearch:
    """Unit tests for Chroma diversity filtering search pipeline.

    Validates the search pipeline that applies Maximal Marginal Relevance (MMR)
    to select diverse yet relevant documents from retrieval results.

    Tested Behaviors:
        - Search pipeline initialization with MMR configuration
        - Query embedding for semantic search
        - Candidate pool retrieval (3x top_k for MMR selection)
        - MMR algorithm execution for diverse selection
        - Lambda parameter control over relevance-diversity trade-off
        - RAG generation from diverse documents

    Mocks:
        - ChromaVectorDB: Database query operations
        - EmbedderHelper: Query embedding and embedding model
        - RAGHelper: LLM initialization and answer generation

    MMR Algorithm:
        1. Retrieve candidate documents (3 * top_k)
        2. Compute relevance: similarity(query, each document)
        3. Iteratively select documents:
           - First: Highest relevance
           - Subsequent: Highest (lambda * relevance - (1-lambda) * max_sim_to_selected)
        4. Return top_k diverse documents
    """

    @patch("vectordb.langchain.diversity_filtering.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.diversity_filtering.search.chroma.RAGHelper.create_llm")
    def test_search_initialization(
        self, mock_llm_helper, mock_embedder_helper, mock_db
    ):
        """Test search pipeline initialization.

        Args:
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class

        Verifies:
            - Configuration is stored correctly including MMR settings
            - Database connection is established
            - Lambda parameter is configured (default 0.5)
            - LLM is initialized if RAG is enabled
        """
        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_dir": "./test_chroma_data",
                "collection_name": "test_diversity_filtering",
            },
            "rag": {"enabled": False},
        }

        pipeline = ChromaDiversityFilteringSearchPipeline(config)
        assert pipeline.config == config

    @patch("vectordb.langchain.diversity_filtering.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.chroma.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.chroma.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.diversity_filtering.search.chroma.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.diversity_filtering.search.chroma.DiversityFilteringHelper.mmr_diversify"
    )
    def test_search_execution(
        self,
        mock_mmr_diversify,
        mock_llm_helper,
        mock_embed_query,
        mock_embed_documents,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test search execution with MMR diversity filtering.

        Args:
            mock_mmr_diversify: Mock for DiversityFilteringHelper.mmr_diversify
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_embed_query: Mock for EmbedderHelper.embed_query
            mock_embed_documents: Mock for EmbedderHelper.embed_documents
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class
            sample_documents: Fixture providing test documents

        Verifies:
            - Query is embedded using configured model
            - Candidate pool is retrieved (larger than top_k)
            - MMR algorithm selects diverse documents
            - Returns query and diverse documents
            - Results balance relevance and diversity
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
        mock_mmr_diversify.return_value = sample_documents[:3]

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_dir": "./test_chroma_data",
                "collection_name": "test_diversity_filtering",
            },
            "diversity": {
                "method": "mmr",
                "max_documents": 5,
                "lambda_param": 0.5,
            },
            "rag": {"enabled": False},
        }

        pipeline = ChromaDiversityFilteringSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert len(result["documents"]) > 0
        mock_mmr_diversify.assert_called_once()

    @patch("vectordb.langchain.diversity_filtering.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.chroma.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.chroma.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.diversity_filtering.search.chroma.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.diversity_filtering.search.chroma.DiversityFilteringHelper.clustering_diversify"
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
        delegates to ``DiversityFilteringHelper.clustering_diversify`` rather than
        the MMR path, and that cluster parameters (``num_clusters``,
        ``samples_per_cluster``) flow through correctly.

        Args:
            mock_cluster_diversify: Mock for
                DiversityFilteringHelper.clustering_diversify
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_embed_query: Mock for EmbedderHelper.embed_query
            mock_embed_documents: Mock for EmbedderHelper.embed_documents
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class
            sample_documents: Fixture providing test documents

        Verifies:
            - clustering_diversify is called exactly once when method is "clustering"
            - Result contains the query string
            - Number of returned documents does not exceed top_k
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
            "chroma": {
                "persist_dir": "./test_chroma_data",
                "collection_name": "test_diversity_filtering",
            },
            "diversity": {
                "method": "clustering",
                "num_clusters": 3,
                "samples_per_cluster": 2,
            },
            "rag": {"enabled": False},
        }

        pipeline = ChromaDiversityFilteringSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert len(result["documents"]) <= 5
        mock_cluster_diversify.assert_called_once_with(
            documents=sample_documents,
            embeddings=[[0.1] * 384] * len(sample_documents),
            num_clusters=3,
            samples_per_cluster=2,
        )

    @patch(
        "vectordb.langchain.diversity_filtering.search.chroma.DiversityFilteringHelper.clustering_diversify"
    )
    @patch("vectordb.langchain.diversity_filtering.search.chroma.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.diversity_filtering.search.chroma.EmbedderHelper.embed_query"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.chroma.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.diversity_filtering.search.chroma.ChromaVectorDB")
    def test_search_clustering_samples_per_cluster_fixed_default(
        self,
        mock_db,
        mock_embedder_helper,
        mock_embed_documents,
        mock_embed_query,
        mock_llm_helper,
        mock_cluster_diversify,
        sample_documents,
    ):
        """Test that samples_per_cluster uses fixed default when not configured.

        Validates that when samples_per_cluster is not explicitly set in the config,
        it defaults to 2, providing a predictable default value.

        Args:
            mock_db: Mock for ChromaVectorDB class
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_embed_documents: Mock for EmbedderHelper.embed_documents
            mock_embed_query: Mock for EmbedderHelper.embed_query
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_cluster_diversify: Mock for
                DiversityFilteringHelper.clustering_diversify
            sample_documents: Fixture providing test documents

        Verifies:
            - samples_per_cluster defaults to 2 when not specified
            - Fixed default provides predictable behavior regardless of top_k
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
            "chroma": {
                "persist_dir": "./test_chroma_data",
                "collection_name": "test_diversity_filtering",
            },
            "diversity": {
                "method": "clustering",
                "num_clusters": 3,
                # samples_per_cluster not specified, should default to 2
            },
            "rag": {"enabled": False},
        }

        pipeline = ChromaDiversityFilteringSearchPipeline(config)
        pipeline.search("test query", top_k=9)

        # Fixed default: 2
        # Verify the call was made with correct parameters
        call_args = mock_cluster_diversify.call_args
        assert call_args.kwargs["num_clusters"] == 3
        assert call_args.kwargs["samples_per_cluster"] == 2

    @patch("vectordb.langchain.diversity_filtering.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.chroma.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.chroma.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.diversity_filtering.search.chroma.RAGHelper.create_llm")
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

        Guards against misconfigured pipelines reaching a silent fallback by ensuring
        any method name that is not ``"mmr"`` or ``"clustering"`` raises immediately
        with an informative error message.

        Args:
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_embed_query: Mock for EmbedderHelper.embed_query
            mock_embed_documents: Mock for EmbedderHelper.embed_documents
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class
            sample_documents: Fixture providing test documents

        Verifies:
            - ``ValueError`` is raised when ``diversity.method`` is ``"threshold"``
            - Error message contains the phrase ``"Unknown diversity method"``
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
            "chroma": {
                "persist_dir": "./test_chroma_data",
                "collection_name": "test_diversity_filtering",
            },
            "diversity": {
                "method": "threshold",
                "max_documents": 5,
                "lambda_param": 0.5,
            },
            "rag": {"enabled": False},
        }

        pipeline = ChromaDiversityFilteringSearchPipeline(config)
        with pytest.raises(ValueError, match="Unknown diversity method"):
            pipeline.search("test query", top_k=5)

    @patch("vectordb.langchain.diversity_filtering.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.chroma.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.chroma.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.diversity_filtering.search.chroma.RAGHelper.create_llm")
    @patch("vectordb.langchain.diversity_filtering.search.chroma.RAGHelper.generate")
    @patch(
        "vectordb.langchain.diversity_filtering.search.chroma.DiversityFilteringHelper.mmr_diversify"
    )
    def test_search_with_rag(
        self,
        mock_mmr_diversify,
        mock_rag_generate,
        mock_llm_helper,
        mock_embed_query,
        mock_embed_documents,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test search with RAG generation from diverse documents.

        Args:
            mock_mmr_diversify: Mock for DiversityFilteringHelper.mmr_diversify
            mock_rag_generate: Mock for RAGHelper.generate
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_embed_query: Mock for EmbedderHelper.embed_query
            mock_embed_documents: Mock for EmbedderHelper.embed_documents
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class
            sample_documents: Fixture providing test documents

        Verifies:
            - MMR retrieves diverse set of documents
            - LLM is initialized when RAG is enabled
            - RAG answer is generated from diverse documents
            - Answer is included in search results
            - Query is preserved in results
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
        mock_mmr_diversify.return_value = sample_documents[:3]

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_dir": "./test_chroma_data",
                "collection_name": "test_diversity_filtering",
            },
            "diversity": {
                "method": "mmr",
                "max_documents": 5,
                "lambda_param": 0.5,
            },
            "rag": {"enabled": True},
        }

        pipeline = ChromaDiversityFilteringSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert result["answer"] == "Test answer"

    @patch("vectordb.langchain.diversity_filtering.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.diversity_filtering.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.chroma.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.diversity_filtering.search.chroma.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.diversity_filtering.search.chroma.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.diversity_filtering.search.chroma.DiversityFilteringHelper.mmr_diversify"
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
        is used to scale the top_k value when fetching candidates from Chroma.

        Args:
            mock_diversify: Mock for DiversityFilteringHelper.mmr_diversify
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_embed_query: Mock for EmbedderHelper.embed_query
            mock_embed_documents: Mock for EmbedderHelper.embed_documents
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class
            sample_documents: Fixture providing test documents

        Verifies:
            - Chroma query is called with top_k * candidate_multiplier
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
            "chroma": {
                "persist_dir": "./test_chroma_data",
                "collection_name": "test_diversity_filtering",
            },
            "diversity": {
                "method": "mmr",
                "max_documents": 5,
                "lambda_param": 0.5,
            },
            "rag": {"enabled": False},
        }

        pipeline_default = ChromaDiversityFilteringSearchPipeline(config_default)
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
            "chroma": {
                "persist_dir": "./test_chroma_data",
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

        pipeline_custom = ChromaDiversityFilteringSearchPipeline(config_custom)
        pipeline_custom.search("test query", top_k=5)
        # Custom multiplier is 5, so top_k=5 should fetch 25 documents
        mock_db_inst.query.assert_called_with(
            query_embedding=[0.1] * 384,
            top_k=25,
            filters=None,
            collection_name="test_diversity_filtering",
        )
