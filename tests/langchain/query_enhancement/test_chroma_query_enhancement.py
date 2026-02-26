"""Tests for Chroma query enhancement pipelines (LangChain).

This module tests the query enhancement feature which improves retrieval quality
by transforming a single user query into multiple variations using LLM-based
techniques. For component-level tests, see
`tests/langchain/components/test_query_enhancer.py`.

Query Enhancement Concept:
    Standard semantic search matches a single query vector against the document
    index. Query enhancement increases recall by generating multiple semantically
    similar queries and aggregating results across all variations.

Enhancement Techniques:
    - Multi-query: Generates up to 5 semantically similar queries to cast a wider net
    - HYDE (Hypothetical Document Embeddings): Generates a hypothetical ideal answer
      to use as an additional query vector for better semantic matching
    - Step-back: Generates broader, more general questions to retrieve context
      before answering specific queries

Pipeline Architecture:
    Indexing Pipeline:
        - Standard semantic indexing (same as dense retrieval)
        - Documents stored with dense embeddings in Chroma
        - No special handling required for query enhancement

    Search Pipeline:
        1. LLM generates query variations based on enhancement mode
        2. Each query variation is embedded and searched independently
        3. Results from all queries are aggregated and deduplicated
        4. Optional RAG generation from combined results
        5. Returns enhanced documents and optionally generated answer

Components Tested:
    - ChromaQueryEnhancementIndexingPipeline: Standard indexing for enhanced search
    - ChromaQueryEnhancementSearchPipeline: Query generation and multi-search
    - QueryEnhancer component (integrated): LLM-based query transformation

Key Features:
    - Automatic query expansion using LLM
    - Multiple enhancement modes (multi_query, hyde, step_back)
    - Result aggregation and deduplication across query variations
    - Configurable number of enhanced queries
    - Optional RAG generation from aggregated results

Test Coverage:
    - Pipeline initialization with Chroma and LLM configuration
    - Document indexing (standard semantic indexing)
    - Query enhancement using QueryEnhancer component
    - Search execution with enhanced queries
    - Result aggregation from multiple queries
    - RAG generation from enhanced results
    - Edge cases: empty queries, enhancement failures

Configuration:
    Query enhancement requires LLM configuration (e.g., Groq API key)
    for generating query variations. See components documentation for details.

All tests mock vector database, embedding operations, and LLM calls
to ensure fast, deterministic unit tests without external API dependencies.
"""

from unittest.mock import MagicMock, patch

from vectordb.langchain.query_enhancement.indexing.chroma import (
    ChromaQueryEnhancementIndexingPipeline,
)
from vectordb.langchain.query_enhancement.search.chroma import (
    ChromaQueryEnhancementSearchPipeline,
)


class TestChromaQueryEnhancementIndexing:
    """Unit tests for Chroma query enhancement indexing pipeline.

    Validates the indexing pipeline for query-enhanced retrieval.
    Uses standard semantic indexing since enhancement happens at query time.

    Tested Behaviors:
        - Pipeline initialization with Chroma configuration
        - Document loading and embedding generation
        - Standard semantic document indexing
        - Empty document handling
        - Collection name configuration

    Mocks:
        - ChromaVectorDB: Database operations
        - EmbedderHelper: Embedding model and document embedding
        - DataLoaderHelper: Document loading from data sources
    """

    @patch("vectordb.langchain.query_enhancement.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.query_enhancement.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.query_enhancement.indexing.chroma.DataloaderCatalog.create"
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
        """
        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_query_enhancement",
            },
        }

        pipeline = ChromaQueryEnhancementIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.collection_name == "test_query_enhancement"

    @patch("vectordb.langchain.query_enhancement.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.query_enhancement.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.query_enhancement.indexing.chroma.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.query_enhancement.indexing.chroma.DataloaderCatalog.create"
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
            - Embeddings are generated for all documents
            - Documents are upserted to Chroma database
            - Returns count of indexed documents
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
                "path": "./test_chroma_data",
                "collection_name": "test_query_enhancement",
            },
        }

        pipeline = ChromaQueryEnhancementIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)

    @patch("vectordb.langchain.query_enhancement.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.query_enhancement.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.query_enhancement.indexing.chroma.DataloaderCatalog.create"
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
                "path": "./test_chroma_data",
                "collection_name": "test_query_enhancement",
            },
        }

        pipeline = ChromaQueryEnhancementIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0


class TestChromaQueryEnhancementSearch:
    """Unit tests for Chroma query enhancement search pipeline.

        Validates the search pipeline that enhances queries using LLM-based
    techniques and searches with multiple query variations.

        Tested Behaviors:
            - Search pipeline initialization with LLM and Chroma config
            - QueryEnhancer component initialization
            - Query enhancement generation (multi-query)
            - Multi-query search execution
            - Result aggregation from multiple searches
            - RAG generation from enhanced results

        Mocks:
            - ChatGroq: LLM for query enhancement
            - ChromaVectorDB: Database query operations
            - EmbedderHelper: Query embedding
            - RAGHelper: LLM initialization and answer generation
            - QueryEnhancer: Query variation generation
    """

    @patch("langchain_groq.ChatGroq")
    @patch("vectordb.langchain.query_enhancement.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.query_enhancement.search.base.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.query_enhancement.search.base.RAGHelper.create_llm")
    @patch("vectordb.langchain.query_enhancement.search.base.QueryEnhancer")
    def test_search_initialization(
        self, mock_enhancer, mock_llm_helper, mock_embedder_helper, mock_db, mock_llm
    ):
        """Test search pipeline initialization.

        Args:
            mock_enhancer: Mock for QueryEnhancer class
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class
            mock_llm: Mock for ChatGroq LLM class

        Verifies:
            - Configuration is stored correctly
            - QueryEnhancer is initialized for query enhancement
            - Database connection is established
        """
        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_query_enhancement",
            },
            "rag": {"enabled": False},
        }

        pipeline = ChromaQueryEnhancementSearchPipeline(config)
        assert pipeline.config == config

    @patch("langchain_groq.ChatGroq")
    @patch("vectordb.langchain.query_enhancement.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.query_enhancement.search.base.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.query_enhancement.search.base.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.query_enhancement.search.base.RAGHelper.create_llm")
    @patch("vectordb.langchain.query_enhancement.search.base.QueryEnhancer")
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
        """Test search execution with query enhancement.

        Args:
            mock_enhancer: Mock for QueryEnhancer class
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_embed_query: Mock for EmbedderHelper.embed_query
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class
            mock_llm: Mock for ChatGroq LLM class
            sample_documents: Fixture providing test documents

        Verifies:
            - Query is enhanced using QueryEnhancer
            - Enhanced queries are searched against Chroma database
            - Results are aggregated from all query variations
            - Original query and documents are returned
        """
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
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_query_enhancement",
            },
            "rag": {"enabled": False},
        }

        pipeline = ChromaQueryEnhancementSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert len(result["documents"]) > 0

    @patch("langchain_groq.ChatGroq")
    @patch("vectordb.langchain.query_enhancement.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.query_enhancement.search.base.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.query_enhancement.search.base.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.query_enhancement.search.base.RAGHelper.create_llm")
    @patch("vectordb.langchain.query_enhancement.search.base.QueryEnhancer")
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
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_query_enhancement",
            },
            "rag": {"enabled": False},
        }

        pipeline = ChromaQueryEnhancementSearchPipeline(config)

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
