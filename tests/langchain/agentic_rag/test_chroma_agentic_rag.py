"""Unit tests for Chroma agentic RAG pipelines (LangChain).

Tests validate both indexing and search pipelines for Chroma-based agentic RAG.
Agentic RAG extends traditional RAG with an iterative decision-making loop that
can search, reflect on results, and generate answers across multiple steps.

Chroma-specific aspects tested include:
- Local persistent storage path configuration
- Collection-based document organization
- Optional collection recreation for fresh starts
- Integration with AgenticRouter for decision-making
- Context compression for document pruning

Search pipeline tests validate the agentic loop with various action sequences:
- search -> generate (simplest path)
- search -> reflect -> generate (with verification)
- Direct generate without retrieval
- Fallback generation when no documents found
- Complete multi-iteration loops

These tests mock external dependencies (ChromaVectorDB, EmbedderHelper,
DataLoaderHelper, RAGHelper, AgenticRouter, ContextCompressor) to enable
fast, isolated unit tests without live Chroma instances.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vectordb.langchain.agentic_rag.indexing.chroma import (
    ChromaAgenticRAGIndexingPipeline,
)
from vectordb.langchain.agentic_rag.search.chroma import (
    ChromaAgenticRAGPipeline,
)


class TestChromaAgenticRAGIndexing:
    """Test suite for Chroma agentic RAG indexing pipeline.

    Validates the indexing pipeline's ability to:
    - Parse configuration with persistent storage path
    - Extract collection names from config
    - Load and embed documents via helper classes
    - Persist documents to local Chroma storage
    - Handle edge cases like empty documents and collection recreation
    """

    @patch("vectordb.langchain.agentic_rag.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.agentic_rag.indexing.chroma.DataloaderCatalog.create")
    def test_indexing_initialization(
        self,
        mock_get_docs: Any,
        mock_embedder_helper: Any,
        mock_db: Any,
        agentic_rag_config: Any,
    ) -> None:
        """Test pipeline initialization stores Chroma-specific configuration.

        Verifies that:
        - Configuration dict is preserved on pipeline instance
        - Collection name is extracted from chroma config section
        - Local storage path is configured for persistence
        - No external calls during initialization

        Args:
            mock_get_docs: Mock for DataloaderCatalog.create.
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db: Mock for ChromaVectorDB class.
            agentic_rag_config: Fixture with standard agentic RAG configuration.

        Returns:
            None
        """
        pipeline = ChromaAgenticRAGIndexingPipeline(agentic_rag_config)
        assert pipeline.config == agentic_rag_config
        assert pipeline.collection_name == "test_agentic_rag"

    @patch("vectordb.langchain.agentic_rag.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.agentic_rag.indexing.chroma.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.agentic_rag.indexing.chroma.DataloaderCatalog.create")
    def test_indexing_run_with_documents(
        self,
        mock_get_docs: Any,
        mock_embed_docs: Any,
        mock_embedder_helper: Any,
        mock_db: Any,
        sample_documents: Any,
    ) -> None:
        """Test successful document indexing into Chroma local storage.

        Validates the complete indexing workflow:
        1. DataLoaderHelper loads documents from ARC dataset
        2. EmbedderHelper generates 384-dimensional embeddings
        3. ChromaVectorDB.upsert persists to local storage
        4. Result reports count of indexed documents

        Args:
            mock_get_docs: Mock returning sample_documents fixture.
            mock_embed_docs: Mock returning (docs, embeddings) tuple.
            mock_embedder_helper: Mock for embedder factory.
            mock_db: Mock for ChromaVectorDB, tracks upsert calls.
            sample_documents: Fixture with 5 sample documents.

        Returns:
            None
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
                "collection_name": "test_agentic_rag",
            },
        }

        pipeline = ChromaAgenticRAGIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.upsert.assert_called_once()

    @patch("vectordb.langchain.agentic_rag.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.agentic_rag.indexing.chroma.DataloaderCatalog.create")
    def test_indexing_run_no_documents(
        self, mock_get_docs: Any, mock_embedder_helper: Any, mock_db: Any
    ) -> None:
        """Test graceful handling of empty document batches.

        Ensures when DataLoaderHelper returns empty list:
        - No exceptions raised
        - Result reports 0 documents indexed
        - No database operations attempted

        Args:
            mock_get_docs: Mock returning empty list.
            mock_embedder_helper: Mock for embedder factory.
            mock_db: Mock for ChromaVectorDB.

        Returns:
            None
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
                "collection_name": "test_agentic_rag",
            },
        }

        pipeline = ChromaAgenticRAGIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0

    @patch("vectordb.langchain.agentic_rag.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.agentic_rag.indexing.chroma.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.agentic_rag.indexing.chroma.DataloaderCatalog.create")
    def test_indexing_with_recreate_option(
        self,
        mock_get_docs: Any,
        mock_embed_docs: Any,
        mock_embedder_helper: Any,
        mock_db: Any,
        sample_documents: Any,
    ) -> None:
        """Test collection recreation for fresh index starts.

        Validates that when recreate=True:
        - Existing collection is dropped and recreated
        - Documents are then indexed normally
        - Useful for testing and data refresh scenarios

        Args:
            mock_get_docs: Mock returning sample_documents fixture.
            mock_embed_docs: Mock returning (docs, embeddings) tuple.
            mock_embedder_helper: Mock for embedder factory.
            mock_db: Mock for ChromaVectorDB, tracks create_collection.
            sample_documents: Fixture with 5 sample documents.

        Returns:
            None
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
                "collection_name": "test_agentic_rag",
                "recreate": True,
            },
        }

        pipeline = ChromaAgenticRAGIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.create_collection.assert_called_once()


class TestChromaAgenticRAGSearch:
    """Test suite for Chroma agentic RAG search pipeline initialization.

    Validates search pipeline setup with various configurations:
    - Agentic parameters (max_iterations, compression_mode)
    - LLM and reranker initialization
    - Router and compressor setup
    - Error handling for missing LLM
    """

    @patch("vectordb.langchain.agentic_rag.search.chroma.ChatGroq")
    @patch("vectordb.langchain.agentic_rag.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.agentic_rag.search.chroma.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.agentic_rag.search.chroma.RerankerHelper.create_reranker"
    )
    @patch("vectordb.langchain.agentic_rag.search.chroma.AgenticRouter")
    @patch("vectordb.langchain.agentic_rag.search.chroma.ContextCompressor")
    def test_search_initialization(
        self,
        mock_compressor: Any,
        mock_router: Any,
        mock_reranker_helper: Any,
        mock_llm_helper: Any,
        mock_embedder_helper: Any,
        mock_db: Any,
        mock_llm: Any,
        mock_reranker: Any,
    ) -> None:
        """Test search pipeline initialization with full agentic configuration.

        Verifies that:
        - Configuration is stored on pipeline instance
        - Agentic parameters extracted (max_iterations, compression_mode)
        - LLM and reranker initialized via helpers
        - Router and compressor classes instantiated
        - All components ready for agentic loop execution

        Args:
            mock_compressor: Mock for ContextCompressor class.
            mock_router: Mock for AgenticRouter class.
            mock_reranker_helper: Mock for RerankerHelper.create_reranker.
            mock_llm_helper: Mock for RAGHelper.create_llm.
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db: Mock for ChromaVectorDB.
            mock_llm: Mock LLM instance.
            mock_reranker: Mock reranker instance.

        Returns:
            None
        """
        mock_llm_helper.return_value = mock_llm
        mock_reranker_helper.return_value = mock_reranker

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_agentic_rag",
            },
            "agentic": {
                "max_iterations": 3,
                "compression_mode": "reranking",
                "router_model": "llama-3.3-70b-versatile",
            },
            "rag": {"enabled": True},
        }

        pipeline = ChromaAgenticRAGPipeline(config)
        assert pipeline.config == config
        assert pipeline.max_iterations == 3
        assert pipeline.compression_mode == "reranking"

    @patch("vectordb.langchain.agentic_rag.search.chroma.ChatGroq")
    @patch("vectordb.langchain.agentic_rag.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.agentic_rag.search.chroma.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.agentic_rag.search.chroma.RerankerHelper.create_reranker"
    )
    @patch("vectordb.langchain.agentic_rag.search.chroma.AgenticRouter")
    @patch("vectordb.langchain.agentic_rag.search.chroma.ContextCompressor")
    def test_search_with_llm_extraction_mode(
        self,
        mock_compressor: Any,
        mock_router: Any,
        mock_reranker_helper: Any,
        mock_llm_helper: Any,
        mock_embedder_helper: Any,
        mock_db: Any,
        mock_llm: Any,
    ) -> None:
        """Test pipeline initialization with LLM-based context extraction.

        Validates compression_mode="llm_extraction" configuration:
        - LLM used to extract relevant context from documents
        - Alternative to reranking-based compression
        - Different summarization strategy

        Args:
            mock_compressor: Mock for ContextCompressor class.
            mock_router: Mock for AgenticRouter class.
            mock_reranker_helper: Mock for RerankerHelper.create_reranker.
            mock_llm_helper: Mock for RAGHelper.create_llm.
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db: Mock for ChromaVectorDB.
            mock_llm: Mock LLM instance.

        Returns:
            None
        """
        mock_llm_helper.return_value = mock_llm

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_agentic_rag",
            },
            "agentic": {
                "max_iterations": 5,
                "compression_mode": "llm_extraction",
                "router_model": "llama-3.3-70b-versatile",
            },
            "rag": {"enabled": True},
        }

        pipeline = ChromaAgenticRAGPipeline(config)
        assert pipeline.compression_mode == "llm_extraction"

    @patch("vectordb.langchain.agentic_rag.search.chroma.ChatGroq")
    @patch("vectordb.langchain.agentic_rag.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.agentic_rag.search.chroma.RAGHelper.create_llm")
    def test_search_raises_error_without_llm(
        self,
        mock_llm_helper: Any,
        mock_embedder_helper: Any,
        mock_db: Any,
        mock_chat: Any,
    ) -> None:
        """Test that search pipeline requires LLM to be configured.

        Validates error handling when RAG LLM is not enabled:
        - ValueError raised with clear message
        - Pipeline cannot operate without generation capability
        - Ensures proper configuration before execution

        Args:
            mock_llm_helper: Mock returning None (no LLM).
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db: Mock for ChromaVectorDB.
            mock_chat: Mock for ChatGroq.

        Raises:
            ValueError: When rag.enabled is False or LLM not configured.

        Returns:
            None
        """
        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_agentic_rag",
            },
            "agentic": {
                "max_iterations": 3,
                "compression_mode": "reranking",
            },
        }

        with pytest.raises(ValueError, match="RAG LLM must be enabled"):
            ChromaAgenticRAGPipeline(config)


class TestChromaAgenticRAGPipelineRun:
    """Unit tests for ChromaAgenticRAGPipeline run() method execution.

    Validates the agentic decision loop with various action sequences:
    - Search action retrieves and compresses documents
    - Reflect action evaluates answer quality
    - Generate action produces final answers
    - Multiple iterations with different paths
    - Edge cases like empty results and max iteration limits
    """

    @patch("vectordb.langchain.agentic_rag.search.chroma.ChatGroq")
    @patch("vectordb.langchain.agentic_rag.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.agentic_rag.search.chroma.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.agentic_rag.search.chroma.RerankerHelper.create_reranker"
    )
    @patch("vectordb.langchain.agentic_rag.search.chroma.AgenticRouter")
    @patch("vectordb.langchain.agentic_rag.search.chroma.ContextCompressor")
    @patch("vectordb.langchain.agentic_rag.search.chroma.EmbedderHelper.embed_query")
    def test_run_with_search_action(
        self,
        mock_embed_query: Any,
        mock_compressor_class: Any,
        mock_router_class: Any,
        mock_reranker_helper: Any,
        mock_llm_helper: Any,
        mock_embedder_helper: Any,
        mock_db: Any,
        mock_chat: Any,
        sample_documents: Any,
    ) -> None:
        """Test run() with search->generate action sequence.

        Validates the search action:
        - Query embedding via EmbedderHelper.embed_query
        - Document retrieval via ChromaVectorDB.query
        - Context compression via ContextCompressor.compress
        - Result structure with documents, steps, and reasoning

        Args:
            mock_embed_query: Mock returning query vector.
            mock_compressor_class: Mock for ContextCompressor.
            mock_router_class: Mock for AgenticRouter.
            mock_reranker_helper: Mock for RerankerHelper.
            mock_llm_helper: Mock for RAGHelper.create_llm.
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db: Mock for ChromaVectorDB.
            mock_chat: Mock for ChatGroq.
            sample_documents: Fixture with sample documents.

        Returns:
            None
        """
        mock_llm = MagicMock()
        mock_llm.api_key = "test-api-key"
        mock_llm_helper.return_value = mock_llm

        mock_reranker = MagicMock()
        mock_reranker_helper.return_value = mock_reranker

        mock_embedder = MagicMock()
        mock_embedder_helper.return_value = mock_embedder
        mock_embed_query.return_value = [0.1] * 384

        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst

        mock_compressor = MagicMock()
        mock_compressor.compress.return_value = sample_documents[:3]
        mock_compressor_class.return_value = mock_compressor

        mock_router = MagicMock()
        mock_router.route.side_effect = [
            {"action": "search", "reasoning": "Need to retrieve documents"},
            {"action": "generate", "reasoning": "Generate answer from documents"},
        ]
        mock_router_class.return_value = mock_router

        with patch(
            "vectordb.langchain.agentic_rag.search.chroma.RAGHelper.generate"
        ) as mock_generate:
            mock_generate.return_value = "Generated answer from documents"

            config = {
                "dataloader": {"type": "arc", "limit": 10},
                "embeddings": {"model": "test-model", "device": "cpu"},
                "chroma": {
                    "path": "./test_chroma_data",
                    "collection_name": "test_agentic_rag",
                },
                "agentic": {
                    "max_iterations": 3,
                    "compression_mode": "reranking",
                    "router_model": "llama-3.3-70b-versatile",
                },
                "rag": {"enabled": True},
            }

            pipeline = ChromaAgenticRAGPipeline(config)
            result = pipeline.run("What is Python?")

            mock_db_inst.query.assert_called_once()
            mock_compressor.compress.assert_called_once()

            assert "final_answer" in result
            assert "documents" in result
            assert "intermediate_steps" in result
            assert "reasoning" in result
            assert len(result["intermediate_steps"]) == 2
            assert result["intermediate_steps"][0]["action"] == "search"
            assert result["intermediate_steps"][0]["documents_retrieved"] == 3
            assert result["intermediate_steps"][1]["action"] == "generate"

    @patch("vectordb.langchain.agentic_rag.search.chroma.ChatGroq")
    @patch("vectordb.langchain.agentic_rag.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.agentic_rag.search.chroma.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.agentic_rag.search.chroma.RerankerHelper.create_reranker"
    )
    @patch("vectordb.langchain.agentic_rag.search.chroma.AgenticRouter")
    @patch("vectordb.langchain.agentic_rag.search.chroma.ContextCompressor")
    @patch("vectordb.langchain.agentic_rag.search.chroma.EmbedderHelper.embed_query")
    def test_run_with_search_action_empty_documents(
        self,
        mock_embed_query: Any,
        mock_compressor_class: Any,
        mock_router_class: Any,
        mock_reranker_helper: Any,
        mock_llm_helper: Any,
        mock_embedder_helper: Any,
        mock_db: Any,
        mock_chat: Any,
    ) -> None:
        """Test run() when search returns empty documents.

        Validates the search action with empty results:
        - Query embedding via EmbedderHelper.embed_query
        - Document retrieval returns empty list
        - No compression attempted on empty results
        - Fallback to LLM for answer generation

        Args:
            mock_embed_query: Mock returning query vector.
            mock_compressor_class: Mock for ContextCompressor.
            mock_router_class: Mock for AgenticRouter.
            mock_reranker_helper: Mock for RerankerHelper.
            mock_llm_helper: Mock for RAGHelper.create_llm.
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db: Mock for ChromaVectorDB.
            mock_chat: Mock for ChatGroq.

        Returns:
            None
        """
        mock_llm = MagicMock()
        mock_llm.api_key = "test-api-key"
        mock_llm_helper.return_value = mock_llm

        mock_reranker = MagicMock()
        mock_reranker_helper.return_value = mock_reranker

        mock_embedder = MagicMock()
        mock_embedder_helper.return_value = mock_embedder
        mock_embed_query.return_value = [0.1] * 384

        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = []
        mock_db.return_value = mock_db_inst

        mock_compressor = MagicMock()
        mock_compressor_class.return_value = mock_compressor

        mock_router = MagicMock()
        mock_router.route.side_effect = [
            {"action": "search", "reasoning": "Need to retrieve documents"},
            {"action": "generate", "reasoning": "Generate answer without documents"},
        ]
        mock_router_class.return_value = mock_router

        with patch(
            "vectordb.langchain.agentic_rag.search.chroma.RAGHelper.generate"
        ) as mock_generate:
            mock_generate.return_value = "Generated answer without context"

            config = {
                "dataloader": {"type": "arc", "limit": 10},
                "embeddings": {"model": "test-model", "device": "cpu"},
                "chroma": {
                    "path": "./test_chroma_data",
                    "collection_name": "test_agentic_rag",
                },
                "agentic": {
                    "max_iterations": 3,
                    "compression_mode": "reranking",
                    "router_model": "llama-3.3-70b-versatile",
                },
                "rag": {"enabled": True},
            }

            pipeline = ChromaAgenticRAGPipeline(config)
            result = pipeline.run("What is Python?")

            mock_db_inst.query.assert_called_once()
            mock_compressor.compress.assert_not_called()

            assert "final_answer" in result
            assert result["documents"] == []
            assert len(result["intermediate_steps"]) == 2
            assert result["intermediate_steps"][0]["action"] == "search"
            assert result["intermediate_steps"][0]["documents_retrieved"] == 0
            assert result["intermediate_steps"][1]["action"] == "generate"

    @patch("vectordb.langchain.agentic_rag.search.chroma.ChatGroq")
    @patch("vectordb.langchain.agentic_rag.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.agentic_rag.search.chroma.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.agentic_rag.search.chroma.RerankerHelper.create_reranker"
    )
    @patch("vectordb.langchain.agentic_rag.search.chroma.AgenticRouter")
    @patch("vectordb.langchain.agentic_rag.search.chroma.ContextCompressor")
    @patch("vectordb.langchain.agentic_rag.search.chroma.EmbedderHelper.embed_query")
    def test_run_with_reflect_action(
        self,
        mock_embed_query: Any,
        mock_compressor_class: Any,
        mock_router_class: Any,
        mock_reranker_helper: Any,
        mock_llm_helper: Any,
        mock_embedder_helper: Any,
        mock_db: Any,
        mock_chat: Any,
        sample_documents: Any,
    ) -> None:
        """Test run() with reflect action to evaluate answer quality.

        Validates the reflect action:
        - Reflection occurs between searches before generation
        - LLM invoked with reflection prompt containing documents and answer
        - Reflection result captured in intermediate steps
        - Search -> Reflect -> Search -> Generate sequence

        Note: Generate action breaks the loop, so reflect must occur before
        the final generate action.

        Args:
            mock_embed_query: Mock returning query vector.
            mock_compressor_class: Mock for ContextCompressor.
            mock_router_class: Mock for AgenticRouter.
            mock_reranker_helper: Mock for RerankerHelper.
            mock_llm_helper: Mock for RAGHelper.create_llm.
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db: Mock for ChromaVectorDB.
            mock_chat: Mock for ChatGroq.
            sample_documents: Fixture with sample documents.

        Returns:
            None
        """
        mock_llm = MagicMock()
        mock_llm.api_key = "test-api-key"
        mock_llm.invoke.return_value = MagicMock(
            content="Reflection: Answer could be improved with more detail"
        )
        mock_llm_helper.return_value = mock_llm

        mock_reranker = MagicMock()
        mock_reranker_helper.return_value = mock_reranker

        mock_embedder = MagicMock()
        mock_embedder_helper.return_value = mock_embedder
        mock_embed_query.return_value = [0.1] * 384

        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst

        mock_compressor = MagicMock()
        mock_compressor.compress.return_value = sample_documents[:3]
        mock_compressor_class.return_value = mock_compressor

        mock_router = MagicMock()
        mock_router.route.side_effect = [
            {"action": "search", "reasoning": "Need to retrieve documents"},
            {"action": "reflect", "reasoning": "Evaluate document quality"},
            {"action": "search", "reasoning": "Search for more context"},
            {"action": "generate", "reasoning": "Generate final answer"},
        ]
        mock_router_class.return_value = mock_router

        with patch(
            "vectordb.langchain.agentic_rag.search.chroma.RAGHelper.generate"
        ) as mock_generate:
            mock_generate.return_value = "Final generated answer"

            config = {
                "dataloader": {"type": "arc", "limit": 10},
                "embeddings": {"model": "test-model", "device": "cpu"},
                "chroma": {
                    "path": "./test_chroma_data",
                    "collection_name": "test_agentic_rag",
                },
                "agentic": {
                    "max_iterations": 5,
                    "compression_mode": "reranking",
                    "router_model": "llama-3.3-70b-versatile",
                },
                "rag": {"enabled": True},
            }

            pipeline = ChromaAgenticRAGPipeline(config)
            result = pipeline.run("What is Python?")

            assert "final_answer" in result
            assert "documents" in result
            assert len(result["intermediate_steps"]) == 4
            assert result["intermediate_steps"][0]["action"] == "search"
            assert result["intermediate_steps"][1]["action"] == "reflect"
            assert result["intermediate_steps"][2]["action"] == "search"
            assert result["intermediate_steps"][3]["action"] == "generate"

    @patch("vectordb.langchain.agentic_rag.search.chroma.ChatGroq")
    @patch("vectordb.langchain.agentic_rag.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.agentic_rag.search.chroma.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.agentic_rag.search.chroma.RerankerHelper.create_reranker"
    )
    @patch("vectordb.langchain.agentic_rag.search.chroma.AgenticRouter")
    @patch("vectordb.langchain.agentic_rag.search.chroma.ContextCompressor")
    @patch("vectordb.langchain.agentic_rag.search.chroma.EmbedderHelper.embed_query")
    def test_run_with_reflect_action_no_documents(
        self,
        mock_embed_query: Any,
        mock_compressor_class: Any,
        mock_router_class: Any,
        mock_reranker_helper: Any,
        mock_llm_helper: Any,
        mock_embedder_helper: Any,
        mock_db: Any,
        mock_chat: Any,
    ) -> None:
        """Test run() with reflect action when no documents available.

        Validates reflection behavior with missing documents:
        - Reflection only occurs when both documents and current_answer exist
        - No reflection performed if documents list is empty
        - Pipeline proceeds to generate without reflection step

        Args:
            mock_embed_query: Mock returning query vector.
            mock_compressor_class: Mock for ContextCompressor.
            mock_router_class: Mock for AgenticRouter.
            mock_reranker_helper: Mock for RerankerHelper.
            mock_llm_helper: Mock for RAGHelper.create_llm.
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db: Mock for ChromaVectorDB.
            mock_chat: Mock for ChatGroq.

        Returns:
            None
        """
        mock_llm = MagicMock()
        mock_llm.api_key = "test-api-key"
        mock_llm.invoke.return_value = MagicMock(content="Fallback answer")
        mock_llm_helper.return_value = mock_llm

        mock_reranker = MagicMock()
        mock_reranker_helper.return_value = mock_reranker

        mock_embedder = MagicMock()
        mock_embedder_helper.return_value = mock_embedder
        mock_embed_query.return_value = [0.1] * 384

        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = []
        mock_db.return_value = mock_db_inst

        mock_compressor = MagicMock()
        mock_compressor_class.return_value = mock_compressor

        mock_router = MagicMock()
        mock_router.route.side_effect = [
            {"action": "search", "reasoning": "Need to retrieve documents"},
            {"action": "reflect", "reasoning": "Should reflect on answer"},
            {"action": "generate", "reasoning": "Generate answer"},
        ]
        mock_router_class.return_value = mock_router

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_agentic_rag",
            },
            "agentic": {
                "max_iterations": 3,
                "compression_mode": "reranking",
                "router_model": "llama-3.3-70b-versatile",
            },
            "rag": {"enabled": True},
        }

        pipeline = ChromaAgenticRAGPipeline(config)
        result = pipeline.run("What is Python?")

        assert "final_answer" in result
        assert result["documents"] == []
        assert len(result["intermediate_steps"]) == 3
        assert result["intermediate_steps"][0]["action"] == "search"
        assert result["intermediate_steps"][1]["action"] == "reflect"
        assert "reflection" not in result["intermediate_steps"][1]
        assert result["intermediate_steps"][2]["action"] == "generate"

    @patch("vectordb.langchain.agentic_rag.search.chroma.ChatGroq")
    @patch("vectordb.langchain.agentic_rag.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.agentic_rag.search.chroma.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.agentic_rag.search.chroma.RerankerHelper.create_reranker"
    )
    @patch("vectordb.langchain.agentic_rag.search.chroma.AgenticRouter")
    @patch("vectordb.langchain.agentic_rag.search.chroma.ContextCompressor")
    @patch("vectordb.langchain.agentic_rag.search.chroma.EmbedderHelper.embed_query")
    def test_run_with_generate_action_with_documents(
        self,
        mock_embed_query: Any,
        mock_compressor_class: Any,
        mock_router_class: Any,
        mock_reranker_helper: Any,
        mock_llm_helper: Any,
        mock_embedder_helper: Any,
        mock_db: Any,
        mock_chat: Any,
        sample_documents: Any,
    ) -> None:
        """Test run() with generate action using retrieved documents.

        Validates the generate action with context:
        - RAGHelper.generate called with LLM, query, and documents
        - Documents passed to generation function
        - Final answer includes context from retrieved documents
        - Loop exits immediately after generation

        Args:
            mock_embed_query: Mock returning query vector.
            mock_compressor_class: Mock for ContextCompressor.
            mock_router_class: Mock for AgenticRouter.
            mock_reranker_helper: Mock for RerankerHelper.
            mock_llm_helper: Mock for RAGHelper.create_llm.
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db: Mock for ChromaVectorDB.
            mock_chat: Mock for ChatGroq.
            sample_documents: Fixture with sample documents.

        Returns:
            None
        """
        mock_llm = MagicMock()
        mock_llm.api_key = "test-api-key"
        mock_llm_helper.return_value = mock_llm

        mock_reranker = MagicMock()
        mock_reranker_helper.return_value = mock_reranker

        mock_embedder = MagicMock()
        mock_embedder_helper.return_value = mock_embedder
        mock_embed_query.return_value = [0.1] * 384

        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst

        mock_compressor = MagicMock()
        mock_compressor.compress.return_value = sample_documents[:3]
        mock_compressor_class.return_value = mock_compressor

        mock_router = MagicMock()
        mock_router.route.side_effect = [
            {"action": "search", "reasoning": "Need to retrieve documents"},
            {"action": "generate", "reasoning": "Generate answer with context"},
        ]
        mock_router_class.return_value = mock_router

        with patch(
            "vectordb.langchain.agentic_rag.search.chroma.RAGHelper.generate"
        ) as mock_generate:
            mock_generate.return_value = "Answer based on retrieved documents"

            config = {
                "dataloader": {"type": "arc", "limit": 10},
                "embeddings": {"model": "test-model", "device": "cpu"},
                "chroma": {
                    "path": "./test_chroma_data",
                    "collection_name": "test_agentic_rag",
                },
                "agentic": {
                    "max_iterations": 3,
                    "compression_mode": "reranking",
                    "router_model": "llama-3.3-70b-versatile",
                },
                "rag": {"enabled": True},
            }

            pipeline = ChromaAgenticRAGPipeline(config)
            result = pipeline.run("What is Python?")

            mock_generate.assert_called_once_with(
                mock_llm, "What is Python?", sample_documents[:3]
            )

            assert "final_answer" in result
            assert result["final_answer"] == "Answer based on retrieved documents"
            assert len(result["intermediate_steps"]) == 2
            assert result["intermediate_steps"][1]["answer_generated"] is True

    @patch("vectordb.langchain.agentic_rag.search.chroma.ChatGroq")
    @patch("vectordb.langchain.agentic_rag.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.agentic_rag.search.chroma.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.agentic_rag.search.chroma.RerankerHelper.create_reranker"
    )
    @patch("vectordb.langchain.agentic_rag.search.chroma.AgenticRouter")
    @patch("vectordb.langchain.agentic_rag.search.chroma.ContextCompressor")
    @patch("vectordb.langchain.agentic_rag.search.chroma.EmbedderHelper.embed_query")
    def test_run_with_generate_action_no_documents(
        self,
        mock_embed_query: Any,
        mock_compressor_class: Any,
        mock_router_class: Any,
        mock_reranker_helper: Any,
        mock_llm_helper: Any,
        mock_embedder_helper: Any,
        mock_db: Any,
        mock_chat: Any,
    ) -> None:
        """Test run() with generate action when no documents retrieved.

        Validates fallback generation without documents:
        - Direct LLM invoke used when no documents available
        - LLM generates answer based on its internal knowledge
        - No RAGHelper.generate call made
        - Loop exits after generation

        Args:
            mock_embed_query: Mock returning query vector.
            mock_compressor_class: Mock for ContextCompressor.
            mock_router_class: Mock for AgenticRouter.
            mock_reranker_helper: Mock for RerankerHelper.
            mock_llm_helper: Mock for RAGHelper.create_llm.
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db: Mock for ChromaVectorDB.
            mock_chat: Mock for ChatGroq.

        Returns:
            None
        """
        mock_llm = MagicMock()
        mock_llm.api_key = "test-api-key"
        mock_llm.invoke.return_value = MagicMock(content="LLM knowledge-based answer")
        mock_llm_helper.return_value = mock_llm

        mock_reranker = MagicMock()
        mock_reranker_helper.return_value = mock_reranker

        mock_embedder = MagicMock()
        mock_embedder_helper.return_value = mock_embedder
        mock_embed_query.return_value = [0.1] * 384

        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = []
        mock_db.return_value = mock_db_inst

        mock_compressor = MagicMock()
        mock_compressor_class.return_value = mock_compressor

        mock_router = MagicMock()
        mock_router.route.side_effect = [
            {"action": "search", "reasoning": "Need to retrieve documents"},
            {"action": "generate", "reasoning": "Generate answer without context"},
        ]
        mock_router_class.return_value = mock_router

        with patch(
            "vectordb.langchain.agentic_rag.search.chroma.RAGHelper.generate"
        ) as mock_generate:
            config = {
                "dataloader": {"type": "arc", "limit": 10},
                "embeddings": {"model": "test-model", "device": "cpu"},
                "chroma": {
                    "path": "./test_chroma_data",
                    "collection_name": "test_agentic_rag",
                },
                "agentic": {
                    "max_iterations": 3,
                    "compression_mode": "reranking",
                    "router_model": "llama-3.3-70b-versatile",
                },
                "rag": {"enabled": True},
            }

            pipeline = ChromaAgenticRAGPipeline(config)
            result = pipeline.run("What is Python?")

            mock_generate.assert_not_called()
            mock_llm.invoke.assert_called_with("What is Python?")

            assert "final_answer" in result
            assert result["final_answer"] == "LLM knowledge-based answer"
            assert result["documents"] == []
            assert len(result["intermediate_steps"]) == 2

    @patch("vectordb.langchain.agentic_rag.search.chroma.ChatGroq")
    @patch("vectordb.langchain.agentic_rag.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.agentic_rag.search.chroma.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.agentic_rag.search.chroma.RerankerHelper.create_reranker"
    )
    @patch("vectordb.langchain.agentic_rag.search.chroma.AgenticRouter")
    @patch("vectordb.langchain.agentic_rag.search.chroma.ContextCompressor")
    @patch("vectordb.langchain.agentic_rag.search.chroma.EmbedderHelper.embed_query")
    def test_run_fallback_answer_generation(
        self,
        mock_embed_query: Any,
        mock_compressor_class: Any,
        mock_router_class: Any,
        mock_reranker_helper: Any,
        mock_llm_helper: Any,
        mock_embedder_helper: Any,
        mock_db: Any,
        mock_chat: Any,
        sample_documents: Any,
    ) -> None:
        """Test run() fallback when no generation occurs within iteration limit.

        Validates fallback mechanism:
        - When router never selects generate action
        - After max_iterations reached without answer
        - LLM directly invoked with original query as fallback
        - Warning logged about missing generation

        Args:
            mock_embed_query: Mock returning query vector.
            mock_compressor_class: Mock for ContextCompressor.
            mock_router_class: Mock for AgenticRouter.
            mock_reranker_helper: Mock for RerankerHelper.
            mock_llm_helper: Mock for RAGHelper.create_llm.
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db: Mock for ChromaVectorDB.
            mock_chat: Mock for ChatGroq.
            sample_documents: Fixture with sample documents.

        Returns:
            None
        """
        mock_llm = MagicMock()
        mock_llm.api_key = "test-api-key"
        mock_llm.invoke.return_value = MagicMock(content="Fallback LLM answer")
        mock_llm_helper.return_value = mock_llm

        mock_reranker = MagicMock()
        mock_reranker_helper.return_value = mock_reranker

        mock_embedder = MagicMock()
        mock_embedder_helper.return_value = mock_embedder
        mock_embed_query.return_value = [0.1] * 384

        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst

        mock_compressor = MagicMock()
        mock_compressor.compress.return_value = sample_documents[:3]
        mock_compressor_class.return_value = mock_compressor

        mock_router = MagicMock()
        mock_router.route.side_effect = [
            {"action": "search", "reasoning": "Iteration 1: search"},
            {"action": "search", "reasoning": "Iteration 2: search again"},
            {
                "action": "search",
                "reasoning": "Iteration 3: search again to hit max iterations",
            },
        ]
        mock_router_class.return_value = mock_router

        with patch(
            "vectordb.langchain.agentic_rag.search.chroma.RAGHelper.generate"
        ) as mock_generate:
            config = {
                "dataloader": {"type": "arc", "limit": 10},
                "embeddings": {"model": "test-model", "device": "cpu"},
                "chroma": {
                    "path": "./test_chroma_data",
                    "collection_name": "test_agentic_rag",
                },
                "agentic": {
                    "max_iterations": 3,
                    "compression_mode": "reranking",
                    "router_model": "llama-3.3-70b-versatile",
                },
                "rag": {"enabled": True},
            }

            pipeline = ChromaAgenticRAGPipeline(config)
            result = pipeline.run("What is Python?")

            mock_generate.assert_not_called()
            mock_llm.invoke.assert_called_with("What is Python?")

            assert "final_answer" in result
            assert result["final_answer"] == "Fallback LLM answer"
            assert len(result["intermediate_steps"]) == 3

    @patch("vectordb.langchain.agentic_rag.search.chroma.ChatGroq")
    @patch("vectordb.langchain.agentic_rag.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.agentic_rag.search.chroma.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.agentic_rag.search.chroma.RerankerHelper.create_reranker"
    )
    @patch("vectordb.langchain.agentic_rag.search.chroma.AgenticRouter")
    @patch("vectordb.langchain.agentic_rag.search.chroma.ContextCompressor")
    @patch("vectordb.langchain.agentic_rag.search.chroma.EmbedderHelper.embed_query")
    def test_run_complete_agentic_loop(
        self,
        mock_embed_query: Any,
        mock_compressor_class: Any,
        mock_router_class: Any,
        mock_reranker_helper: Any,
        mock_llm_helper: Any,
        mock_embedder_helper: Any,
        mock_db: Any,
        mock_chat: Any,
        sample_documents: Any,
    ) -> None:
        """Test run() complete agentic loop with multiple iterations.

        Validates complex multi-step reasoning:
        - Search -> Reflect -> Search -> Generate sequence
        - Multiple search iterations with different reasoning
        - Reflection step between searches
        - Final generation breaks the loop
        - Complete trace in intermediate_steps

        Note: Generate action breaks the loop immediately, so it must be last.

        Args:
            mock_embed_query: Mock returning query vector.
            mock_compressor_class: Mock for ContextCompressor.
            mock_router_class: Mock for AgenticRouter.
            mock_reranker_helper: Mock for RerankerHelper.
            mock_llm_helper: Mock for RAGHelper.create_llm.
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db: Mock for ChromaVectorDB.
            mock_chat: Mock for ChatGroq.
            sample_documents: Fixture with sample documents.

        Returns:
            None
        """
        mock_llm = MagicMock()
        mock_llm.api_key = "test-api-key"
        mock_llm.invoke.return_value = MagicMock(content="Reflection feedback")
        mock_llm_helper.return_value = mock_llm

        mock_reranker = MagicMock()
        mock_reranker_helper.return_value = mock_reranker

        mock_embedder = MagicMock()
        mock_embedder_helper.return_value = mock_embedder
        mock_embed_query.return_value = [0.1] * 384

        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst

        mock_compressor = MagicMock()
        mock_compressor.compress.return_value = sample_documents[:3]
        mock_compressor_class.return_value = mock_compressor

        mock_router = MagicMock()
        mock_router.route.side_effect = [
            {"action": "search", "reasoning": "Initial search for context"},
            {"action": "reflect", "reasoning": "Check if more info needed"},
            {"action": "search", "reasoning": "Need more specific documents"},
            {"action": "generate", "reasoning": "Sufficient information gathered"},
        ]
        mock_router_class.return_value = mock_router

        with patch(
            "vectordb.langchain.agentic_rag.search.chroma.RAGHelper.generate"
        ) as mock_generate:
            mock_generate.return_value = (
                "Comprehensive answer after multiple iterations"
            )

            config = {
                "dataloader": {"type": "arc", "limit": 10},
                "embeddings": {"model": "test-model", "device": "cpu"},
                "chroma": {
                    "path": "./test_chroma_data",
                    "collection_name": "test_agentic_rag",
                },
                "agentic": {
                    "max_iterations": 5,
                    "compression_mode": "reranking",
                    "router_model": "llama-3.3-70b-versatile",
                },
                "rag": {"enabled": True},
            }

            pipeline = ChromaAgenticRAGPipeline(config)
            result = pipeline.run("What is Python?")

            assert mock_db_inst.query.call_count == 2

            assert "final_answer" in result
            assert (
                result["final_answer"]
                == "Comprehensive answer after multiple iterations"
            )
            assert len(result["intermediate_steps"]) == 4
            assert result["intermediate_steps"][0]["action"] == "search"
            assert result["intermediate_steps"][1]["action"] == "reflect"
            assert result["intermediate_steps"][2]["action"] == "search"
            assert result["intermediate_steps"][3]["action"] == "generate"
            assert result["intermediate_steps"][3]["answer_generated"] is True

    @patch("vectordb.langchain.agentic_rag.search.chroma.ChatGroq")
    @patch("vectordb.langchain.agentic_rag.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.agentic_rag.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.agentic_rag.search.chroma.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.agentic_rag.search.chroma.RerankerHelper.create_reranker"
    )
    @patch("vectordb.langchain.agentic_rag.search.chroma.AgenticRouter")
    @patch("vectordb.langchain.agentic_rag.search.chroma.ContextCompressor")
    @patch("vectordb.langchain.agentic_rag.search.chroma.EmbedderHelper.embed_query")
    def test_run_with_filters(
        self,
        mock_embed_query: Any,
        mock_compressor_class: Any,
        mock_router_class: Any,
        mock_reranker_helper: Any,
        mock_llm_helper: Any,
        mock_embedder_helper: Any,
        mock_db: Any,
        mock_chat: Any,
        sample_documents: Any,
    ) -> None:
        """Test run() with metadata filters applied to search.

        Validates filter parameter passing:
        - Filters passed through to ChromaVectorDB.query
        - Metadata filtering restricts returned documents
        - Filtered documents compressed and used for generation
        - Filter parameter preserved across iterations

        Args:
            mock_embed_query: Mock returning query vector.
            mock_compressor_class: Mock for ContextCompressor.
            mock_router_class: Mock for AgenticRouter.
            mock_reranker_helper: Mock for RerankerHelper.
            mock_llm_helper: Mock for RAGHelper.create_llm.
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db: Mock for ChromaVectorDB.
            mock_chat: Mock for ChatGroq.
            sample_documents: Fixture with sample documents.

        Returns:
            None
        """
        mock_llm = MagicMock()
        mock_llm.api_key = "test-api-key"
        mock_llm_helper.return_value = mock_llm

        mock_reranker = MagicMock()
        mock_reranker_helper.return_value = mock_reranker

        mock_embedder = MagicMock()
        mock_embedder_helper.return_value = mock_embedder
        mock_embed_query.return_value = [0.1] * 384

        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents[:2]
        mock_db.return_value = mock_db_inst

        mock_compressor = MagicMock()
        mock_compressor.compress.return_value = sample_documents[:2]
        mock_compressor_class.return_value = mock_compressor

        mock_router = MagicMock()
        mock_router.route.side_effect = [
            {"action": "search", "reasoning": "Search with filters"},
            {"action": "generate", "reasoning": "Generate from filtered results"},
        ]
        mock_router_class.return_value = mock_router

        with patch(
            "vectordb.langchain.agentic_rag.search.chroma.RAGHelper.generate"
        ) as mock_generate:
            mock_generate.return_value = "Answer from filtered documents"

            config = {
                "dataloader": {"type": "arc", "limit": 10},
                "embeddings": {"model": "test-model", "device": "cpu"},
                "chroma": {
                    "path": "./test_chroma_data",
                    "collection_name": "test_agentic_rag",
                },
                "agentic": {
                    "max_iterations": 3,
                    "compression_mode": "reranking",
                    "router_model": "llama-3.3-70b-versatile",
                },
                "rag": {"enabled": True},
            }

            pipeline = ChromaAgenticRAGPipeline(config)
            filters = {"source": "wiki"}
            result = pipeline.run("What is Python?", filters=filters)

            mock_db_inst.query.assert_called_once_with(
                query_embedding=[0.1] * 384,
                top_k=10,
                filters=filters,
                collection_name="test_agentic_rag",
            )

            assert "final_answer" in result
            assert result["final_answer"] == "Answer from filtered documents"
            assert len(result["documents"]) == 2
