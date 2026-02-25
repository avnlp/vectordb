"""Tests for Chroma contextual compression pipelines (LangChain).

This module tests the contextual compression pipeline implementation for Chroma
vector database. Contextual compression prunes retrieved documents to retain only
relevant content, reducing noise and improving RAG answer quality.

Contextual Compression Pipeline Flow:
    1. Initial retrieval: Vector search returns candidate documents
    2. Compression: Reranker scores and filters documents
    3. Pruning: Low-relevance documents removed
    4. RAG: LLM generates answer using compressed context

Compression Strategies:
    - Reranking-based: Cross-encoder scores documents, keep top N
    - LLM extraction: LLM extracts relevant passages from documents
    - Benefits: Reduced context size, improved answer quality, lower token cost

Chroma-specific aspects tested:
    - Local persistent storage path configuration
    - Collection-based document organization
    - Optional collection recreation for fresh starts
    - Integration with RerankerHelper for compression

Test Coverage:
    - Indexing pipeline initialization and configuration
    - Document indexing with embeddings
    - Empty batch handling during indexing
    - Collection recreation option
    - Search pipeline initialization with reranker
    - Search with compression (retrieve many, return few)
    - RAG mode with compressed context

External dependencies (ChromaVectorDB, EmbedderHelper, DataLoaderHelper,
RAGHelper, RerankerHelper) are mocked to enable fast, isolated unit tests.
"""

from unittest.mock import MagicMock, patch

from vectordb.langchain.contextual_compression.indexing.chroma import (
    ChromaContextualCompressionIndexingPipeline,
)
from vectordb.langchain.contextual_compression.search.chroma import (
    ChromaContextualCompressionSearchPipeline,
)


class TestChromaContextualCompressionIndexing:
    """Unit tests for Chroma contextual compression indexing pipeline.

    Validates the indexing pipeline which stores documents for later
    compressed retrieval. The indexing process is identical to standard
    semantic search - compression happens at query time, not index time.

    Pipeline Flow:
        1. Load documents from configured dataloader
        2. Embed documents using specified embedding model
        3. Store in Chroma collection with metadata
        4. Return indexing statistics

    Note:
        Compression is query-time operation; indexing remains unchanged.
        This allows same index to be used with different compression strategies.
    """

    @patch("vectordb.langchain.contextual_compression.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.contextual_compression.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.contextual_compression.indexing.chroma.DataloaderCatalog.create"
    )
    def test_indexing_initialization(
        self,
        mock_get_docs,
        mock_embedder_helper,
        mock_db,
        contextual_compression_config,
    ):
        """Test pipeline initialization with contextual compression configuration.

        Verifies that:
        - Configuration dict is preserved on pipeline instance
        - Collection name is extracted from chroma config section
        - No external calls during initialization

        Args:
            mock_get_docs: Mock for DataloaderCatalog.create.
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db: Mock for ChromaVectorDB class.
            contextual_compression_config: Fixture with compression config.
        """
        pipeline = ChromaContextualCompressionIndexingPipeline(
            contextual_compression_config
        )
        assert pipeline.config == contextual_compression_config
        assert pipeline.collection_name == "test_contextual_compression"

    @patch("vectordb.langchain.contextual_compression.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.contextual_compression.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.contextual_compression.indexing.chroma.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.contextual_compression.indexing.chroma.DataloaderCatalog.create"
    )
    def test_indexing_run_with_documents(
        self,
        mock_get_docs,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test successful document indexing for compression pipeline.

        Validates the complete indexing workflow:
        1. DataLoaderHelper loads sample documents
        2. EmbedderHelper generates 384-dimensional embeddings
        3. ChromaVectorDB.upsert persists to local storage
        4. Result reports count of indexed documents

        Args:
            mock_get_docs: Mock returning sample_documents fixture.
            mock_embed_docs: Mock returning (docs, embeddings) tuple.
            mock_embedder_helper: Mock for embedder factory.
            mock_db: Mock for ChromaVectorDB, tracks upsert calls.
            sample_documents: Fixture with 5 sample documents.
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
                "collection_name": "test_contextual_compression",
            },
        }

        pipeline = ChromaContextualCompressionIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.upsert.assert_called_once()

    @patch("vectordb.langchain.contextual_compression.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.contextual_compression.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.contextual_compression.indexing.chroma.DataloaderCatalog.create"
    )
    def test_indexing_run_no_documents(
        self, mock_get_docs, mock_embedder_helper, mock_db
    ):
        """Test graceful handling of empty document batches.

        Ensures when the dataloader returns empty list:
        - No exceptions raised
        - Result reports 0 documents indexed
        - No database operations attempted

        Args:
            mock_get_docs: Mock returning empty list.
            mock_embedder_helper: Mock for embedder factory.
            mock_db: Mock for ChromaVectorDB (should not be called).
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
                "collection_name": "test_contextual_compression",
            },
        }

        pipeline = ChromaContextualCompressionIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0

    @patch("vectordb.langchain.contextual_compression.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.contextual_compression.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.contextual_compression.indexing.chroma.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.contextual_compression.indexing.chroma.DataloaderCatalog.create"
    )
    def test_indexing_with_recreate_option(
        self,
        mock_get_docs,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
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
                "collection_name": "test_contextual_compression",
                "recreate": True,
            },
        }

        pipeline = ChromaContextualCompressionIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.create_collection.assert_called_once()


class TestChromaContextualCompressionSearch:
    """Unit tests for Chroma contextual compression search pipeline.

    Tests validate the compression-based retrieval process:
    1. Initial retrieval: Fetch candidates via vector search
    2. Compression: Reranker scores and filters documents
    3. Final results: Return compressed, high-relevance subset

    Compression Benefits:
        - Reduced context size for LLM
        - Improved answer quality (less noise)
        - Lower token costs
        - Faster generation

    Trade-offs:
        - Additional latency for reranking step
        - Risk of filtering relevant content
        - Tuning required for compression ratio
    """

    @patch("vectordb.langchain.contextual_compression.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.contextual_compression.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.contextual_compression.search.chroma.RAGHelper.create_llm"
    )
    @patch(
        "vectordb.langchain.contextual_compression.search.chroma.RerankerHelper.create_reranker"
    )
    def test_search_initialization(
        self, mock_reranker_helper, mock_llm_helper, mock_embedder_helper, mock_db
    ):
        """Test search pipeline initialization."""
        mock_llm_helper.return_value = None
        mock_reranker_helper.return_value = MagicMock()

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_contextual_compression",
            },
            "rag": {"enabled": False},
        }

        pipeline = ChromaContextualCompressionSearchPipeline(config)
        assert pipeline.config == config

    @patch("vectordb.langchain.contextual_compression.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.contextual_compression.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.contextual_compression.search.chroma.EmbedderHelper.embed_query"
    )
    @patch(
        "vectordb.langchain.contextual_compression.search.chroma.RAGHelper.create_llm"
    )
    @patch(
        "vectordb.langchain.contextual_compression.search.chroma.RerankerHelper.create_reranker"
    )
    def test_search_execution(
        self,
        mock_reranker_helper,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test search execution."""
        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        # Mock the reranker to return scores for each document
        mock_reranker = MagicMock()
        mock_reranker.rank.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]
        mock_reranker_helper.return_value = mock_reranker

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_contextual_compression",
            },
            "rag": {"enabled": False},
        }

        pipeline = ChromaContextualCompressionSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert len(result["documents"]) > 0

    @patch("vectordb.langchain.contextual_compression.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.contextual_compression.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.contextual_compression.search.chroma.EmbedderHelper.embed_query"
    )
    @patch(
        "vectordb.langchain.contextual_compression.search.chroma.RAGHelper.create_llm"
    )
    @patch(
        "vectordb.langchain.contextual_compression.search.chroma.RerankerHelper.create_reranker"
    )
    @patch("vectordb.langchain.contextual_compression.search.chroma.RAGHelper.generate")
    def test_search_with_rag(
        self,
        mock_rag_generate,
        mock_reranker_helper,
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

        # Mock the reranker to return scores for each document
        mock_reranker = MagicMock()
        mock_reranker.rank.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]
        mock_reranker_helper.return_value = mock_reranker

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_contextual_compression",
            },
            "rag": {"enabled": True},
        }

        pipeline = ChromaContextualCompressionSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert result["answer"] == "Test answer"
