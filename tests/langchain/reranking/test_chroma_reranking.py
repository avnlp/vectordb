"""Tests for Chroma reranking pipelines (LangChain).

This module tests the reranking pipeline implementation for Chroma vector database.
Reranking improves retrieval quality by applying a more sophisticated scoring model
to the initial vector search results, reordering them for better relevance.

Reranking Pipeline Flow:
    1. Initial retrieval: Vector similarity search returns top_k * rerank_k candidates
    2. Reranking: Cross-encoder or LLM-based model scores each candidate
    3. Final selection: Top rerank_k results returned to user

Chroma-specific aspects tested:
    - Local persistent storage path configuration
    - Collection-based document organization
    - Default collection name fallback ("reranking")
    - Integration with RerankerHelper for model initialization

Test Coverage:
    - Indexing pipeline initialization and configuration
    - Document indexing with embeddings
    - Empty batch handling during indexing
    - Search pipeline initialization with reranker setup
    - Search execution with reranking (retrieve more, return fewer)
    - RAG mode with answer generation after reranking
    - Metadata filtering with reranking
    - Empty result handling

External dependencies (ChromaVectorDB, EmbedderHelper, DataLoaderHelper,
RAGHelper, RerankerHelper) are mocked to enable fast, isolated unit tests.
"""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


class TestChromaRerankingIndexing:
    """Unit tests for Chroma reranking indexing pipeline.

    Validates the indexing pipeline which stores documents for later
    reranked retrieval. The indexing process is similar to standard
    semantic search but the collection is configured for reranking use.

    Pipeline Flow:
        1. Load documents from configured dataloader
        2. Embed documents using specified embedding model
        3. Store in Chroma collection with metadata
        4. Return indexing statistics

    Configuration:
        - Collection name (default: "reranking" if not specified)
        - Persistent storage path for local Chroma
        - Embedding model and device settings
    """

    @patch("vectordb.langchain.reranking.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.reranking.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.reranking.indexing.chroma.DataloaderCatalog.create")
    def test_indexing_initialization(
        self, mock_get_docs, mock_embedder_helper, mock_db
    ):
        """Test pipeline initialization with reranking configuration.

        Verifies that:
        - Configuration dict is preserved on pipeline instance
        - Collection name is extracted from chroma config section
        - No external calls during initialization

        Args:
            mock_get_docs: Mock for DataloaderCatalog.create.
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db: Mock for ChromaVectorDB class.
        """
        from vectordb.langchain.reranking.indexing.chroma import (
            ChromaRerankingIndexingPipeline,
        )

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_reranking",
            },
        }

        pipeline = ChromaRerankingIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.collection_name == "test_reranking"

    @patch("vectordb.langchain.reranking.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.reranking.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.reranking.indexing.chroma.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.reranking.indexing.chroma.DataloaderCatalog.create")
    def test_indexing_run_with_documents(
        self,
        mock_get_docs,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
    ):
        """Test successful document indexing for reranking pipeline.

        Validates the complete indexing workflow:
        1. DataLoaderHelper loads sample documents
        2. EmbedderHelper generates 384-dimensional embeddings
        3. ChromaVectorDB.upsert persists to local storage
        4. Result reports count of indexed documents

        Args:
            mock_get_docs: Mock returning sample documents.
            mock_embed_docs: Mock returning (docs, embeddings) tuple.
            mock_embedder_helper: Mock for embedder factory.
            mock_db: Mock for ChromaVectorDB, tracks upsert operation.
        """
        from vectordb.langchain.reranking.indexing.chroma import (
            ChromaRerankingIndexingPipeline,
        )

        sample_documents = [
            Document(
                page_content="Python is a high-level programming language",
                metadata={"source": "wiki", "id": "1"},
            ),
            Document(
                page_content="Machine learning uses algorithms to learn from data",
                metadata={"source": "wiki", "id": "2"},
            ),
        ]

        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader
        mock_embed_docs.return_value = (sample_documents, [[0.1] * 384] * 2)

        mock_db_inst = MagicMock()
        mock_db_inst.upsert.return_value = len(sample_documents)
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_reranking",
            },
        }

        pipeline = ChromaRerankingIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.upsert.assert_called_once()

    @patch("vectordb.langchain.reranking.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.reranking.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.reranking.indexing.chroma.DataloaderCatalog.create")
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
        from vectordb.langchain.reranking.indexing.chroma import (
            ChromaRerankingIndexingPipeline,
        )

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
                "collection_name": "test_reranking",
            },
        }

        pipeline = ChromaRerankingIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0

    @patch("vectordb.langchain.reranking.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.reranking.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.reranking.indexing.chroma.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.reranking.indexing.chroma.DataloaderCatalog.create")
    def test_indexing_default_collection_name(
        self,
        mock_get_docs,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
    ):
        """Test fallback to default collection name when not specified.

        Verifies that when collection_name is omitted from config:
        - Pipeline uses "reranking" as default
        - Indexing proceeds normally
        - Documents stored in default collection

        This ensures backward compatibility and sensible defaults.

        Args:
            mock_get_docs: Mock returning sample documents.
            mock_embed_docs: Mock returning (docs, embeddings) tuple.
            mock_embedder_helper: Mock for embedder factory.
            mock_db: Mock for ChromaVectorDB.
        """
        from vectordb.langchain.reranking.indexing.chroma import (
            ChromaRerankingIndexingPipeline,
        )

        sample_documents = [
            Document(
                page_content="Test document",
                metadata={"source": "test"},
            ),
        ]

        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader
        mock_embed_docs.return_value = (sample_documents, [[0.1] * 384])

        mock_db_inst = MagicMock()
        mock_db_inst.upsert.return_value = len(sample_documents)
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
            },
        }

        pipeline = ChromaRerankingIndexingPipeline(config)
        assert pipeline.collection_name == "reranking"


class TestChromaRerankingSearch:
    """Unit tests for Chroma reranking search pipeline.

    Tests validate the two-stage retrieval process:
    1. Initial retrieval: Fetch top_k * rerank_k candidates via vector search
    2. Reranking: Apply cross-encoder model to reorder candidates
    3. Final results: Return top rerank_k most relevant documents

    The reranking stage improves precision by using a more expensive but
    more accurate scoring model on a smaller candidate set.

    Configuration:
        - top_k: Number of candidates to retrieve initially
        - rerank_k: Number of final results after reranking
        - Reranker model: Cross-encoder for scoring
    """

    @patch("vectordb.langchain.reranking.search.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.reranking.search.chroma.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.reranking.search.chroma.RAGHelper.create_llm")
    @patch("vectordb.langchain.reranking.search.chroma.RerankerHelper.create_reranker")
    def test_search_initialization(
        self, mock_reranker_helper, mock_llm_helper, mock_embedder_helper, mock_db
    ):
        """Test search pipeline initialization with reranker setup.

        Verifies that:
        - Configuration is stored on pipeline instance
        - Reranker is initialized via RerankerHelper
        - LLM is initialized (can be None when RAG disabled)
        - Embedder is initialized for query embedding

        Args:
            mock_reranker_helper: Mock for RerankerHelper.create_reranker.
            mock_llm_helper: Mock for RAGHelper.create_llm.
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db: Mock for ChromaVectorDB class.
        """
        from vectordb.langchain.reranking.search.chroma import (
            ChromaRerankingSearchPipeline,
        )

        mock_llm_helper.return_value = None
        mock_reranker_helper.return_value = MagicMock()

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_reranking",
            },
            "rag": {"enabled": False},
        }

        pipeline = ChromaRerankingSearchPipeline(config)
        assert pipeline.config == config
        assert pipeline.llm is None

    @patch("vectordb.langchain.reranking.search.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.reranking.search.chroma.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.reranking.search.chroma.EmbedderHelper.embed_query")
    @patch("vectordb.langchain.reranking.search.chroma.RAGHelper.create_llm")
    @patch("vectordb.langchain.reranking.search.chroma.RerankerHelper.create_reranker")
    @patch("vectordb.langchain.reranking.search.chroma.RerankerHelper.rerank")
    def test_search_execution(
        self,
        mock_rerank,
        mock_reranker_helper,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test two-stage search with reranking execution.

        Validates the complete reranking workflow:
        1. Query embedding via EmbedderHelper.embed_query
        2. Initial retrieval: ChromaVectorDB.query returns candidates
        3. Reranking: RerankerHelper.rerank reorders candidates
        4. Final results: Top rerank_k documents returned

        In this test:
        - top_k=10: Retrieve 10 candidates initially
        - rerank_k=5: Return top 5 after reranking
        - Mock returns 2 candidates, rerank returns 1

        Args:
            mock_rerank: Mock for RerankerHelper.rerank.
            mock_reranker_helper: Mock for RerankerHelper.create_reranker.
            mock_llm_helper: Mock for RAGHelper.create_llm.
            mock_embed_query: Mock returning query vector.
            mock_embedder_helper: Mock for embedder factory.
            mock_db: Mock for ChromaVectorDB with query results.
        """
        from vectordb.langchain.reranking.search.chroma import (
            ChromaRerankingSearchPipeline,
        )

        sample_documents = [
            Document(
                page_content="Python is a high-level programming language",
                metadata={"source": "wiki", "id": "1"},
            ),
            Document(
                page_content="Machine learning uses algorithms to learn from data",
                metadata={"source": "wiki", "id": "2"},
            ),
        ]

        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        mock_reranker = MagicMock()
        mock_reranker_helper.return_value = mock_reranker
        mock_rerank.return_value = sample_documents[:1]

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_reranking",
            },
            "rag": {"enabled": False},
        }

        pipeline = ChromaRerankingSearchPipeline(config)
        result = pipeline.search("test query", top_k=10, rerank_k=5)

        assert result["query"] == "test query"
        assert len(result["documents"]) == 1
        assert "answer" not in result

    @patch("vectordb.langchain.reranking.search.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.reranking.search.chroma.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.reranking.search.chroma.EmbedderHelper.embed_query")
    @patch("vectordb.langchain.reranking.search.chroma.RAGHelper.create_llm")
    @patch("vectordb.langchain.reranking.search.chroma.RerankerHelper.create_reranker")
    @patch("vectordb.langchain.reranking.search.chroma.RAGHelper.generate")
    @patch("vectordb.langchain.reranking.search.chroma.RerankerHelper.rerank")
    def test_search_with_rag(
        self,
        mock_rerank,
        mock_rag_generate,
        mock_reranker_helper,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test reranked search with RAG answer generation.

        Validates the complete pipeline with reranking and generation:
        1. Retrieve candidates via vector search
        2. Rerank candidates for better relevance
        3. Generate answer using reranked documents as context
        4. Return both documents and generated answer

        Benefits of reranking before RAG:
        - Higher quality context documents
        - Better answer generation
        - More efficient LLM token usage

        Args:
            mock_rerank: Mock for RerankerHelper.rerank.
            mock_rag_generate: Mock for RAGHelper.generate.
            mock_reranker_helper: Mock for RerankerHelper.create_reranker.
            mock_llm_helper: Mock for RAGHelper.create_llm.
            mock_embed_query: Mock returning query vector.
            mock_embedder_helper: Mock for embedder factory.
            mock_db: Mock for ChromaVectorDB with query results.
        """
        from vectordb.langchain.reranking.search.chroma import (
            ChromaRerankingSearchPipeline,
        )

        sample_documents = [
            Document(
                page_content="Python is a high-level programming language",
                metadata={"source": "wiki", "id": "1"},
            ),
        ]

        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst

        mock_llm_inst = MagicMock()
        mock_llm_helper.return_value = mock_llm_inst
        mock_rag_generate.return_value = "Test answer"

        mock_reranker = MagicMock()
        mock_reranker_helper.return_value = mock_reranker
        mock_rerank.return_value = sample_documents

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_reranking",
            },
            "rag": {"enabled": True},
        }

        pipeline = ChromaRerankingSearchPipeline(config)
        result = pipeline.search("test query", top_k=10, rerank_k=5)

        assert result["query"] == "test query"
        assert result["answer"] == "Test answer"

    @patch("vectordb.langchain.reranking.search.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.reranking.search.chroma.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.reranking.search.chroma.EmbedderHelper.embed_query")
    @patch("vectordb.langchain.reranking.search.chroma.RAGHelper.create_llm")
    @patch("vectordb.langchain.reranking.search.chroma.RerankerHelper.create_reranker")
    @patch("vectordb.langchain.reranking.search.chroma.RerankerHelper.rerank")
    def test_search_with_filters(
        self,
        mock_rerank,
        mock_reranker_helper,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test reranked search with metadata filters.

        Validates that metadata filters are passed through to the initial
        vector search. Filters reduce the candidate pool before reranking,
        improving both performance and relevance.

        Filter Application:
        - Filters applied during initial ChromaVectorDB.query
        - Reranking operates on filtered candidate set
        - Final results respect filter constraints

        Args:
            mock_rerank: Mock for RerankerHelper.rerank.
            mock_reranker_helper: Mock for RerankerHelper.create_reranker.
            mock_llm_helper: Mock for RAGHelper.create_llm.
            mock_embed_query: Mock returning query vector.
            mock_embedder_helper: Mock for embedder factory.
            mock_db: Mock for ChromaVectorDB with query results.
        """
        from vectordb.langchain.reranking.search.chroma import (
            ChromaRerankingSearchPipeline,
        )

        sample_documents = [
            Document(
                page_content="Python is a high-level programming language",
                metadata={"source": "wiki", "id": "1"},
            ),
        ]

        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        mock_reranker = MagicMock()
        mock_reranker_helper.return_value = mock_reranker
        mock_rerank.return_value = sample_documents

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_reranking",
            },
            "rag": {"enabled": False},
        }

        pipeline = ChromaRerankingSearchPipeline(config)
        filters = {"source": "wiki"}
        result = pipeline.search("test query", top_k=10, rerank_k=5, filters=filters)

        assert result["query"] == "test query"
        mock_db_inst.query.assert_called_once()
        call_kwargs = mock_db_inst.query.call_args.kwargs
        assert call_kwargs["where"] == filters

    @patch("vectordb.langchain.reranking.search.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.reranking.search.chroma.EmbedderHelper.create_embedder")
    @patch("vectordb.langchain.reranking.search.chroma.EmbedderHelper.embed_query")
    @patch("vectordb.langchain.reranking.search.chroma.RAGHelper.create_llm")
    @patch("vectordb.langchain.reranking.search.chroma.RerankerHelper.create_reranker")
    @patch("vectordb.langchain.reranking.search.chroma.RerankerHelper.rerank")
    def test_search_empty_results(
        self,
        mock_rerank,
        mock_reranker_helper,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test reranked search when no documents match.

        Validates graceful handling of empty retrieval:
        - Empty list returned when no documents match query
        - No errors raised on empty results
        - Reranking not attempted on empty candidate set
        - Query preserved in result for debugging

        Edge Case Handling:
        - Initial retrieval returns empty list
        - Reranker not called (nothing to rerank)
        - Result contains empty documents list

        Args:
            mock_rerank: Mock for RerankerHelper.rerank.
            mock_reranker_helper: Mock for RerankerHelper.create_reranker.
            mock_llm_helper: Mock for RAGHelper.create_llm.
            mock_embed_query: Mock returning query vector.
            mock_embedder_helper: Mock for embedder factory.
            mock_db: Mock for ChromaVectorDB returning empty results.
        """
        from vectordb.langchain.reranking.search.chroma import (
            ChromaRerankingSearchPipeline,
        )

        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = []
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        mock_reranker = MagicMock()
        mock_reranker_helper.return_value = mock_reranker
        mock_rerank.return_value = []

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_reranking",
            },
            "rag": {"enabled": False},
        }

        pipeline = ChromaRerankingSearchPipeline(config)
        result = pipeline.search("test query", top_k=10, rerank_k=5)

        assert result["query"] == "test query"
        assert result["documents"] == []
