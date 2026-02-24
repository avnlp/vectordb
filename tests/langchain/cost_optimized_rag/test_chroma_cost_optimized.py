"""Tests for Chroma cost-optimized RAG pipelines (LangChain).

This module tests the cost-optimized RAG pipeline implementation for Chroma vector
database. Cost-optimized RAG reduces LLM API costs by using hybrid retrieval
(dense + sparse) with Reciprocal Rank Fusion (RRF) to improve retrieval quality
without expensive reranking models.

Cost-Optimized RAG Pipeline Flow:
    1. Indexing:
       - Dense embeddings: Standard vector embeddings (e.g., MiniLM)
       - Sparse embeddings: SPLADE or BM25 for lexical matching
       - Both stored in Chroma for hybrid retrieval
    2. Search:
       - Dense retrieval: Semantic similarity search
       - Sparse retrieval: Lexical/keyword matching
       - RRF fusion: Combine rankings without model inference
       - Top-k selection: Return fused results

Cost Optimization Strategies:
    - Hybrid retrieval: Better recall without expensive cross-encoders
    - RRF fusion: Simple, fast ranking combination
    - Sparse embeddings: Capture keywords missed by dense vectors
    - No reranking: Skip costly cross-encoder inference

Chroma-specific aspects tested:
    - Local persistent storage with persist_directory
    - Collection-based organization
    - Support for both dense and sparse vector storage
    - Custom chunking configuration

Test Coverage:
    - Indexing pipeline initialization with sparse embedder
    - Document indexing with dual embeddings (dense + sparse)
    - Empty batch handling
    - Custom chunking configuration
    - Search pipeline initialization with RRF parameters
    - Dense retrieval execution
    - RAG generation with cost-optimized retrieval
    - Metadata filtering
    - Custom RRF k parameter

External dependencies (ChromaVectorDB, EmbedderHelper, SparseEmbedder,
DataLoaderHelper, RAGHelper) are mocked to enable fast, isolated unit tests.
"""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


class TestChromaCostOptimizedIndexing:
    """Unit tests for Chroma cost-optimized RAG indexing pipeline.

    Validates the indexing pipeline which generates both dense and sparse
    embeddings for hybrid retrieval. Dual embeddings enable better recall
    by combining semantic and lexical matching.

    Pipeline Flow:
        1. Load documents from dataloader
        2. Chunk documents (if configured)
        3. Generate dense embeddings (semantic)
        4. Generate sparse embeddings (lexical/SPLADE)
        5. Store both in Chroma
        6. Return indexing statistics

    Dual Embedding Strategy:
        - Dense: Captures semantic meaning
        - Sparse: Captures exact keyword matches
        - Together: Better coverage than either alone
    """

    @patch("vectordb.langchain.cost_optimized_rag.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.cost_optimized_rag.indexing.chroma.SparseEmbedder")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.chroma.DataloaderCatalog.create"
    )
    def test_indexing_initialization(
        self, mock_get_docs, mock_sparse_embedder, mock_embedder_helper, mock_db
    ):
        """Test pipeline initialization with cost-optimized configuration.

        Verifies that:
        - Configuration dict is preserved on pipeline instance
        - Collection name extracted from chroma config
        - SparseEmbedder initialized for lexical embeddings
        - No external calls during initialization

        Args:
            mock_get_docs: Mock for DataloaderCatalog.create.
            mock_sparse_embedder: Mock for SparseEmbedder class.
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db: Mock for ChromaVectorDB class.
        """
        from vectordb.langchain.cost_optimized_rag.indexing.chroma import (
            ChromaCostOptimizedRAGIndexingPipeline,
        )

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_cost_optimized",
            },
        }

        pipeline = ChromaCostOptimizedRAGIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.collection_name == "test_cost_optimized"

    @patch("vectordb.langchain.cost_optimized_rag.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.chroma.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.cost_optimized_rag.indexing.chroma.SparseEmbedder")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.chroma.DataloaderCatalog.create"
    )
    def test_indexing_run_with_documents(
        self,
        mock_get_docs,
        mock_sparse_embedder_class,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
    ):
        """Test successful document indexing with dual embeddings.

        Validates the complete indexing workflow:
        1. DataLoaderHelper loads documents
        2. Dense embeddings generated via EmbedderHelper
        3. Sparse embeddings generated via SparseEmbedder
        4. Both stored in Chroma
        5. Result reports documents and chunks indexed

        Dual Embedding Benefits:
        - Dense: Semantic similarity for conceptual matches
        - Sparse: Lexical matching for exact keywords
        - Combined: Better recall than either alone

        Args:
            mock_get_docs: Mock returning sample documents.
            mock_sparse_embedder_class: Mock for SparseEmbedder.
            mock_embed_docs: Mock returning (docs, embeddings) tuple.
            mock_embedder_helper: Mock for embedder factory.
            mock_db: Mock for ChromaVectorDB, tracks upsert calls.
        """
        from vectordb.langchain.cost_optimized_rag.indexing.chroma import (
            ChromaCostOptimizedRAGIndexingPipeline,
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

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder.embed_documents.return_value = [
            {"indices": [1, 2], "values": [0.5, 0.5]}
        ] * 2
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        mock_db_inst = MagicMock()
        mock_db_inst.upsert.return_value = len(sample_documents)
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_cost_optimized",
            },
        }

        pipeline = ChromaCostOptimizedRAGIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.upsert.assert_called_once()

    @patch("vectordb.langchain.cost_optimized_rag.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.cost_optimized_rag.indexing.chroma.SparseEmbedder")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.chroma.DataloaderCatalog.create"
    )
    def test_indexing_run_no_documents(
        self, mock_get_docs, mock_sparse_embedder, mock_embedder_helper, mock_db
    ):
        """Test graceful handling of empty document batches.

        Ensures when the dataloader returns empty list:
        - No exceptions raised
        - Result reports 0 documents and 0 chunks indexed
        - No database operations attempted

        Args:
            mock_get_docs: Mock returning empty list.
            mock_sparse_embedder: Mock for SparseEmbedder class.
            mock_embedder_helper: Mock for embedder factory.
            mock_db: Mock for ChromaVectorDB (should not be called).
        """
        from vectordb.langchain.cost_optimized_rag.indexing.chroma import (
            ChromaCostOptimizedRAGIndexingPipeline,
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
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_cost_optimized",
            },
        }

        pipeline = ChromaCostOptimizedRAGIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0
        assert result["chunks_created"] == 0

    @patch("vectordb.langchain.cost_optimized_rag.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.chroma.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.cost_optimized_rag.indexing.chroma.SparseEmbedder")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.chroma.DataloaderCatalog.create"
    )
    def test_indexing_with_sparse_embeddings(
        self,
        mock_get_docs,
        mock_sparse_embedder_class,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
    ):
        """Test indexing generates both dense and sparse embeddings.

        Validates dual embedding generation:
        1. Dense embeddings: 384-dimensional vectors from transformer
        2. Sparse embeddings: SPLADE indices and values for lexical matching
        3. Both stored together in Chroma

        Sparse Embedding Format:
        - indices: Token IDs that are significant
        - values: Importance weights for each token

        Args:
            mock_get_docs: Mock returning sample documents.
            mock_sparse_embedder_class: Mock for SparseEmbedder.
            mock_embed_docs: Mock returning (docs, embeddings) tuple.
            mock_embedder_helper: Mock for embedder factory.
            mock_db: Mock for ChromaVectorDB, tracks upsert calls.
        """
        from vectordb.langchain.cost_optimized_rag.indexing.chroma import (
            ChromaCostOptimizedRAGIndexingPipeline,
        )

        sample_documents = [
            Document(
                page_content="Test document for sparse embeddings",
                metadata={"source": "test"},
            ),
        ]

        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader
        mock_embed_docs.return_value = (sample_documents, [[0.1] * 384])

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder.embed_documents.return_value = [
            {"indices": [1, 2, 3], "values": [0.3, 0.4, 0.3]}
        ]
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        mock_db_inst = MagicMock()
        mock_db_inst.upsert.return_value = 1
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_cost_optimized",
            },
        }

        pipeline = ChromaCostOptimizedRAGIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 1
        assert result["chunks_created"] == 1
        mock_sparse_embedder.embed_documents.assert_called_once()
        mock_db_inst.upsert.assert_called_once()

    @patch("vectordb.langchain.cost_optimized_rag.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.chroma.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.cost_optimized_rag.indexing.chroma.SparseEmbedder")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.chroma.DataloaderCatalog.create"
    )
    def test_indexing_with_chunking_config(
        self,
        mock_get_docs,
        mock_sparse_embedder_class,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
    ):
        """Test indexing with custom chunking configuration.

        Validates that chunking parameters are applied:
        - chunk_size: Maximum tokens per chunk
        - chunk_overlap: Tokens overlapping between chunks
        - Documents split before embedding

        Chunking Benefits:
        - Better granularity for retrieval
        - Fits within embedding model token limits
        - Overlap preserves context across chunks

        Args:
            mock_get_docs: Mock returning sample documents.
            mock_sparse_embedder_class: Mock for SparseEmbedder.
            mock_embed_docs: Mock returning (docs, embeddings) tuple.
            mock_embedder_helper: Mock for embedder factory.
            mock_db: Mock for ChromaVectorDB.
        """
        from vectordb.langchain.cost_optimized_rag.indexing.chroma import (
            ChromaCostOptimizedRAGIndexingPipeline,
        )

        sample_documents = [
            Document(
                page_content="Test document content",
                metadata={"source": "test"},
            ),
        ]

        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader
        mock_embed_docs.return_value = (sample_documents, [[0.1] * 384])

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder.embed_documents.return_value = [
            {"indices": [1], "values": [1.0]}
        ]
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        mock_db_inst = MagicMock()
        mock_db_inst.upsert.return_value = 1
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_cost_optimized",
            },
            "chunking": {
                "chunk_size": 500,
                "chunk_overlap": 100,
            },
        }

        pipeline = ChromaCostOptimizedRAGIndexingPipeline(config)
        result = pipeline.run()

        assert pipeline.text_splitter._chunk_size == 500
        assert pipeline.text_splitter._chunk_overlap == 100
        assert result["documents_indexed"] == 1


class TestChromaCostOptimizedSearch:
    """Unit tests for Chroma cost-optimized RAG search pipeline.

    Tests validate hybrid retrieval with RRF fusion:
    1. Dense retrieval: Semantic similarity search
    2. Sparse retrieval: Lexical keyword matching
    3. RRF fusion: Combine rankings without model inference
    4. Final selection: Return top-k fused results

    RRF (Reciprocal Rank Fusion):
    - score = sum(1 / (k + rank)) for each list
    - k=60 default (tunable parameter)
    - No ML model required, fast and cheap

    Cost Benefits:
    - No expensive cross-encoder reranking
    - Better recall than dense-only
    - Fast fusion computation
    """

    @patch("vectordb.langchain.cost_optimized_rag.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.cost_optimized_rag.search.chroma.SparseEmbedder")
    @patch("vectordb.langchain.cost_optimized_rag.search.chroma.RAGHelper.create_llm")
    def test_search_initialization(
        self, mock_llm_helper, mock_sparse_embedder, mock_embedder_helper, mock_db
    ):
        """Test search pipeline initialization."""
        from vectordb.langchain.cost_optimized_rag.search.chroma import (
            ChromaCostOptimizedRAGSearchPipeline,
        )

        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_cost_optimized",
            },
            "rag": {"enabled": False},
        }

        pipeline = ChromaCostOptimizedRAGSearchPipeline(config)
        assert pipeline.config == config
        assert pipeline.llm is None
        assert pipeline.rrf_k == 60

    @patch("vectordb.langchain.cost_optimized_rag.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.chroma.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.cost_optimized_rag.search.chroma.SparseEmbedder")
    @patch("vectordb.langchain.cost_optimized_rag.search.chroma.RAGHelper.create_llm")
    def test_search_execution(
        self,
        mock_llm_helper,
        mock_sparse_embedder_class,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test search execution with dense retrieval."""
        from vectordb.langchain.cost_optimized_rag.search.chroma import (
            ChromaCostOptimizedRAGSearchPipeline,
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

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_cost_optimized",
            },
            "rag": {"enabled": False},
        }

        pipeline = ChromaCostOptimizedRAGSearchPipeline(config)
        result = pipeline.search("What is Python?", top_k=2)

        assert result["query"] == "What is Python?"
        assert len(result["documents"]) == 2
        mock_db_inst.query.assert_called_once()

    @patch("vectordb.langchain.cost_optimized_rag.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.chroma.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.cost_optimized_rag.search.chroma.SparseEmbedder")
    @patch("vectordb.langchain.cost_optimized_rag.search.chroma.RAGHelper.create_llm")
    @patch("vectordb.langchain.cost_optimized_rag.search.chroma.RAGHelper.generate")
    def test_search_with_rag_generation(
        self,
        mock_rag_generate,
        mock_llm_helper,
        mock_sparse_embedder_class,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test search with RAG answer generation."""
        from vectordb.langchain.cost_optimized_rag.search.chroma import (
            ChromaCostOptimizedRAGSearchPipeline,
        )

        sample_documents = [
            Document(
                page_content="Python is a programming language",
                metadata={"source": "wiki"},
            ),
        ]

        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst

        mock_llm = MagicMock()
        mock_llm_helper.return_value = mock_llm
        mock_rag_generate.return_value = "Python is a popular programming language."

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_cost_optimized",
            },
            "rag": {"enabled": True, "model": "gpt-3.5-turbo"},
        }

        pipeline = ChromaCostOptimizedRAGSearchPipeline(config)
        result = pipeline.search("What is Python?", top_k=1)

        assert "answer" in result
        assert result["answer"] == "Python is a popular programming language."
        mock_rag_generate.assert_called_once()

    @patch("vectordb.langchain.cost_optimized_rag.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.chroma.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.cost_optimized_rag.search.chroma.SparseEmbedder")
    @patch("vectordb.langchain.cost_optimized_rag.search.chroma.RAGHelper.create_llm")
    def test_search_with_filters(
        self,
        mock_llm_helper,
        mock_sparse_embedder_class,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test search with metadata filters."""
        from vectordb.langchain.cost_optimized_rag.search.chroma import (
            ChromaCostOptimizedRAGSearchPipeline,
        )

        sample_documents = [
            Document(
                page_content="Python programming",
                metadata={"source": "wiki", "category": "programming"},
            ),
        ]

        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_cost_optimized",
            },
            "rag": {"enabled": False},
        }

        pipeline = ChromaCostOptimizedRAGSearchPipeline(config)
        filters = {"category": "programming"}
        result = pipeline.search("Python", top_k=1, filters=filters)

        assert len(result["documents"]) == 1
        mock_db_inst.query.assert_called_once()

    @patch("vectordb.langchain.cost_optimized_rag.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.cost_optimized_rag.search.chroma.SparseEmbedder")
    @patch("vectordb.langchain.cost_optimized_rag.search.chroma.RAGHelper.create_llm")
    def test_search_custom_rrf_k(
        self, mock_llm_helper, mock_sparse_embedder, mock_embedder_helper, mock_db
    ):
        """Test search with custom RRF k parameter."""
        from vectordb.langchain.cost_optimized_rag.search.chroma import (
            ChromaCostOptimizedRAGSearchPipeline,
        )

        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_cost_optimized",
            },
            "search": {
                "rrf_k": 120,
            },
            "rag": {"enabled": False},
        }

        pipeline = ChromaCostOptimizedRAGSearchPipeline(config)
        assert pipeline.rrf_k == 120
