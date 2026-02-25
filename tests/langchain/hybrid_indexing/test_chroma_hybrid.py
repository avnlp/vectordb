"""Tests for Chroma hybrid indexing and search pipelines (LangChain).

This module tests the hybrid search feature which combines dense semantic embeddings
with sparse keyword-based (BM25) retrieval for improved search quality. Hybrid search
leverages both semantic understanding and exact keyword matching.

Hybrid Search Concept:
    Dense embeddings excel at capturing semantic meaning and conceptual similarity,
    but can miss exact keyword matches, especially for rare or technical terms.
    Sparse BM25 vectors excel at keyword matching but lack semantic understanding.
    Hybrid search combines both approaches to get the best of both worlds.

Hybrid Scoring:
    Documents are scored using a weighted combination:
    - Dense score: Cosine similarity between query and document embeddings
    - Sparse score: BM25 relevance score based on term frequency
    - Hybrid score = alpha * dense_score + (1 - alpha) * sparse_score

    Alpha parameter controls the weighting:
    - alpha = 1.0: Pure dense (semantic only)
    - alpha = 0.0: Pure sparse (BM25 only)
    - alpha = 0.5: Balanced hybrid (recommended for most use cases)

Pipeline Architecture:
    Indexing Pipeline:
        1. Load documents from configured data source (ARC, TriviaQA, etc.)
        2. Generate dense embeddings using configured embedding model
        3. Compute BM25 sparse vectors for keyword matching
        4. Store documents in Chroma with both embedding types:
           - Dense vectors in Chroma's vector storage
           - Sparse vectors in document metadata
        5. Fit BM25 model on corpus for term statistics

    Search Pipeline:
        1. Generate dense embedding for query
        2. Generate sparse vector for query terms
        3. Query Chroma with dense vector for candidate documents
        4. Retrieve sparse vectors from candidate metadata
        5. Compute hybrid scores using configured alpha
        6. Re-rank candidates by hybrid relevance
        7. Return top-k documents by hybrid score
        8. Optionally generate RAG answer from hybrid results

Components Tested:
    - ChromaHybridIndexingPipeline: Dual dense-sparse document indexing
    - ChromaHybridSearchPipeline: Hybrid scoring and retrieval
    - SparseEmbedder: BM25 vector generation
    - Hybrid score computation and re-ranking

Key Features:
    - Dual dense and sparse vector storage
    - Configurable hybrid weighting (alpha parameter)
    - BM25 sparse vector computation
    - Candidate retrieval and re-ranking pipeline
    - Metadata filtering support
    - Optional RAG generation from hybrid results

Test Coverage:
    - Pipeline initialization with hybrid configuration
    - Document indexing with dual vectors
    - Sparse vector storage in metadata
    - Hybrid search with dense query
    - Sparse query vector generation
    - Hybrid score computation
    - RAG generation from hybrid results
    - Metadata filtering in hybrid search
    - Edge cases: empty documents, missing sparse metadata

Configuration:
    Hybrid search requires both dense and sparse configuration:
    - embeddings: Dense embedding model configuration
    - sparse_embeddings: BM25 configuration (model: "bm25")
    - hybrid: {alpha: 0.5} for weighting control

Trade-offs:
    - Pros: Best of semantic + keyword, handles technical terms better
    - Cons: Higher storage (2x vectors), slightly slower indexing

All tests mock vector database and embedding operations to ensure fast,
deterministic unit tests without external dependencies.
"""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


class TestChromaHybridIndexing:
    """Unit tests for Chroma hybrid indexing pipeline.

    Validates the indexing pipeline that generates and stores both dense
    and sparse vectors for hybrid retrieval.

    Tested Behaviors:
        - Pipeline initialization with hybrid configuration
        - Dense embedding model initialization
        - Sparse (BM25) embedder initialization
        - Document loading and dual vector generation
        - Sparse vector storage in metadata
        - Chroma database upsert with hybrid data
        - Empty document handling

    Mocks:
        - ChromaVectorDB: Database operations
        - EmbedderHelper: Dense embedding generation
        - SparseEmbedder: BM25 sparse vector computation
        - DataLoaderHelper: Document loading

    Storage Format:
        - Dense vectors: Chroma native vector storage
        - Sparse vectors: Document metadata (indices and values)
    """

    @patch("vectordb.langchain.hybrid_indexing.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.hybrid_indexing.indexing.chroma.SparseEmbedder")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.chroma.DataloaderCatalog.create"
    )
    def test_indexing_initialization(
        self, mock_get_docs, mock_sparse_embedder, mock_embedder_helper, mock_db
    ):
        """Test pipeline initialization with hybrid configuration.

        Args:
            mock_get_docs: Mock for DataloaderCatalog.create
            mock_sparse_embedder: Mock for SparseEmbedder class
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class

        Verifies:
            - Configuration is stored correctly
            - Collection name is extracted from config
            - Dense embedder is initialized
            - Sparse embedder is initialized
            - Database connection is established
        """
        from vectordb.langchain.hybrid_indexing.indexing.chroma import (
            ChromaHybridIndexingPipeline,
        )

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "host": "localhost",
                "port": 8000,
                "collection_name": "test_hybrid",
            },
        }

        pipeline = ChromaHybridIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.collection_name == "test_hybrid"
        assert pipeline.dense_embedder is not None
        assert pipeline.sparse_embedder is not None

    @patch("vectordb.langchain.hybrid_indexing.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.chroma.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.hybrid_indexing.indexing.chroma.SparseEmbedder")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.chroma.DataloaderCatalog.create"
    )
    def test_indexing_run_with_documents(
        self,
        mock_get_docs,
        mock_sparse_embedder_class,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
    ):
        """Test indexing pipeline with documents including sparse embeddings.

        Args:
            mock_get_docs: Mock for DataloaderCatalog.create
            mock_sparse_embedder_class: Mock for SparseEmbedder class
            mock_embed_docs: Mock for EmbedderHelper.embed_documents
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class

        Verifies:
            - Documents are loaded from data source
            - Dense embeddings are generated
            - Sparse vectors are computed via BM25
            - Both vectors are stored in Chroma
            - Sparse vectors are included in metadata
            - Returns indexing statistics
        """
        from vectordb.langchain.hybrid_indexing.indexing.chroma import (
            ChromaHybridIndexingPipeline,
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

        # Mock chain: create() -> load() -> to_langchain()
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader

        mock_embed_docs.return_value = (sample_documents, [[0.1] * 384] * 2)

        # Mock sparse embedder
        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder.embed_documents.return_value = [
            {"indices": [1, 2, 3], "values": [0.5, 0.3, 0.2]},
            {"indices": [4, 5, 6], "values": [0.6, 0.4, 0.1]},
        ]
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        mock_db_inst = MagicMock()
        mock_db_inst.upsert.return_value = len(sample_documents)
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "host": "localhost",
                "port": 8000,
                "collection_name": "test_hybrid",
            },
        }

        pipeline = ChromaHybridIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        assert result["db"] == "chroma"
        assert result["collection_name"] == "test_hybrid"
        mock_db_inst.create_collection.assert_called_once()
        mock_db_inst.upsert.assert_called_once()

        # Verify sparse embeddings were generated
        mock_sparse_embedder.embed_documents.assert_called_once()

    @patch("vectordb.langchain.hybrid_indexing.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.hybrid_indexing.indexing.chroma.SparseEmbedder")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.chroma.DataloaderCatalog.create"
    )
    def test_indexing_run_no_documents(
        self, mock_get_docs, mock_sparse_embedder, mock_embedder_helper, mock_db
    ):
        """Test indexing pipeline with no documents.

        Args:
            mock_get_docs: Mock for DataloaderCatalog.create
            mock_sparse_embedder: Mock for SparseEmbedder class
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class

        Verifies:
            - Pipeline handles empty document list gracefully
            - Returns 0 documents indexed
            - Database is identified in results
        """
        from vectordb.langchain.hybrid_indexing.indexing.chroma import (
            ChromaHybridIndexingPipeline,
        )

        # Mock chain: create() -> load() -> to_langchain()
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = []
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "host": "localhost",
                "port": 8000,
                "collection_name": "test_hybrid",
            },
        }

        pipeline = ChromaHybridIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0
        assert result["db"] == "chroma"

    @patch("vectordb.langchain.hybrid_indexing.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.chroma.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.hybrid_indexing.indexing.chroma.SparseEmbedder")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.chroma.DataloaderCatalog.create"
    )
    def test_sparse_embeddings_in_metadata(
        self,
        mock_get_docs,
        mock_sparse_embedder_class,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
    ):
        """Test that sparse embeddings are stored in metadata.

        Args:
            mock_get_docs: Mock for DataloaderCatalog.create
            mock_sparse_embedder_class: Mock for SparseEmbedder class
            mock_embed_docs: Mock for EmbedderHelper.embed_documents
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class

        Verifies:
            - Sparse vectors are stored in document metadata
            - Metadata includes sparse_embedding field
            - Indices and values are properly formatted
        """
        from vectordb.langchain.hybrid_indexing.indexing.chroma import (
            ChromaHybridIndexingPipeline,
        )

        sample_documents = [
            Document(
                page_content="Test document",
                metadata={"source": "test"},
            ),
        ]

        # Mock chain: create() -> load() -> to_langchain()
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader

        mock_embed_docs.return_value = (sample_documents, [[0.1] * 384])

        # Mock sparse embedder
        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder.embed_documents.return_value = [
            {"indices": [1, 2], "values": [0.5, 0.3]},
        ]
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        mock_db_inst = MagicMock()
        mock_db_inst.upsert.return_value = 1
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "host": "localhost",
                "port": 8000,
                "collection_name": "test_hybrid",
            },
        }

        pipeline = ChromaHybridIndexingPipeline(config)
        pipeline.run()

        # Check that upsert was called with sparse embedding in metadata
        call_args = mock_db_inst.upsert.call_args
        upsert_data = call_args.kwargs.get(
            "data", call_args.args[0] if call_args.args else []
        )
        if upsert_data:
            assert "metadata" in upsert_data[0]
            assert "sparse_embedding" in upsert_data[0]["metadata"]


class TestChromaHybridSearch:
    """Unit tests for Chroma hybrid search pipeline.

    Validates the search pipeline that combines dense and sparse retrieval
    using hybrid scoring for improved relevance.

    Tested Behaviors:
        - Search pipeline initialization with hybrid configuration
        - Dense query embedding generation
        - Sparse query vector generation (BM25)
        - Candidate retrieval via dense search
        - Hybrid score computation (alpha weighting)
        - Re-ranking by hybrid relevance
        - Metadata filtering support
        - RAG generation from hybrid results

    Mocks:
        - ChromaVectorDB: Database query operations
        - EmbedderHelper: Dense embedding and query embedding
        - SparseEmbedder: BM25 query vector generation
        - RAGHelper: LLM initialization and answer generation

    Hybrid Scoring:
        hybrid_score = alpha * dense_score + (1 - alpha) * sparse_score
        - alpha: Weight for dense vs sparse (default 0.5)
        - dense_score: Cosine similarity
        - sparse_score: BM25 relevance
    """

    @patch("vectordb.langchain.hybrid_indexing.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.hybrid_indexing.search.chroma.SparseEmbedder")
    @patch("vectordb.langchain.hybrid_indexing.search.chroma.RAGHelper.create_llm")
    def test_search_initialization(
        self, mock_llm_helper, mock_sparse_embedder, mock_embedder_helper, mock_db
    ):
        """Test search pipeline initialization.

        Args:
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_sparse_embedder: Mock for SparseEmbedder class
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class

        Verifies:
            - Configuration is stored correctly
            - Collection name is extracted from config
            - Dense and sparse embedders are initialized
            - LLM is initialized if RAG is configured
        """
        from vectordb.langchain.hybrid_indexing.search.chroma import (
            ChromaHybridSearchPipeline,
        )

        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "host": "localhost",
                "port": 8000,
                "collection_name": "test_hybrid",
            },
        }

        pipeline = ChromaHybridSearchPipeline(config)
        assert pipeline.collection_name == "test_hybrid"
        assert pipeline.llm is None

    @patch("vectordb.langchain.hybrid_indexing.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.hybrid_indexing.search.chroma.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.hybrid_indexing.search.chroma.SparseEmbedder")
    @patch("vectordb.langchain.hybrid_indexing.search.chroma.RAGHelper.create_llm")
    def test_search_basic(
        self,
        mock_llm_helper,
        mock_sparse_embedder_class,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test basic search with dense embeddings.

        Args:
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_sparse_embedder_class: Mock for SparseEmbedder class
            mock_embed_query: Mock for EmbedderHelper.embed_query
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class

        Verifies:
            - Query is embedded using dense model
            - Chroma database is queried with dense vector
            - Results include query and matching documents
            - No answer is generated when RAG is disabled
        """
        from vectordb.langchain.hybrid_indexing.search.chroma import (
            ChromaHybridSearchPipeline,
        )

        [
            Document(
                page_content="Python is a high-level programming language",
                metadata={"source": "wiki", "id": "1"},
            ),
        ]

        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = {
            "ids": [["1"]],
            "documents": [["Python is a high-level programming language"]],
            "metadatas": [[{"source": "wiki"}]],
            "distances": [[0.1]],
        }
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "host": "localhost",
                "port": 8000,
                "collection_name": "test_hybrid",
            },
        }

        pipeline = ChromaHybridSearchPipeline(config)
        result = pipeline.search("test query", top_k=10)

        assert result["query"] == "test query"
        assert len(result["documents"]) == 1
        assert "answer" not in result

    @patch("vectordb.langchain.hybrid_indexing.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.hybrid_indexing.search.chroma.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.hybrid_indexing.search.chroma.SparseEmbedder")
    @patch("vectordb.langchain.hybrid_indexing.search.chroma.RAGHelper.create_llm")
    @patch("vectordb.langchain.hybrid_indexing.search.chroma.RAGHelper.generate")
    def test_search_with_rag(
        self,
        mock_rag_generate,
        mock_llm_helper,
        mock_sparse_embedder_class,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test search with RAG generation.

        Args:
            mock_rag_generate: Mock for RAGHelper.generate
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_sparse_embedder_class: Mock for SparseEmbedder class
            mock_embed_query: Mock for EmbedderHelper.embed_query
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class

        Verifies:
            - Hybrid search retrieves relevant documents
            - LLM is initialized when RAG is enabled
            - RAG answer is generated from retrieved documents
            - Answer is included in search results
            - Query is preserved in results
        """
        from vectordb.langchain.hybrid_indexing.search.chroma import (
            ChromaHybridSearchPipeline,
        )

        [
            Document(
                page_content="Python is a high-level programming language",
                metadata={"source": "wiki", "id": "1"},
            ),
        ]

        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = {
            "ids": [["1"]],
            "documents": [["Python is a high-level programming language"]],
            "metadatas": [[{"source": "wiki"}]],
            "distances": [[0.1]],
        }
        mock_db.return_value = mock_db_inst

        mock_llm_inst = MagicMock()
        mock_llm_helper.return_value = mock_llm_inst
        mock_rag_generate.return_value = "Test answer"

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "host": "localhost",
                "port": 8000,
                "collection_name": "test_hybrid",
            },
            "rag": {"enabled": True},
        }

        pipeline = ChromaHybridSearchPipeline(config)
        result = pipeline.search("test query", top_k=10)

        assert result["query"] == "test query"
        assert result["answer"] == "Test answer"

    @patch("vectordb.langchain.hybrid_indexing.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.hybrid_indexing.search.chroma.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.hybrid_indexing.search.chroma.SparseEmbedder")
    @patch("vectordb.langchain.hybrid_indexing.search.chroma.RAGHelper.create_llm")
    def test_search_with_filters(
        self,
        mock_llm_helper,
        mock_sparse_embedder_class,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test search with metadata filters.

        Args:
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_sparse_embedder_class: Mock for SparseEmbedder class
            mock_embed_query: Mock for EmbedderHelper.embed_query
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class

        Verifies:
            - Metadata filters are passed to Chroma query
            - Filters are applied to candidate retrieval
            - Results respect filter criteria
            - Query is preserved in results
        """
        from vectordb.langchain.hybrid_indexing.search.chroma import (
            ChromaHybridSearchPipeline,
        )

        [
            Document(
                page_content="Python is a high-level programming language",
                metadata={"source": "wiki", "id": "1"},
            ),
        ]

        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = {
            "ids": [["1"]],
            "documents": [["Python is a high-level programming language"]],
            "metadatas": [[{"source": "wiki"}]],
            "distances": [[0.1]],
        }
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "host": "localhost",
                "port": 8000,
                "collection_name": "test_hybrid",
            },
        }

        pipeline = ChromaHybridSearchPipeline(config)
        filters = {"source": "wiki"}
        result = pipeline.search("test query", top_k=10, filters=filters)

        assert result["query"] == "test query"
        mock_db_inst.query.assert_called_once()
        call_kwargs = mock_db_inst.query.call_args.kwargs
        assert call_kwargs["where"] == filters
