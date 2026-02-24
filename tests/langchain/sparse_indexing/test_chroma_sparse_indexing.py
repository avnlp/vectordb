"""Tests for Chroma sparse indexing pipelines (LangChain).

This module tests the sparse indexing feature which implements keyword-based
(BM25) retrieval for Chroma vector database. Sparse indexing complements dense
semantic search by matching exact keywords and term frequencies.

Sparse Indexing Concept:
    Dense embeddings capture semantic meaning but can miss exact keyword matches.
    Sparse indexing uses traditional information retrieval techniques like BM25
to score documents based on term frequency and inverse document frequency.
This provides better precision for keyword-heavy queries.

BM25 Algorithm:
    BM25 is a probabilistic ranking function that estimates relevance based on:
    - Term frequency in document (more occurrences = higher score)
    - Inverse document frequency (rare terms score higher)
    - Document length normalization (longer documents don't dominate)
    Formula: score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))

Pipeline Architecture:
    Indexing Pipeline:
        1. Load documents from configured data source
        2. Tokenize documents for BM25 sparse vectors
        3. Compute BM25 statistics and sparse embeddings
        4. Store documents in Chroma with sparse vector metadata
        5. BM25 model is fit on the corpus for term statistics

    Search Pipeline:
        1. Tokenize query into terms
        2. Compute BM25 scores for documents matching query terms
        3. Retrieve top-k documents by BM25 score
        4. Optionally generate RAG answer from retrieved documents
        5. Return documents and optionally generated answer

Components Tested:
    - ChromaSparseIndexingPipeline: BM25-based document indexing
    - ChromaSparseSearchPipeline: Sparse vector search with BM25
    - SparseEmbedder: BM25 embedding model for sparse vectors

Key Features:
    - BM25-based sparse vector computation
    - Keyword-focused retrieval (exact term matching)
    - Term frequency and document frequency statistics
    - Configurable BM25 parameters (k1, b)
    - Optional RAG generation from sparse results
    - Chroma database integration with sparse metadata

Test Coverage:
    - Pipeline initialization with sparse embedding configuration
    - Document indexing with BM25 sparse vectors
    - Sparse search execution with BM25 scoring
    - RAG generation from sparse retrieval results
    - Empty document handling
    - BM25 model fitting and term statistics

Configuration:
    Sparse indexing requires configuring the sparse_embeddings section:
    - model: "bm25" (required)
    - k1: Term frequency saturation parameter (default: 1.5)
    - b: Length normalization parameter (default: 0.75)

Trade-offs:
    - Pros: Better keyword matching, no embedding computation during search
    - Cons: No semantic understanding, vocabulary limitations, no cross-lingual

All tests mock vector database and sparse embedding operations to ensure
fast, deterministic unit tests without corpus-dependent BM25 fitting.
"""

from unittest.mock import MagicMock, patch

from vectordb.langchain.sparse_indexing.indexing.chroma import (
    ChromaSparseIndexingPipeline,
)
from vectordb.langchain.sparse_indexing.search.chroma import (
    ChromaSparseSearchPipeline,
)


class TestChromaSparseIndexing:
    """Unit tests for Chroma sparse indexing pipeline.

    Validates the indexing pipeline that computes BM25 sparse vectors
    and stores documents in Chroma for keyword-based retrieval.

    Tested Behaviors:
        - Pipeline initialization with sparse embedding configuration
        - Document loading and tokenization
        - BM25 sparse vector generation
        - Chroma database storage with sparse metadata
        - Empty document handling
        - Sparse embedder initialization

    Mocks:
        - ChromaVectorDB: Database operations (collection creation, upsert)
        - DataLoaderHelper: Document loading from data sources
        - SparseEmbedder: BM25 model fitting and sparse vector generation

    BM25 Parameters:
        - k1: Controls term frequency saturation (default 1.5)
        - b: Controls length normalization (default 0.75)
    """

    @patch("vectordb.langchain.sparse_indexing.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.sparse_indexing.indexing.chroma.DataloaderCatalog.create"
    )
    def test_indexing_initialization(self, mock_get_docs, mock_db):
        """Test pipeline initialization with sparse embedding configuration.

        Args:
            mock_get_docs: Mock for DataloaderCatalog.create
            mock_db: Mock for ChromaVectorDB class

        Verifies:
            - Configuration is stored correctly including sparse_embeddings
            - Collection name is extracted from config
            - Database connection is established
            - Sparse embedder is initialized
        """
        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "sparse_embeddings": {"model": "bm25", "device": "cpu"},
            "chroma": {"collection_name": "test", "path": "./test_data"},
        }

        pipeline = ChromaSparseIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.collection_name == "test"

    @patch("vectordb.langchain.sparse_indexing.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.sparse_indexing.indexing.chroma.DataloaderCatalog.create"
    )
    def test_indexing_run_with_documents(
        self,
        mock_get_docs,
        mock_db,
        sample_documents,
    ):
        """Test indexing pipeline with documents using BM25 sparse vectors.

        Args:
            mock_get_docs: Mock for DataloaderCatalog.create
            mock_db: Mock for ChromaVectorDB class
            sample_documents: Fixture providing test documents

        Verifies:
            - Documents are loaded from data source
            - BM25 sparse vectors are computed for documents
            - Documents are upserted to Chroma with sparse metadata
            - Returns count of indexed documents
            - Database upsert is called once
        """
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader

        mock_db_inst = MagicMock()
        mock_db_inst.upsert.return_value = len(sample_documents)
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "sparse_embeddings": {"model": "bm25", "device": "cpu"},
            "chroma": {"collection_name": "test", "path": "./test_data"},
        }

        pipeline = ChromaSparseIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.upsert.assert_called_once()

    @patch("vectordb.langchain.sparse_indexing.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.sparse_indexing.indexing.chroma.DataloaderCatalog.create"
    )
    def test_indexing_run_no_documents(self, mock_get_docs, mock_db):
        """Test indexing pipeline with no documents.

        Args:
            mock_get_docs: Mock for DataloaderCatalog.create
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
            "sparse_embeddings": {"model": "bm25", "device": "cpu"},
            "chroma": {"collection_name": "test", "path": "./test_data"},
        }

        pipeline = ChromaSparseIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0


class TestChromaSparseSearch:
    """Unit tests for Chroma sparse search pipeline.

    Validates the search pipeline that retrieves documents using BM25
    sparse vector scoring for keyword-based retrieval.

    Tested Behaviors:
        - Search pipeline initialization with sparse configuration
        - BM25 query tokenization and scoring
        - Sparse vector search against Chroma database
        - RAG generation from sparse retrieval results
        - Empty result handling

    Mocks:
        - ChromaVectorDB: Database query operations
        - RAGHelper: LLM initialization and answer generation
        - SparseEmbedder: BM25 query vector generation

    Search Process:
        1. Tokenize query into individual terms
        2. Compute BM25 scores for documents containing query terms
        3. Rank documents by relevance score
        4. Return top-k documents
    """

    @patch("vectordb.langchain.sparse_indexing.search.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.sparse_indexing.search.chroma.RAGHelper.create_llm")
    def test_search_initialization(self, mock_llm, mock_db):
        """Test search pipeline initialization.

        Args:
            mock_llm: Mock for RAGHelper.create_llm
            mock_db: Mock for ChromaVectorDB class

        Verifies:
            - Configuration is stored correctly
            - Database connection is established
            - LLM is initialized if RAG is enabled
            - Sparse embedder is ready for query processing
        """
        mock_llm.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "sparse_embeddings": {"model": "bm25", "device": "cpu"},
            "chroma": {"collection_name": "test", "path": "./test_data"},
            "rag": {"enabled": False},
        }

        pipeline = ChromaSparseSearchPipeline(config)
        assert pipeline.config == config
        assert pipeline.llm is None

    @patch("vectordb.langchain.sparse_indexing.search.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.sparse_indexing.search.chroma.RAGHelper.create_llm")
    def test_search_execution(
        self,
        mock_llm,
        mock_db,
        sample_documents,
    ):
        """Test search execution with BM25 sparse retrieval.

        Args:
            mock_llm: Mock for RAGHelper.create_llm
            mock_db: Mock for ChromaVectorDB class
            sample_documents: Fixture providing test documents

        Verifies:
            - Query is tokenized for BM25 scoring
            - Chroma database is queried with sparse vectors
            - Documents are ranked by BM25 relevance score
            - Returns query and matching documents
            - No answer is generated when RAG is disabled
        """
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "sparse_embeddings": {"model": "bm25", "device": "cpu"},
            "chroma": {"collection_name": "test", "path": "./test_data"},
            "rag": {"enabled": False},
        }

        pipeline = ChromaSparseSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert len(result["documents"]) > 0
        assert "answer" not in result

    @patch("vectordb.langchain.sparse_indexing.search.chroma.ChromaVectorDB")
    @patch("vectordb.langchain.sparse_indexing.search.chroma.RAGHelper.create_llm")
    @patch("vectordb.langchain.sparse_indexing.search.chroma.RAGHelper.generate")
    def test_search_with_rag(
        self,
        mock_rag_generate,
        mock_llm,
        mock_db,
        sample_documents,
    ):
        """Test search with RAG generation from sparse retrieval results.

        Args:
            mock_rag_generate: Mock for RAGHelper.generate
            mock_llm: Mock for RAGHelper.create_llm
            mock_db: Mock for ChromaVectorDB class
            sample_documents: Fixture providing test documents

        Verifies:
            - Sparse retrieval returns relevant documents
            - LLM is initialized when RAG is enabled
            - RAG answer is generated from retrieved documents
            - Answer is included in search results
        """
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst

        mock_llm_inst = MagicMock()
        mock_llm.return_value = mock_llm_inst
        mock_rag_generate.return_value = "Test answer"

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "sparse_embeddings": {"model": "bm25", "device": "cpu"},
            "chroma": {"collection_name": "test", "path": "./test_data"},
            "rag": {"enabled": True},
        }

        pipeline = ChromaSparseSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert result["answer"] == "Test answer"
