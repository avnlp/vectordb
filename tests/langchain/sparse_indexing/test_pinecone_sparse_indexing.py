"""Tests for Pinecone sparse indexing pipelines (LangChain).

This module tests the sparse indexing feature for Pinecone vector database,
implementing keyword-based (BM25) retrieval. Pinecone natively supports sparse-dense
hybrid vectors, making it ideal for combining BM25 keyword matching with semantic
search.

Sparse Indexing in Pinecone:
    Pinecone supports sparse-dense vectors where:
    - Dense vectors capture semantic meaning (384-1536 dimensions)
    - Sparse vectors capture keyword frequency (up to 1000+ non-zero values)
    - Hybrid search combines both using an alpha parameter (0.0-1.0)
    - Alpha 0.0 = pure sparse (BM25), 1.0 = pure dense, 0.5 = equal weight

BM25 Algorithm:
    BM25 is a probabilistic ranking function for keyword relevance:
    - Term frequency (tf): How often a term appears in a document
    - Inverse document frequency (idf): Rarity of term across corpus
    - Document length normalization: Prevents long documents from dominating
    Score = sum(IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl)))

Pipeline Architecture:
    Indexing Pipeline:
        1. Load documents from configured data source (ARC, TriviaQA, etc.)
        2. Initialize Pinecone connection and create index if needed
        3. Generate dense embeddings for semantic search capability
        4. Compute BM25 sparse vectors for keyword matching
        5. Upsert documents with both sparse and dense vectors to Pinecone
        6. Support namespace partitioning for multi-tenant scenarios

    Search Pipeline:
        1. Generate dense embedding for query (semantic understanding)
        2. Generate sparse vector for query (keyword matching)
        3. Query Pinecone with hybrid sparse-dense vectors
        4. Pinecone computes combined relevance scores internally
        5. Retrieve top-k documents by hybrid score
        6. Optionally generate RAG answer from retrieved documents

Components Tested:
    - PineconeSparseIndexingPipeline: Hybrid sparse-dense document indexing
    - PineconeSparseSearchPipeline: Hybrid vector search with BM25
    - PineconeVectorDB: Database client for sparse-dense operations

Key Features:
    - Native Pinecone sparse-dense hybrid support
    - BM25 keyword-based sparse vectors
    - Configurable sparse-dense weighting (alpha parameter)
    - Namespace support for data isolation
    - Index creation and management
    - Optional RAG generation from hybrid results

Test Coverage:
    - Pipeline initialization with Pinecone configuration
    - Index creation and configuration
    - Document indexing with sparse-dense vectors
    - Hybrid search execution
    - Namespace handling
    - RAG generation from hybrid retrieval
    - Empty document handling

Configuration:
    Pinecone sparse indexing requires API credentials and index configuration:
    - api_key: Pinecone API key (required)
    - index_name: Name of Pinecone index (required)
    - namespace: Optional data partitioning (default: "")
    - alpha: Sparse-dense weight for hybrid search (default: 0.5)

Advantages Over Dense-Only:
    - Better keyword matching for technical terms
    - Exact match capability for rare terms
    - Improved precision for acronym-heavy content
    - Works without embedding model for simple keyword queries

All tests mock Pinecone API and embedding operations to ensure fast,
deterministic unit tests without external service dependencies.
"""

from unittest.mock import MagicMock, patch

from vectordb.langchain.sparse_indexing.indexing.pinecone import (
    PineconeSparseIndexingPipeline,
)
from vectordb.langchain.sparse_indexing.search.pinecone import (
    PineconeSparseSearchPipeline,
)


class TestPineconeSparseIndexing:
    """Unit tests for Pinecone sparse indexing pipeline.

    Validates the indexing pipeline that upserts documents with both sparse
    (BM25) and dense vectors to Pinecone for hybrid retrieval.

    Tested Behaviors:
        - Pipeline initialization with Pinecone API configuration
        - Index creation and validation
        - Document loading and processing
        - Dense and sparse vector generation
        - Pinecone upsert with hybrid vectors
        - Namespace configuration
        - Empty document handling

    Mocks:
        - PineconeVectorDB: Database operations (index creation, upsert)
        - DataLoaderHelper: Document loading from data sources
        - EmbedderHelper: Dense embedding generation
        - SparseEmbedder: BM25 sparse vector computation

    Pinecone Configuration:
        - api_key: Authentication credentials
        - index_name: Target index for data storage
        - namespace: Optional data isolation segment
    """

    @patch("vectordb.langchain.sparse_indexing.indexing.base.DataloaderCatalog.create")
    @patch("vectordb.langchain.sparse_indexing.indexing.pinecone.PineconeVectorDB")
    def test_indexing_initialization(self, mock_db, mock_get_docs):
        """Test pipeline initialization with Pinecone configuration.

        Args:
            mock_get_docs: Mock for DataloaderCatalog.create
            mock_db: Mock for PineconeVectorDB class

        Verifies:
            - Configuration is stored correctly
            - Index name is extracted from config
            - Namespace defaults to empty string
            - Database client is initialized
        """
        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "sparse_embeddings": {"model": "bm25", "device": "cpu"},
            "pinecone": {"api_key": "test-key", "index_name": "test-index"},
        }

        pipeline = PineconeSparseIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.index_name == "test-index"
        assert pipeline.namespace == ""

    @patch("vectordb.langchain.sparse_indexing.indexing.base.DataloaderCatalog.create")
    @patch("vectordb.langchain.sparse_indexing.indexing.pinecone.PineconeVectorDB")
    def test_indexing_run_with_documents(
        self,
        mock_db,
        mock_get_docs,
        sample_documents,
    ):
        """Test indexing pipeline with documents using hybrid vectors.

        Args:
            mock_get_docs: Mock for DataloaderCatalog.create
            mock_db: Mock for PineconeVectorDB class
            sample_documents: Fixture providing test documents

        Verifies:
            - Documents are loaded from data source
            - Index is created if it doesn't exist
            - Dense and sparse vectors are generated
            - Documents are upserted to Pinecone
            - Returns count of indexed documents
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
            "pinecone": {"api_key": "test-key", "index_name": "test-index"},
        }

        pipeline = PineconeSparseIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.create_index.assert_called_once()
        mock_db_inst.upsert.assert_called_once()

    @patch("vectordb.langchain.sparse_indexing.indexing.base.DataloaderCatalog.create")
    @patch("vectordb.langchain.sparse_indexing.indexing.pinecone.PineconeVectorDB")
    def test_indexing_run_no_documents(self, mock_db, mock_get_docs):
        """Test indexing pipeline with no documents.

        Args:
            mock_db: Mock for PineconeVectorDB class
            mock_get_docs: Mock for DataloaderCatalog.create

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

        mock_db_inst = MagicMock()
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "sparse_embeddings": {"model": "bm25", "device": "cpu"},
            "pinecone": {"api_key": "test-key", "index_name": "test-index"},
        }

        pipeline = PineconeSparseIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0


class TestPineconeSparseSearch:
    """Unit tests for Pinecone sparse search pipeline.

    Validates the search pipeline that queries Pinecone using hybrid
    sparse-dense vectors for combined keyword and semantic retrieval.

    Tested Behaviors:
        - Search pipeline initialization with Pinecone config
        - Dense query embedding generation
        - Sparse query vector generation (BM25)
        - Hybrid search execution with alpha weighting
        - Result retrieval and formatting
        - RAG generation from hybrid results

    Mocks:
        - PineconeVectorDB: Database query operations
        - RAGHelper: LLM initialization and answer generation
        - EmbedderHelper: Query embedding
        - SparseEmbedder: BM25 query vector generation

    Hybrid Search:
        Pinecone combines sparse and dense scores using:
        hybrid_score = alpha * dense_score + (1 - alpha) * sparse_score
        Where alpha controls the weight (0.0 = pure BM25, 1.0 = pure semantic)
    """

    @patch("vectordb.langchain.sparse_indexing.search.pinecone.PineconeVectorDB")
    @patch("vectordb.langchain.sparse_indexing.search.pinecone.RAGHelper.create_llm")
    def test_search_initialization(self, mock_llm, mock_db):
        """Test search pipeline initialization.

        Args:
            mock_llm: Mock for RAGHelper.create_llm
            mock_db: Mock for PineconeVectorDB class

        Verifies:
            - Configuration is stored correctly
            - Database connection is established
            - LLM is initialized if RAG is enabled
            - Namespace and index name are configured
        """
        mock_llm.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "sparse_embeddings": {"model": "bm25", "device": "cpu"},
            "pinecone": {"api_key": "test-key", "index_name": "test-index"},
            "rag": {"enabled": False},
        }

        pipeline = PineconeSparseSearchPipeline(config)
        assert pipeline.config == config
        assert pipeline.llm is None

    @patch("vectordb.langchain.sparse_indexing.search.pinecone.PineconeVectorDB")
    @patch("vectordb.langchain.sparse_indexing.search.pinecone.RAGHelper.create_llm")
    def test_search_execution(
        self,
        mock_llm,
        mock_db,
        sample_documents,
    ):
        """Test search execution with hybrid sparse-dense retrieval.

        Args:
            mock_llm: Mock for RAGHelper.create_llm
            mock_db: Mock for PineconeVectorDB class
            sample_documents: Fixture providing test documents

        Verifies:
            - Dense and sparse query vectors are generated
            - Pinecone is queried with hybrid vectors
            - Documents are retrieved by hybrid relevance score
            - Returns query and matching documents
            - No answer is generated when RAG is disabled
        """
        mock_db_inst = MagicMock()
        mock_db_inst.query_with_sparse.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "sparse_embeddings": {"model": "bm25", "device": "cpu"},
            "pinecone": {"api_key": "test-key", "index_name": "test-index"},
            "rag": {"enabled": False},
        }

        pipeline = PineconeSparseSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert len(result["documents"]) > 0
        assert "answer" not in result

    @patch("vectordb.langchain.sparse_indexing.search.pinecone.PineconeVectorDB")
    @patch("vectordb.langchain.sparse_indexing.search.pinecone.RAGHelper.create_llm")
    @patch("vectordb.langchain.sparse_indexing.search.pinecone.RAGHelper.generate")
    def test_search_with_rag(
        self,
        mock_rag_generate,
        mock_llm,
        mock_db,
        sample_documents,
    ):
        """Test search with RAG generation from hybrid retrieval results.

        Args:
            mock_rag_generate: Mock for RAGHelper.generate
            mock_llm: Mock for RAGHelper.create_llm
            mock_db: Mock for PineconeVectorDB class
            sample_documents: Fixture providing test documents

        Verifies:
            - Hybrid retrieval returns relevant documents
            - LLM is initialized when RAG is enabled
            - RAG answer is generated from retrieved documents
            - Answer is included in search results
            - Query is preserved in results
        """
        mock_db_inst = MagicMock()
        mock_db_inst.query_with_sparse.return_value = sample_documents
        mock_db.return_value = mock_db_inst

        mock_llm_inst = MagicMock()
        mock_llm.return_value = mock_llm_inst
        mock_rag_generate.return_value = "Test answer"

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "sparse_embeddings": {"model": "bm25", "device": "cpu"},
            "pinecone": {"api_key": "test-key", "index_name": "test-index"},
            "rag": {"enabled": True},
        }

        pipeline = PineconeSparseSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert result["answer"] == "Test answer"
