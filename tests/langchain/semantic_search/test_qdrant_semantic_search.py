"""Tests for Qdrant semantic search pipelines using LangChain.

This module tests the semantic search pipeline implementation for Qdrant vector
database.
It validates both the indexing pipeline (document loading, embedding, storage) and the
search pipeline (query embedding, vector retrieval, RAG generation).

Indexing Pipeline:
    - Loads documents from configured dataloaders (ARC, TriviaQA, PopQA, etc.)
    - Embeds documents using specified embedding models
    - Stores vectors and metadata in Qdrant collections
    - Supports remote Qdrant server connections

Search Pipeline:
    - Embeds user queries using the same embedding model
    - Performs vector similarity search against Qdrant collections
    - Returns ranked documents by relevance score
    - Optional RAG mode: Generates answers using retrieved context

Database Configuration:
    - url: Qdrant server URL (e.g., http://localhost:6333)
    - collection_name: Name of the Qdrant collection to use

RAG Modes:
    - enabled=False: Returns only retrieved documents
    - enabled=True: Generates answers using LLM with retrieved context
"""

from unittest.mock import MagicMock, patch

from vectordb.langchain.semantic_search.indexing.qdrant import (
    QdrantSemanticIndexingPipeline,
)
from vectordb.langchain.semantic_search.search.qdrant import (
    QdrantSemanticSearchPipeline,
)


class TestQdrantSemanticIndexing:
    """Unit tests for Qdrant semantic indexing pipeline.

    Tests the indexing pipeline which orchestrates document loading, embedding,
    and vector storage operations for Qdrant vector database.

    Pipeline Flow:
        1. Load documents from configured dataloader with limit
        2. Embed documents using EmbedderHelper with specified model
        3. Initialize QdrantVectorDB with URL and collection name
        4. Upsert embeddings and metadata into Qdrant collection
        5. Return indexing statistics
    """

    @patch("vectordb.langchain.semantic_search.indexing.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.semantic_search.indexing.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.semantic_search.indexing.qdrant.DataloaderCatalog.create"
    )
    def test_indexing_initialization(
        self, mock_get_docs, mock_embedder_helper, mock_db
    ):
        """Test pipeline initialization with Qdrant-specific configuration.

        Verifies that the indexing pipeline correctly initializes with:
            - Dataloader configuration (type and document limit)
            - Embedding model settings (model name and device)
            - Qdrant connection parameters (server URL, collection name)
            - Collection name is extracted and stored as instance attribute

        Args:
            mock_get_docs: Mock for DataloaderCatalog.create
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for QdrantVectorDB class
        """
        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "collection_name": "test_semantic_search",
            },
        }

        pipeline = QdrantSemanticIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.collection_name == "test_semantic_search"

    @patch("vectordb.langchain.semantic_search.indexing.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.semantic_search.indexing.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.semantic_search.indexing.qdrant.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.semantic_search.indexing.qdrant.DataloaderCatalog.create"
    )
    def test_indexing_run_with_documents(
        self,
        mock_get_docs,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
        sample_langchain_documents,
    ):
        """Test full indexing pipeline execution with sample documents.

        Validates the complete indexing flow:
            1. Documents are loaded via dataloader helper
            2. Documents are embedded into 384-dimensional vectors
            3. QdrantVectorDB is initialized with config parameters
            4. Embeddings are upserted into Qdrant collection
            5. Returns count of indexed documents

        Args:
            mock_get_docs: Mock returning sample documents from dataloader
            mock_embed_docs: Mock returning embedded documents with vectors
            mock_embedder_helper: Mock for embedder initialization
            mock_db: Mock for QdrantVectorDB with upsert tracking
            sample_langchain_documents: Pytest fixture providing test documents
        """
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_langchain_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader
        mock_embed_docs.return_value = (
            sample_langchain_documents,
            [[0.1] * 384] * 5,
        )

        mock_db_inst = MagicMock()
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "collection_name": "test_semantic_search",
            },
        }

        pipeline = QdrantSemanticIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_langchain_documents)
        mock_db_inst.client.upsert.assert_called()

    @patch("vectordb.langchain.semantic_search.indexing.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.semantic_search.indexing.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.semantic_search.indexing.qdrant.DataloaderCatalog.create"
    )
    def test_indexing_run_no_documents(
        self, mock_get_docs, mock_embedder_helper, mock_db
    ):
        """Test indexing pipeline behavior when no documents are loaded.

        Verifies that when the dataloader returns an empty document list:
            - Pipeline completes without errors
            - No embeddings are generated
            - No database operations are performed
            - Returns documents_indexed count of 0

        Args:
            mock_get_docs: Mock returning empty document list
            mock_embedder_helper: Mock for embedder initialization
            mock_db: Mock for QdrantVectorDB
        """
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = []
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "collection_name": "test_semantic_search",
            },
        }

        pipeline = QdrantSemanticIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0


class TestQdrantSemanticSearch:
    """Unit tests for Qdrant semantic search pipeline.

    Tests the search pipeline which orchestrates query embedding, vector search,
    and optional RAG-based answer generation for Qdrant vector database.

    Pipeline Flow (RAG disabled):
        1. Embed user query using configured embedding model
        2. Query QdrantVectorDB with embedded query vector
        3. Return retrieved documents with relevance scores

    Pipeline Flow (RAG enabled):
        1. Embed user query using configured embedding model
        2. Query QdrantVectorDB with embedded query vector
        3. Generate answer using LLM with retrieved documents as context
        4. Return both documents and generated answer
    """

    @patch("vectordb.langchain.semantic_search.search.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.semantic_search.search.qdrant.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.semantic_search.search.qdrant.RAGHelper.create_llm")
    def test_search_initialization(self, mock_llm, mock_embedder_helper, mock_db):
        """Test search pipeline initialization with RAG disabled.

        Verifies that when RAG is disabled:
            - Configuration is stored correctly
            - Embedder is initialized via EmbedderHelper
            - QdrantVectorDB is initialized with config parameters
            - LLM is not created (mock_llm returns None)
            - llm attribute remains None

        Args:
            mock_llm: Mock for RAGHelper.create_llm returning None
            mock_embedder_helper: Mock for embedder initialization
            mock_db: Mock for QdrantVectorDB class
        """
        mock_llm.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "collection_name": "test_semantic_search",
            },
            "rag": {"enabled": False},
        }

        pipeline = QdrantSemanticSearchPipeline(config)
        assert pipeline.config == config
        assert pipeline.llm is None

    @patch("vectordb.langchain.semantic_search.search.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.semantic_search.search.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.semantic_search.search.qdrant.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.semantic_search.search.qdrant.RAGHelper.create_llm")
    def test_search_execution(
        self,
        mock_llm,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test search pipeline execution with RAG disabled.

        Validates the search flow without answer generation:
            1. Query is embedded into 384-dimensional vector
            2. QdrantVectorDB.search is called with embedded query
            3. Documents are returned in result with query text
            4. Answer field is not present in result

        Return Format:
            - query: Original query string
            - documents: List of retrieved Document objects

        Args:
            mock_llm: Mock for LLM creation (returns None)
            mock_embed_query: Mock returning embedded query vector
            mock_embedder_helper: Mock for embedder initialization
            mock_db: Mock for QdrantVectorDB with query results
            sample_documents: Pytest fixture providing test documents
        """
        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.search.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "collection_name": "test_semantic_search",
            },
            "rag": {"enabled": False},
        }

        pipeline = QdrantSemanticSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert len(result["documents"]) > 0
        assert "answer" not in result

    @patch("vectordb.langchain.semantic_search.search.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.semantic_search.search.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.semantic_search.search.qdrant.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.semantic_search.search.qdrant.RAGHelper.create_llm")
    @patch("vectordb.langchain.semantic_search.search.qdrant.RAGHelper.generate")
    def test_search_with_rag(
        self,
        mock_rag_generate,
        mock_llm,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test search pipeline execution with RAG enabled.

        Validates the search flow with answer generation:
            1. Query is embedded into 384-dimensional vector
            2. QdrantVectorDB.search retrieves relevant documents
            3. LLM is initialized via RAGHelper.create_llm
            4. RAGHelper.generate produces answer using query and context
            5. Result contains both documents and generated answer

        Return Format:
            - query: Original query string
            - documents: List of retrieved Document objects
            - answer: LLM-generated answer based on retrieved context

        Args:
            mock_rag_generate: Mock returning generated answer string
            mock_llm: Mock returning LLM instance
            mock_embed_query: Mock returning embedded query vector
            mock_embedder_helper: Mock for embedder initialization
            mock_db: Mock for QdrantVectorDB with search results
            sample_documents: Pytest fixture providing test documents
        """
        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.search.return_value = sample_documents
        mock_db.return_value = mock_db_inst

        mock_llm_inst = MagicMock()
        mock_llm.return_value = mock_llm_inst
        mock_rag_generate.return_value = "Test answer"

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "collection_name": "test_semantic_search",
            },
            "rag": {"enabled": True},
        }

        pipeline = QdrantSemanticSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert result["answer"] == "Test answer"
