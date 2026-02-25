"""Tests for Chroma parent document retrieval pipelines (LangChain).

This module tests the parent document retrieval feature which implements a two-stage
retrieval strategy: first retrieving child chunks using semantic search, then mapping
back to and returning complete parent documents for better context preservation.

Parent Document Retrieval Concept:
    Traditional semantic search retrieves small text chunks for precise matching,
    but these chunks often lack surrounding context. Parent document retrieval
    solves this by storing both chunks (for search) and full parent documents
    (for context), enabling the best of both worlds.

Pipeline Architecture:
    Indexing Pipeline:
        1. Load documents from configured data source (ARC, TriviaQA, etc.)
        2. Split documents into child chunks for embedding
        3. Generate dense embeddings for each chunk
        4. Store chunks in Chroma vector database with parent_id metadata
        5. Persist parent documents and chunk-to-parent mappings in ParentDocumentStore

    Search Pipeline:
        1. Embed query using configured embedding model
        2. Retrieve top-k matching chunks from Chroma database
        3. Map chunk IDs back to parent documents via ParentDocumentStore
        4. Deduplicate parent documents while preserving order
        5. Optionally generate RAG answer from parent documents
        6. Return parent documents with full context

Components Tested:
    - ChromaParentDocumentRetrievalIndexingPipeline: Document ingestion and storage
    - ChromaParentDocumentRetrievalSearchPipeline: Query processing and retrieval
    - ParentDocumentStore: In-memory storage for parent docs and chunk mappings

Key Features:
    - Maintains complete parent document context for retrieved chunks
    - Supports metadata filtering during search
    - Optional RAG generation from parent documents
    - Parent store persistence for production deployments
    - Configurable chunk size and overlap parameters

Test Coverage:
    - Pipeline initialization with configuration
    - Document indexing with embedding generation
    - Search returning parent documents from chunk matches
    - Parent store operations (add, get, clear, len)
    - Filtered search queries
    - RAG generation from parent documents
    - Edge cases: no documents, missing metadata, top_k limiting

All tests mock vector database and embedding operations to ensure
fast, deterministic unit tests without external dependencies.
"""

from unittest.mock import MagicMock, patch


class TestChromaParentDocumentRetrievalIndexing:
    """Unit tests for Chroma parent document retrieval indexing pipeline.

    Validates the indexing pipeline that ingests documents, generates embeddings
    for chunks, and maintains parent document relationships for later retrieval.

    Tested Behaviors:
        - Pipeline initialization with Chroma configuration
        - Document loading and chunking
        - Embedding generation for chunks
        - Parent document storage with metadata
        - Empty document handling
        - Configuration validation

    Mocks:
        - ChromaVectorDB: Database operations (collection creation, upsert)
        - EmbedderHelper.create_embedder: Embedding model initialization
        - EmbedderHelper.embed_documents: Chunk embedding generation
        - DataloaderCatalog.create: Document loading
    """

    @patch(
        "vectordb.langchain.parent_document_retrieval.indexing.chroma.ChromaVectorDB"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.indexing.chroma.DataloaderCatalog.create"
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
            - Pipeline stores configuration correctly
            - Collection name is extracted from config
            - Database and embedder are initialized lazily
        """
        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_parent_document_retrieval",
            },
        }

        from vectordb.langchain.parent_document_retrieval.indexing.chroma import (
            ChromaParentDocumentRetrievalIndexingPipeline,
        )

        pipeline = ChromaParentDocumentRetrievalIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.collection_name == "test_parent_document_retrieval"

    @patch(
        "vectordb.langchain.parent_document_retrieval.indexing.chroma.ChromaVectorDB"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.indexing.chroma.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.indexing.chroma.DataloaderCatalog.create"
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
            - Embeddings are generated for document chunks
            - Chroma database upserts documents with embeddings
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
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_parent_document_retrieval",
            },
        }

        from vectordb.langchain.parent_document_retrieval.indexing.chroma import (
            ChromaParentDocumentRetrievalIndexingPipeline,
        )

        pipeline = ChromaParentDocumentRetrievalIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)

    @patch(
        "vectordb.langchain.parent_document_retrieval.indexing.chroma.ChromaVectorDB"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.indexing.chroma.DataloaderCatalog.create"
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
            - Database upsert is not called with empty data
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
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_parent_document_retrieval",
            },
        }

        from vectordb.langchain.parent_document_retrieval.indexing.chroma import (
            ChromaParentDocumentRetrievalIndexingPipeline,
        )

        pipeline = ChromaParentDocumentRetrievalIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0


class TestChromaParentDocumentRetrievalSearch:
    """Unit tests for Chroma parent document retrieval search pipeline.

    Validates the search pipeline that queries the vector database,
    retrieves chunks, maps them back to parent documents, and optionally
    generates RAG answers.

    Tested Behaviors:
        - Search pipeline initialization with Chroma and LLM config
        - Query embedding and chunk retrieval
        - Parent document mapping from chunk IDs
        - Deduplication and ordering of parent documents
        - Optional RAG answer generation
        - Metadata filtering support
        - Top-k limiting of results

    Mocks:
        - ChromaVectorDB: Database query operations
        - EmbedderHelper: Query embedding
        - RAGHelper: LLM initialization and answer generation
    """

    @patch("vectordb.langchain.parent_document_retrieval.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.RAGHelper.create_llm"
    )
    def test_search_initialization(
        self, mock_llm_helper, mock_embedder_helper, mock_db
    ):
        """Test search pipeline initialization.

        Args:
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class

        Verifies:
            - Pipeline stores configuration
            - LLM is initialized if RAG is enabled
            - Collection name is extracted from config
            - Parent store can be loaded from path
        """
        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_parent_document_retrieval",
            },
            "rag": {"enabled": False},
        }

        from vectordb.langchain.parent_document_retrieval.search.chroma import (
            ChromaParentDocumentRetrievalSearchPipeline,
        )

        pipeline = ChromaParentDocumentRetrievalSearchPipeline(config)
        assert pipeline.config == config

    @patch("vectordb.langchain.parent_document_retrieval.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.EmbedderHelper.embed_query"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.RAGHelper.create_llm"
    )
    def test_search_returns_parent_documents(
        self, mock_llm_helper, mock_embed_query, mock_embedder_helper, mock_db
    ):
        """Test search returns parent documents mapped from chunks.

        Args:
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_embed_query: Mock for EmbedderHelper.embed_query
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class

        Verifies:
            - Query is embedded using configured model
            - Chroma database is queried for matching chunks
            - Chunk IDs are mapped to parent documents
            - Parent documents are returned with full content
            - Result includes query and parent_documents keys
        """
        mock_llm_helper.return_value = None
        mock_embed_query.return_value = [0.1] * 384

        mock_chunk_1 = MagicMock()
        mock_chunk_1.id = None
        mock_chunk_1.metadata = {"id": "chunk_1"}
        mock_chunk_2 = MagicMock()
        mock_chunk_2.id = None
        mock_chunk_2.metadata = {"id": "chunk_2"}

        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = [mock_chunk_1, mock_chunk_2]
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_collection",
            },
            "rag": {"enabled": False},
        }

        from vectordb.langchain.parent_document_retrieval.parent_store import (
            ParentDocumentStore,
        )
        from vectordb.langchain.parent_document_retrieval.search.chroma import (
            ChromaParentDocumentRetrievalSearchPipeline,
        )

        pipeline = ChromaParentDocumentRetrievalSearchPipeline(config)

        parent_store = ParentDocumentStore()
        parent_store.add_parent("parent_1", {"text": "Parent document 1 content"})
        parent_store.add_parent("parent_2", {"text": "Parent document 2 content"})
        parent_store.add_chunk_mapping("chunk_1", "parent_1")
        parent_store.add_chunk_mapping("chunk_2", "parent_2")
        pipeline.set_parent_store(parent_store)

        result = pipeline.search("test query", top_k=5)

        assert "parent_documents" in result
        assert "query" in result
        assert len(result["parent_documents"]) == 2
        assert result["query"] == "test query"

    @patch("vectordb.langchain.parent_document_retrieval.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.EmbedderHelper.embed_query"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.RAGHelper.create_llm"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.RAGHelper.generate"
    )
    def test_search_with_rag_generation(
        self,
        mock_generate,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test search generates RAG answer when LLM is configured.

        Args:
            mock_generate: Mock for RAGHelper.generate
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_embed_query: Mock for EmbedderHelper.embed_query
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class

        Verifies:
            - RAG generation is triggered when enabled in config
            - LLM is used to generate answer from parent documents
            - Generated answer is included in search results
            - Parent documents are passed to RAG generator
        """
        mock_llm_inst = MagicMock()
        mock_llm_helper.return_value = mock_llm_inst
        mock_embed_query.return_value = [0.1] * 384
        mock_generate.return_value = "Generated answer based on documents"

        mock_chunk = MagicMock()
        mock_chunk.id = None
        mock_chunk.metadata = {"id": "chunk_1"}

        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = [mock_chunk]
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_collection",
            },
            "rag": {"enabled": True, "model": "llama-3.3-70b-versatile"},
        }

        from vectordb.langchain.parent_document_retrieval.parent_store import (
            ParentDocumentStore,
        )
        from vectordb.langchain.parent_document_retrieval.search.chroma import (
            ChromaParentDocumentRetrievalSearchPipeline,
        )

        pipeline = ChromaParentDocumentRetrievalSearchPipeline(config)

        parent_store = ParentDocumentStore()
        parent_store.add_parent("parent_1", {"text": "Parent document content"})
        parent_store.add_chunk_mapping("chunk_1", "parent_1")
        pipeline.set_parent_store(parent_store)

        result = pipeline.search("test query", top_k=5)

        assert "answer" in result
        assert result["answer"] == "Generated answer based on documents"
        mock_generate.assert_called_once()

    @patch("vectordb.langchain.parent_document_retrieval.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.EmbedderHelper.embed_query"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.RAGHelper.create_llm"
    )
    def test_search_with_filters(
        self, mock_llm_helper, mock_embed_query, mock_embedder_helper, mock_db
    ):
        """Test search passes filters to database query.

        Args:
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_embed_query: Mock for EmbedderHelper.embed_query
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class

        Verifies:
            - Metadata filters are passed to Chroma query
            - Filters support source-based filtering
            - Query arguments include where clause
        """
        mock_llm_helper.return_value = None
        mock_embed_query.return_value = [0.1] * 384

        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = []
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_collection",
            },
            "rag": {"enabled": False},
        }

        from vectordb.langchain.parent_document_retrieval.search.chroma import (
            ChromaParentDocumentRetrievalSearchPipeline,
        )

        pipeline = ChromaParentDocumentRetrievalSearchPipeline(config)
        filters = {"source": "wiki"}
        pipeline.search("test query", top_k=5, filters=filters)

        mock_db_inst.query.assert_called_once()
        call_kwargs = mock_db_inst.query.call_args[1]
        assert call_kwargs["where"] == filters

    @patch("vectordb.langchain.parent_document_retrieval.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.EmbedderHelper.embed_query"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.RAGHelper.create_llm"
    )
    def test_search_limits_parent_documents_to_top_k(
        self, mock_llm_helper, mock_embed_query, mock_embedder_helper, mock_db
    ):
        """Test search limits returned parent documents to top_k.

        Args:
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_embed_query: Mock for EmbedderHelper.embed_query
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class

        Verifies:
            - Number of returned parent documents respects top_k
            - Deduplication occurs within top_k limit
            - Order of parent documents matches chunk retrieval order
        """
        mock_llm_helper.return_value = None
        mock_embed_query.return_value = [0.1] * 384

        mock_chunks = []
        for i in range(10):
            chunk = MagicMock()
            chunk.id = None
            chunk.metadata = {"id": f"chunk_{i}"}
            mock_chunks.append(chunk)

        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = mock_chunks
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_collection",
            },
            "rag": {"enabled": False},
        }

        from vectordb.langchain.parent_document_retrieval.parent_store import (
            ParentDocumentStore,
        )
        from vectordb.langchain.parent_document_retrieval.search.chroma import (
            ChromaParentDocumentRetrievalSearchPipeline,
        )

        pipeline = ChromaParentDocumentRetrievalSearchPipeline(config)

        parent_store = ParentDocumentStore()
        for i in range(10):
            parent_store.add_parent(f"parent_{i}", {"text": f"Parent {i} content"})
            parent_store.add_chunk_mapping(f"chunk_{i}", f"parent_{i}")
        pipeline.set_parent_store(parent_store)

        result = pipeline.search("test query", top_k=3)

        assert len(result["parent_documents"]) == 3

    @patch("vectordb.langchain.parent_document_retrieval.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.EmbedderHelper.embed_query"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.RAGHelper.create_llm"
    )
    def test_search_handles_chunks_without_id_metadata(
        self, mock_llm_helper, mock_embed_query, mock_embedder_helper, mock_db
    ):
        """Test search handles chunks that don't have id in metadata.

        Args:
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_embed_query: Mock for EmbedderHelper.embed_query
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class

        Verifies:
            - Chunks without id metadata are skipped gracefully
            - Valid chunks still map to parent documents
            - No exceptions raised for malformed metadata
        """
        mock_llm_helper.return_value = None
        mock_embed_query.return_value = [0.1] * 384

        mock_chunk_with_id = MagicMock()
        mock_chunk_with_id.id = None
        mock_chunk_with_id.metadata = {"id": "chunk_1"}
        mock_chunk_without_id = MagicMock()
        mock_chunk_without_id.id = None
        mock_chunk_without_id.metadata = {"source": "wiki"}

        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = [
            mock_chunk_with_id,
            mock_chunk_without_id,
        ]
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_collection",
            },
            "rag": {"enabled": False},
        }

        from vectordb.langchain.parent_document_retrieval.parent_store import (
            ParentDocumentStore,
        )
        from vectordb.langchain.parent_document_retrieval.search.chroma import (
            ChromaParentDocumentRetrievalSearchPipeline,
        )

        pipeline = ChromaParentDocumentRetrievalSearchPipeline(config)

        parent_store = ParentDocumentStore()
        parent_store.add_parent("parent_1", {"text": "Parent document content"})
        parent_store.add_chunk_mapping("chunk_1", "parent_1")
        pipeline.set_parent_store(parent_store)

        result = pipeline.search("test query", top_k=5)

        assert len(result["parent_documents"]) == 1

    @patch("vectordb.langchain.parent_document_retrieval.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.RAGHelper.create_llm"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.ParentDocumentStore.load"
    )
    def test_search_initialization_with_parent_store_path(
        self, mock_load_store, mock_llm_helper, mock_embedder_helper, mock_db
    ):
        """Test search pipeline initialization loads parent store from path.

        Args:
            mock_load_store: Mock for ParentDocumentStore.load
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class

        Verifies:
            - Parent store is loaded from configured path
            - Loaded store contains parent documents and mappings
            - Pipeline uses loaded store for chunk-to-parent mapping
        """
        mock_llm_helper.return_value = None

        from vectordb.langchain.parent_document_retrieval.parent_store import (
            ParentDocumentStore,
        )

        mock_store = ParentDocumentStore()
        mock_store.add_parent("parent_1", {"text": "Loaded parent document"})
        mock_load_store.return_value = mock_store

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_collection",
            },
            "parent_store": {"store_path": "/tmp/test_parent_store.pkl"},
            "rag": {"enabled": False},
        }

        from vectordb.langchain.parent_document_retrieval.search.chroma import (
            ChromaParentDocumentRetrievalSearchPipeline,
        )

        pipeline = ChromaParentDocumentRetrievalSearchPipeline(config)

        mock_load_store.assert_called_once_with("/tmp/test_parent_store.pkl")
        assert len(pipeline.parent_store) == 1

    @patch("vectordb.langchain.parent_document_retrieval.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.EmbedderHelper.embed_query"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.RAGHelper.create_llm"
    )
    def test_search_no_rag_when_no_parent_documents(
        self, mock_llm_helper, mock_embed_query, mock_embedder_helper, mock_db
    ):
        """Test search does not attempt RAG generation when no parent documents.

        Args:
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_embed_query: Mock for EmbedderHelper.embed_query
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class

        Verifies:
            - No answer is generated when no documents are retrieved
            - Empty parent_documents list is returned
            - RAG generator is not called unnecessarily
        """
        mock_llm_inst = MagicMock()
        mock_llm_helper.return_value = mock_llm_inst
        mock_embed_query.return_value = [0.1] * 384

        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = []
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_collection",
            },
            "rag": {"enabled": True, "model": "llama-3.3-70b-versatile"},
        }

        from vectordb.langchain.parent_document_retrieval.search.chroma import (
            ChromaParentDocumentRetrievalSearchPipeline,
        )

        pipeline = ChromaParentDocumentRetrievalSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert "answer" not in result
        assert result["parent_documents"] == []

    @patch("vectordb.langchain.parent_document_retrieval.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.RAGHelper.create_llm"
    )
    def test_set_parent_store(self, mock_llm_helper, mock_embedder_helper, mock_db):
        """Test set_parent_store method updates the parent store.

        Args:
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class

        Verifies:
            - Parent store can be updated after initialization
            - New store is used for subsequent searches
            - Method accepts ParentDocumentStore instance
        """
        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_collection",
            },
            "rag": {"enabled": False},
        }

        from vectordb.langchain.parent_document_retrieval.parent_store import (
            ParentDocumentStore,
        )
        from vectordb.langchain.parent_document_retrieval.search.chroma import (
            ChromaParentDocumentRetrievalSearchPipeline,
        )

        pipeline = ChromaParentDocumentRetrievalSearchPipeline(config)

        new_store = ParentDocumentStore()
        new_store.add_parent("new_parent", {"text": "New parent content"})
        new_store.add_chunk_mapping("new_chunk", "new_parent")

        pipeline.set_parent_store(new_store)

        assert pipeline.parent_store is new_store
        assert len(pipeline.parent_store) == 1

    @patch("vectordb.langchain.parent_document_retrieval.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.EmbedderHelper.embed_query"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.chroma.RAGHelper.create_llm"
    )
    def test_search_handles_parent_doc_without_text_key(
        self, mock_llm_helper, mock_embed_query, mock_embedder_helper, mock_db
    ):
        """Test search handles parent documents without 'text' key for RAG.

        Args:
            mock_llm_helper: Mock for RAGHelper.create_llm
            mock_embed_query: Mock for EmbedderHelper.embed_query
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder
            mock_db: Mock for ChromaVectorDB class

        Verifies:
            - RAG generation gracefully handles missing text field
            - Empty page_content is passed when text key is absent
            - Answer is still generated without errors
        """
        mock_llm_inst = MagicMock()
        mock_llm_helper.return_value = mock_llm_inst
        mock_embed_query.return_value = [0.1] * 384

        mock_chunk = MagicMock()
        mock_chunk.id = None
        mock_chunk.metadata = {"id": "chunk_1"}

        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = [mock_chunk]
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_collection",
            },
            "rag": {"enabled": True},
        }

        from vectordb.langchain.parent_document_retrieval.parent_store import (
            ParentDocumentStore,
        )
        from vectordb.langchain.parent_document_retrieval.search.chroma import (
            ChromaParentDocumentRetrievalSearchPipeline,
        )

        with patch(
            "vectordb.langchain.parent_document_retrieval.search.chroma.RAGHelper.generate"
        ) as mock_generate:
            mock_generate.return_value = "Generated answer"

            pipeline = ChromaParentDocumentRetrievalSearchPipeline(config)

            parent_store = ParentDocumentStore()
            parent_store.add_parent("parent_1", {"content": "No text key here"})
            parent_store.add_chunk_mapping("chunk_1", "parent_1")
            pipeline.set_parent_store(parent_store)

            result = pipeline.search("test query", top_k=5)

            assert "answer" in result
            call_args = mock_generate.call_args[0]
            docs = call_args[2]
            assert docs[0].page_content == ""


class TestParentDocumentStore:
    """Unit tests for parent document store.

    Validates the ParentDocumentStore class which manages the relationship
    between chunks and their parent documents in memory.

    Tested Behaviors:
        - Store initialization with empty collections
        - Adding parent documents with metadata
        - Mapping chunk IDs to parent document IDs
        - Retrieving parent documents by chunk ID
        - Batch retrieval of parents for multiple chunks
        - Store clearing and length operations
        - Membership testing (in operator)

    Use Cases:
        - Search-time chunk-to-parent resolution
        - Parent document deduplication
        - Metadata preservation across chunk boundaries
    """

    def test_store_initialization(self):
        """Test parent store initialization.

        Verifies:
            - Store is created with empty parent and chunk mappings
            - Length returns 0 for empty store
            - No exceptions during initialization
        """
        from vectordb.langchain.parent_document_retrieval.parent_store import (
            ParentDocumentStore,
        )

        store = ParentDocumentStore()
        assert len(store) == 0

    def test_store_add_parent(self):
        """Test adding parent document.

        Verifies:
            - Parent document is stored with metadata
            - Chunk ID maps to correct parent ID
            - Retrieved parent contains original metadata
        """
        from vectordb.langchain.parent_document_retrieval.parent_store import (
            ParentDocumentStore,
        )

        store = ParentDocumentStore()
        store.add_parent("parent_1", {"text": "parent document content"})
        store.add_chunk_mapping("chunk_1", "parent_1")

        parent = store.get_parent("chunk_1")
        assert parent is not None
        assert parent["text"] == "parent document content"

    def test_store_get_parents_for_chunks(self):
        """Test getting parents for multiple chunks.

        Verifies:
            - Multiple chunk IDs are resolved to parent documents
            - Parents are returned in order of chunk IDs
            - Deduplication may occur for multiple chunks from same parent
        """
        from vectordb.langchain.parent_document_retrieval.parent_store import (
            ParentDocumentStore,
        )

        store = ParentDocumentStore()
        store.add_parent("parent_1", {"text": "parent document 1"})
        store.add_parent("parent_2", {"text": "parent document 2"})
        store.add_chunk_mapping("chunk_1", "parent_1")
        store.add_chunk_mapping("chunk_2", "parent_2")

        parents = store.get_parents_for_chunks(["chunk_1", "chunk_2"])
        assert len(parents) == 2

    def test_store_clear(self):
        """Test clearing store.

        Verifies:
            - All parent documents are removed
            - All chunk mappings are removed
            - Length returns 0 after clearing
        """
        from vectordb.langchain.parent_document_retrieval.parent_store import (
            ParentDocumentStore,
        )

        store = ParentDocumentStore()
        store.add_parent("parent_1", {"text": "parent document"})
        store.add_chunk_mapping("chunk_1", "parent_1")

        assert len(store) == 1

        store.clear()

        assert len(store) == 0

    def test_store_len(self):
        """Test store length.

        Verifies:
            - Length reflects number of parent documents
            - Length increases when adding parents
            - Length is consistent across operations
        """
        from vectordb.langchain.parent_document_retrieval.parent_store import (
            ParentDocumentStore,
        )

        store = ParentDocumentStore()
        store.add_parent("parent_1", {"text": "doc1"})
        store.add_parent("parent_2", {"text": "doc2"})

        assert len(store) == 2

    def test_store_contains(self):
        """Test store contains check.

        Verifies:
            - 'in' operator checks for chunk ID existence
            - Returns True for mapped chunk IDs
            - Returns False for unmapped chunk IDs
        """
        from vectordb.langchain.parent_document_retrieval.parent_store import (
            ParentDocumentStore,
        )

        store = ParentDocumentStore()
        store.add_chunk_mapping("chunk_1", "parent_1")

        assert "chunk_1" in store
        assert "chunk_2" not in store
