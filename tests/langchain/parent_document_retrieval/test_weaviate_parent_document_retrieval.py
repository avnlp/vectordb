"""Tests for Weaviate parent document retrieval pipelines (LangChain)."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


class TestWeaviateParentDocumentRetrievalIndexing:
    """Unit tests for Weaviate parent document retrieval indexing pipeline."""

    @patch(
        "vectordb.langchain.parent_document_retrieval.indexing.weaviate.WeaviateVectorDB"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.indexing.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.indexing.weaviate.DataloaderCatalog.create"
    )
    def test_indexing_initialization(
        self, mock_get_docs, mock_embedder_helper, mock_db
    ):
        """Test pipeline initialization."""
        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "",
                "collection_name": "TestParentDocumentRetrieval",
            },
        }

        from vectordb.langchain.parent_document_retrieval.indexing.weaviate import (
            WeaviateParentDocumentRetrievalIndexingPipeline,
        )

        pipeline = WeaviateParentDocumentRetrievalIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.collection_name == "TestParentDocumentRetrieval"

    @patch(
        "vectordb.langchain.parent_document_retrieval.indexing.weaviate.WeaviateVectorDB"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.indexing.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.indexing.weaviate.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.indexing.weaviate.DataloaderCatalog.create"
    )
    def test_indexing_run_with_documents(
        self,
        mock_get_docs,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test indexing with documents."""
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader
        mock_embed_docs.return_value = (sample_documents, [[0.1] * 384] * 10)

        mock_db_inst = MagicMock()
        mock_db_inst.upsert.return_value = 10
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "",
                "collection_name": "TestParentDocumentRetrieval",
            },
            "chunking": {
                "chunk_size": 50,
                "chunk_overlap": 10,
            },
        }

        from vectordb.langchain.parent_document_retrieval.indexing.weaviate import (
            WeaviateParentDocumentRetrievalIndexingPipeline,
        )

        pipeline = WeaviateParentDocumentRetrievalIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        assert result["chunks_created"] > 0

    @patch(
        "vectordb.langchain.parent_document_retrieval.indexing.weaviate.WeaviateVectorDB"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.indexing.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.indexing.weaviate.DataloaderCatalog.create"
    )
    def test_indexing_run_no_documents(
        self, mock_get_docs, mock_embedder_helper, mock_db
    ):
        """Test indexing with no documents."""
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = []
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "",
                "collection_name": "TestParentDocumentRetrieval",
            },
        }

        from vectordb.langchain.parent_document_retrieval.indexing.weaviate import (
            WeaviateParentDocumentRetrievalIndexingPipeline,
        )

        pipeline = WeaviateParentDocumentRetrievalIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0
        assert result["chunks_created"] == 0
        assert result["parent_store_path"] is None


class TestWeaviateParentDocumentRetrievalSearch:
    """Unit tests for Weaviate parent document retrieval search pipeline."""

    @patch(
        "vectordb.langchain.parent_document_retrieval.search.weaviate.WeaviateVectorDB"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.weaviate.RAGHelper.create_llm"
    )
    def test_search_initialization(
        self, mock_llm_helper, mock_embedder_helper, mock_db
    ):
        """Test search pipeline initialization."""
        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "",
                "collection_name": "TestParentDocumentRetrieval",
            },
            "rag": {"enabled": False},
        }

        from vectordb.langchain.parent_document_retrieval.search.weaviate import (
            WeaviateParentDocumentRetrievalSearchPipeline,
        )

        pipeline = WeaviateParentDocumentRetrievalSearchPipeline(config)
        assert pipeline.config == config
        assert pipeline.collection_name == "TestParentDocumentRetrieval"

    @patch(
        "vectordb.langchain.parent_document_retrieval.search.weaviate.WeaviateVectorDB"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.weaviate.EmbedderHelper.embed_query"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.weaviate.RAGHelper.create_llm"
    )
    def test_search_with_results(
        self,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test search with results."""
        mock_llm_helper.return_value = None
        mock_embed_query.return_value = [0.1] * 384

        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = [
            Document(
                page_content="Chunk 1",
                metadata={"id": "chunk_1", "parent_id": "parent_1"},
            ),
            Document(
                page_content="Chunk 2",
                metadata={"id": "chunk_2", "parent_id": "parent_2"},
            ),
        ]
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "",
                "collection_name": "TestParentDocumentRetrieval",
            },
            "rag": {"enabled": False},
        }

        from vectordb.langchain.parent_document_retrieval.search.weaviate import (
            WeaviateParentDocumentRetrievalSearchPipeline,
        )

        pipeline = WeaviateParentDocumentRetrievalSearchPipeline(config)

        # Set up parent store with test data
        from vectordb.langchain.parent_document_retrieval.parent_store import (
            ParentDocumentStore,
        )

        parent_store = ParentDocumentStore()
        parent_store.add_parent("parent_1", {"text": "Parent document 1"})
        parent_store.add_parent("parent_2", {"text": "Parent document 2"})
        parent_store.add_chunk_mapping("chunk_1", "parent_1")
        parent_store.add_chunk_mapping("chunk_2", "parent_2")
        pipeline.set_parent_store(parent_store)

        result = pipeline.search("test query", top_k=2)

        assert "parent_documents" in result
        assert result["query"] == "test query"

    @patch(
        "vectordb.langchain.parent_document_retrieval.search.weaviate.WeaviateVectorDB"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.weaviate.EmbedderHelper.embed_query"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.weaviate.RAGHelper.create_llm"
    )
    def test_search_with_filters(
        self,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test search with filters."""
        mock_llm_helper.return_value = None
        mock_embed_query.return_value = [0.1] * 384

        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = []
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "",
                "collection_name": "TestParentDocumentRetrieval",
            },
            "rag": {"enabled": False},
        }

        from vectordb.langchain.parent_document_retrieval.search.weaviate import (
            WeaviateParentDocumentRetrievalSearchPipeline,
        )

        pipeline = WeaviateParentDocumentRetrievalSearchPipeline(config)
        filters = {"source": "wiki"}
        result = pipeline.search("test query", top_k=5, filters=filters)

        assert "parent_documents" in result
        mock_db_inst.query.assert_called_once()
        call_kwargs = mock_db_inst.query.call_args.kwargs
        assert call_kwargs.get("where") == filters

    @patch(
        "vectordb.langchain.parent_document_retrieval.search.weaviate.WeaviateVectorDB"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.weaviate.EmbedderHelper.embed_query"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.weaviate.RAGHelper.create_llm"
    )
    @patch(
        "vectordb.langchain.parent_document_retrieval.search.weaviate.RAGHelper.generate"
    )
    def test_search_with_rag(
        self,
        mock_rag_generate,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test search with RAG generation."""
        mock_llm = MagicMock()
        mock_llm_helper.return_value = mock_llm
        mock_embed_query.return_value = [0.1] * 384
        mock_rag_generate.return_value = "Generated answer"

        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = [
            Document(
                page_content="Chunk 1",
                metadata={"id": "chunk_1", "parent_id": "parent_1"},
            ),
        ]
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "",
                "collection_name": "TestParentDocumentRetrieval",
            },
            "rag": {"enabled": True},
        }

        from vectordb.langchain.parent_document_retrieval.search.weaviate import (
            WeaviateParentDocumentRetrievalSearchPipeline,
        )

        pipeline = WeaviateParentDocumentRetrievalSearchPipeline(config)

        # Set up parent store with test data
        from vectordb.langchain.parent_document_retrieval.parent_store import (
            ParentDocumentStore,
        )

        parent_store = ParentDocumentStore()
        parent_store.add_parent("parent_1", {"text": "Parent document 1"})
        parent_store.add_chunk_mapping("chunk_1", "parent_1")
        pipeline.set_parent_store(parent_store)

        result = pipeline.search("test query", top_k=1)

        assert "answer" in result
        assert result["answer"] == "Generated answer"
