"""Tests for Qdrant sparse indexing pipelines (LangChain)."""

from unittest.mock import MagicMock, patch

from vectordb.langchain.sparse_indexing.indexing.qdrant import (
    QdrantSparseIndexingPipeline,
)
from vectordb.langchain.sparse_indexing.search.qdrant import (
    QdrantSparseSearchPipeline,
)


class TestQdrantSparseIndexing:
    """Unit tests for Qdrant sparse indexing pipeline."""

    @patch("vectordb.langchain.sparse_indexing.indexing.base.DataloaderCatalog.create")
    @patch("vectordb.langchain.sparse_indexing.indexing.qdrant.QdrantVectorDB")
    def test_indexing_initialization(self, mock_db, mock_get_docs):
        """Test pipeline initialization."""
        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "sparse_embeddings": {"model": "bm25", "device": "cpu"},
            "qdrant": {"url": "http://localhost:6333", "collection_name": "test"},
        }

        pipeline = QdrantSparseIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.collection_name == "test"

    @patch("vectordb.langchain.sparse_indexing.indexing.base.DataloaderCatalog.create")
    @patch("vectordb.langchain.sparse_indexing.indexing.qdrant.QdrantVectorDB")
    def test_indexing_run_with_documents(
        self,
        mock_db,
        mock_get_docs,
        sample_documents,
    ):
        """Test indexing with documents."""
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader

        mock_db_inst = MagicMock()
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "sparse_embeddings": {"model": "bm25", "device": "cpu"},
            "qdrant": {"url": "http://localhost:6333", "collection_name": "test"},
        }

        pipeline = QdrantSparseIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.index_documents.assert_called_once()

    @patch("vectordb.langchain.sparse_indexing.indexing.base.DataloaderCatalog.create")
    @patch("vectordb.langchain.sparse_indexing.indexing.qdrant.QdrantVectorDB")
    def test_indexing_run_no_documents(self, mock_get_docs, mock_db):
        """Test indexing with no documents."""
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = []
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "sparse_embeddings": {"model": "bm25", "device": "cpu"},
            "qdrant": {"url": "http://localhost:6333", "collection_name": "test"},
        }

        pipeline = QdrantSparseIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0


class TestQdrantSparseSearch:
    """Unit tests for Qdrant sparse search pipeline."""

    @patch("vectordb.langchain.sparse_indexing.search.qdrant.QdrantVectorDB")
    @patch("vectordb.langchain.sparse_indexing.search.qdrant.RAGHelper.create_llm")
    def test_search_initialization(self, mock_llm, mock_db):
        """Test search pipeline initialization."""
        mock_llm.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "sparse_embeddings": {"model": "bm25", "device": "cpu"},
            "qdrant": {"url": "http://localhost:6333", "collection_name": "test"},
            "rag": {"enabled": False},
        }

        pipeline = QdrantSparseSearchPipeline(config)
        assert pipeline.config == config
        assert pipeline.llm is None

    @patch("vectordb.langchain.sparse_indexing.search.qdrant.QdrantVectorDB")
    @patch("vectordb.langchain.sparse_indexing.search.qdrant.RAGHelper.create_llm")
    def test_search_execution(
        self,
        mock_llm,
        mock_db,
        sample_documents,
    ):
        """Test search execution."""
        mock_db_inst = MagicMock()
        mock_db_inst.search.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "sparse_embeddings": {"model": "bm25", "device": "cpu"},
            "qdrant": {"url": "http://localhost:6333", "collection_name": "test"},
            "rag": {"enabled": False},
        }

        pipeline = QdrantSparseSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert len(result["documents"]) > 0
        assert "answer" not in result

    @patch("vectordb.langchain.sparse_indexing.search.qdrant.QdrantVectorDB")
    @patch("vectordb.langchain.sparse_indexing.search.qdrant.RAGHelper.create_llm")
    @patch("vectordb.langchain.sparse_indexing.search.qdrant.RAGHelper.generate")
    def test_search_with_rag(
        self,
        mock_rag_generate,
        mock_llm,
        mock_db,
        sample_documents,
    ):
        """Test search with RAG generation."""
        mock_db_inst = MagicMock()
        mock_db_inst.search.return_value = sample_documents
        mock_db.return_value = mock_db_inst

        mock_llm_inst = MagicMock()
        mock_llm.return_value = mock_llm_inst
        mock_rag_generate.return_value = "Test answer"

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "sparse_embeddings": {"model": "bm25", "device": "cpu"},
            "qdrant": {"url": "http://localhost:6333", "collection_name": "test"},
            "rag": {"enabled": True},
        }

        pipeline = QdrantSparseSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert result["answer"] == "Test answer"
