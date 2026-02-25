"""Tests for Chroma cost-optimized RAG pipelines (LangChain)."""

from unittest.mock import MagicMock, patch

from vectordb.langchain.cost_optimized_rag.indexing.chroma import (
    ChromaCostOptimizedRAGIndexingPipeline,
)


class TestChromaCostOptimizedRAGIndexing:
    """Unit tests for Chroma cost-optimized RAG indexing pipeline."""

    @patch("vectordb.langchain.cost_optimized_rag.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.cost_optimized_rag.indexing.chroma.SparseEmbedder")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.chroma.DataloaderCatalog.create"
    )
    def test_indexing_initialization(
        self, mock_get_docs, mock_sparse_embedder_cls, mock_embedder_helper, mock_db
    ):
        """Test pipeline initialization."""
        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_cost_optimized_rag",
            },
            "chunking": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
            },
        }

        pipeline = ChromaCostOptimizedRAGIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.collection_name == "test_cost_optimized_rag"

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
        mock_sparse_embedder_cls,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
        sample_langchain_documents,
    ):
        """Test indexing with documents."""
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_langchain_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader
        mock_embed_docs.return_value = (sample_langchain_documents, [[0.1] * 384] * 5)

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder.embed_documents.return_value = [
            [0.5, 0.3, 0.2] for _ in range(5)
        ]
        mock_sparse_embedder_cls.return_value = mock_sparse_embedder

        mock_db_inst = MagicMock()
        mock_db_inst.upsert.return_value = len(sample_langchain_documents)
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_cost_optimized_rag",
            },
            "chunking": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
            },
        }

        pipeline = ChromaCostOptimizedRAGIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_langchain_documents)
        assert "chunks_created" in result

    @patch("vectordb.langchain.cost_optimized_rag.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.cost_optimized_rag.indexing.chroma.SparseEmbedder")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.chroma.DataloaderCatalog.create"
    )
    def test_indexing_run_no_documents(
        self, mock_get_docs, mock_sparse_embedder_cls, mock_embedder_helper, mock_db
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
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_cost_optimized_rag",
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
    def test_indexing_with_custom_chunking(
        self,
        mock_get_docs,
        mock_sparse_embedder_cls,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
        sample_langchain_documents,
    ):
        """Test indexing with custom chunking configuration."""
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_langchain_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader
        mock_embed_docs.return_value = (sample_langchain_documents, [[0.1] * 384] * 5)

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder.embed_documents.return_value = [
            [0.5, 0.3, 0.2] for _ in range(5)
        ]
        mock_sparse_embedder_cls.return_value = mock_sparse_embedder

        mock_db_inst = MagicMock()
        mock_db_inst.upsert.return_value = len(sample_langchain_documents)
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_cost_optimized_rag",
            },
            "chunking": {
                "chunk_size": 500,
                "chunk_overlap": 100,
                "separators": ["\n\n", "\n", ".", " "],
            },
        }

        pipeline = ChromaCostOptimizedRAGIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_langchain_documents)

    @patch("vectordb.langchain.cost_optimized_rag.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.cost_optimized_rag.indexing.chroma.SparseEmbedder")
    def test_use_text_splitter_false(
        self, mock_sparse_cls, mock_embedder_helper, mock_db
    ):
        """Test initialization with use_text_splitter=False."""
        config = {
            "dataloader": {
                "type": "arc",
                "limit": 10,
                "use_text_splitter": False,
            },
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_cost_optimized_rag",
            },
        }

        pipeline = ChromaCostOptimizedRAGIndexingPipeline(config)
        assert pipeline.use_text_splitter is False
        assert pipeline.text_splitter is None

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
    def test_run_with_use_text_splitter_false(
        self,
        mock_get_docs,
        mock_sparse_cls,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
        sample_langchain_documents,
    ):
        """Test run() with use_text_splitter=False (no splitting)."""
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_langchain_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader
        mock_embed_docs.return_value = (
            sample_langchain_documents,
            [[0.1] * 384 for _ in range(len(sample_langchain_documents))],
        )

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder.embed_documents.return_value = [
            [0.5, 0.3, 0.2] for _ in range(len(sample_langchain_documents))
        ]
        mock_sparse_cls.return_value = mock_sparse_embedder

        mock_db_inst = MagicMock()
        mock_db_inst.upsert.return_value = len(sample_langchain_documents)
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {
                "type": "arc",
                "limit": 10,
                "use_text_splitter": False,
            },
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_cost_optimized_rag",
            },
        }

        pipeline = ChromaCostOptimizedRAGIndexingPipeline(config)
        result = pipeline.run()

        # Documents should be used as-is without splitting
        assert result["documents_indexed"] == len(sample_langchain_documents)
        assert result["chunks_created"] == len(sample_langchain_documents)

    @patch("vectordb.langchain.cost_optimized_rag.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.cost_optimized_rag.indexing.chroma.SparseEmbedder")
    def test_use_text_splitter_default_true(
        self, mock_sparse_cls, mock_embedder_helper, mock_db
    ):
        """Test that use_text_splitter defaults to True when not specified."""
        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_cost_optimized_rag",
            },
            "chunking": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
            },
        }

        pipeline = ChromaCostOptimizedRAGIndexingPipeline(config)
        assert pipeline.use_text_splitter is True
        assert pipeline.text_splitter is not None
