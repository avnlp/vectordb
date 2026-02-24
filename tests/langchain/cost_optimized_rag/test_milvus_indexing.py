"""Tests for Milvus cost-optimized RAG indexing pipeline (LangChain)."""

from unittest.mock import MagicMock, patch

from vectordb.langchain.cost_optimized_rag.indexing.milvus import (
    MilvusCostOptimizedRAGIndexingPipeline,
)


class TestMilvusCostOptimizedRAGIndexingPipeline:
    """Tests for MilvusCostOptimizedRAGIndexingPipeline."""

    @patch("vectordb.langchain.cost_optimized_rag.indexing.milvus.MilvusVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.milvus.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.cost_optimized_rag.indexing.milvus.SparseEmbedder")
    def test_init_with_valid_config(
        self, mock_sparse_cls, mock_embedder_helper, mock_db_cls, milvus_config
    ):
        """Test initialization with valid config dict."""
        mock_embedder = MagicMock()
        mock_embedder_helper.return_value = mock_embedder

        mock_sparse_embedder = MagicMock()
        mock_sparse_cls.return_value = mock_sparse_embedder

        mock_db_instance = MagicMock()
        mock_db_cls.return_value = mock_db_instance

        pipeline = MilvusCostOptimizedRAGIndexingPipeline(milvus_config)
        assert pipeline is not None
        assert pipeline.collection_name == "test_cost_optimized_rag"

    @patch("vectordb.langchain.cost_optimized_rag.indexing.milvus.MilvusVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.milvus.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.cost_optimized_rag.indexing.milvus.SparseEmbedder")
    def test_init_with_config_path(
        self, mock_sparse_cls, mock_embedder_helper, mock_db_cls, tmp_path
    ):
        """Test initialization with config file path."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
milvus:
  uri: http://localhost:19530
  db_name: default
  collection_name: test_cost_optimized_rag
  dimension: 384
embeddings:
  model: sentence-transformers/all-MiniLM-L6-v2
  device: cpu
dataloader:
  type: arc
  split: test
  limit: 10
chunking:
  chunk_size: 1000
  chunk_overlap: 200
            """
        )

        mock_embedder = MagicMock()
        mock_embedder_helper.return_value = mock_embedder

        mock_sparse_embedder = MagicMock()
        mock_sparse_cls.return_value = mock_sparse_embedder

        mock_db_instance = MagicMock()
        mock_db_cls.return_value = mock_db_instance

        pipeline = MilvusCostOptimizedRAGIndexingPipeline(str(config_file))
        assert pipeline is not None

    class TestRun:
        """Tests for run method."""

        @patch("vectordb.langchain.cost_optimized_rag.indexing.milvus.MilvusVectorDB")
        @patch(
            "vectordb.langchain.cost_optimized_rag.indexing.milvus.EmbedderHelper.create_embedder"
        )
        @patch(
            "vectordb.langchain.cost_optimized_rag.indexing.milvus.EmbedderHelper.embed_documents"
        )
        @patch("vectordb.langchain.cost_optimized_rag.indexing.milvus.SparseEmbedder")
        @patch(
            "vectordb.langchain.cost_optimized_rag.indexing.milvus.DataloaderCatalog.create"
        )
        def test_run_with_documents(
            self,
            mock_get_docs,
            mock_sparse_cls,
            mock_embed_docs,
            mock_embedder_helper,
            mock_db_cls,
            sample_documents,
            milvus_config,
        ):
            """Test run() method with documents."""
            mock_dataset = MagicMock()
            mock_dataset.to_langchain.return_value = sample_documents
            mock_loader = MagicMock()
            mock_loader.load.return_value = mock_dataset
            mock_get_docs.return_value = mock_loader
            mock_embed_docs.return_value = (
                sample_documents,
                [[0.1] * 384 for _ in range(len(sample_documents))],
            )

            mock_sparse_embedder = MagicMock()
            mock_sparse_embedder.embed_documents.return_value = [
                {"indices": [0, 1, 2], "values": [0.5, 0.3, 0.2]}
                for _ in range(len(sample_documents))
            ]
            mock_sparse_cls.return_value = mock_sparse_embedder

            mock_db_instance = MagicMock()
            mock_db_instance.upsert.return_value = len(sample_documents)
            mock_db_cls.return_value = mock_db_instance

            pipeline = MilvusCostOptimizedRAGIndexingPipeline(milvus_config)
            result = pipeline.run()

            assert result["documents_indexed"] == len(sample_documents)
            assert "chunks_created" in result

        @patch("vectordb.langchain.cost_optimized_rag.indexing.milvus.MilvusVectorDB")
        @patch(
            "vectordb.langchain.cost_optimized_rag.indexing.milvus.EmbedderHelper.create_embedder"
        )
        @patch("vectordb.langchain.cost_optimized_rag.indexing.milvus.SparseEmbedder")
        @patch(
            "vectordb.langchain.cost_optimized_rag.indexing.milvus.DataloaderCatalog.create"
        )
        def test_run_with_empty_documents(
            self,
            mock_get_docs,
            mock_sparse_cls,
            mock_embedder_helper,
            mock_db_cls,
            milvus_config,
        ):
            """Test run() method with empty documents (edge case)."""
            mock_dataset = MagicMock()
            mock_dataset.to_langchain.return_value = []
            mock_loader = MagicMock()
            mock_loader.load.return_value = mock_dataset
            mock_get_docs.return_value = mock_loader

            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_sparse_embedder = MagicMock()
            mock_sparse_cls.return_value = mock_sparse_embedder

            mock_db_instance = MagicMock()
            mock_db_cls.return_value = mock_db_instance

            pipeline = MilvusCostOptimizedRAGIndexingPipeline(milvus_config)
            result = pipeline.run()

            assert result["documents_indexed"] == 0
            assert result["chunks_created"] == 0

        @patch("vectordb.langchain.cost_optimized_rag.indexing.milvus.MilvusVectorDB")
        @patch(
            "vectordb.langchain.cost_optimized_rag.indexing.milvus.EmbedderHelper.create_embedder"
        )
        @patch(
            "vectordb.langchain.cost_optimized_rag.indexing.milvus.EmbedderHelper.embed_documents"
        )
        @patch("vectordb.langchain.cost_optimized_rag.indexing.milvus.SparseEmbedder")
        @patch(
            "vectordb.langchain.cost_optimized_rag.indexing.milvus.DataloaderCatalog.create"
        )
        @patch(
            "langchain_text_splitters.RecursiveCharacterTextSplitter.split_documents"
        )
        def test_run_with_no_chunks_created(
            self,
            mock_split,
            mock_get_docs,
            mock_sparse_cls,
            mock_embed_docs,
            mock_embedder_helper,
            mock_db_cls,
            sample_documents,
            milvus_config,
        ):
            """Test run() method with no chunks created (edge case)."""
            mock_dataset = MagicMock()
            mock_dataset.to_langchain.return_value = sample_documents
            mock_loader = MagicMock()
            mock_loader.load.return_value = mock_dataset
            mock_get_docs.return_value = mock_loader
            # Return empty chunks from text splitter
            mock_split.return_value = []

            mock_sparse_embedder = MagicMock()
            mock_sparse_embedder.embed_documents.return_value = []
            mock_sparse_cls.return_value = mock_sparse_embedder

            mock_db_instance = MagicMock()
            mock_db_cls.return_value = mock_db_instance

            pipeline = MilvusCostOptimizedRAGIndexingPipeline(milvus_config)
            result = pipeline.run()

            assert result["documents_indexed"] == 0
            assert result["chunks_created"] == 0

    class TestChunkingParameters:
        """Tests for chunking parameters validation."""

        @patch("vectordb.langchain.cost_optimized_rag.indexing.milvus.MilvusVectorDB")
        @patch(
            "vectordb.langchain.cost_optimized_rag.indexing.milvus.EmbedderHelper.create_embedder"
        )
        @patch("vectordb.langchain.cost_optimized_rag.indexing.milvus.SparseEmbedder")
        def test_custom_chunking_params(
            self, mock_sparse_cls, mock_embedder_helper, mock_db_cls
        ):
            """Test initialization with custom chunking parameters."""
            config = {
                "dataloader": {"type": "arc", "limit": 10},
                "embeddings": {"model": "test-model", "device": "cpu"},
                "milvus": {
                    "uri": "http://localhost:19530",
                    "db_name": "default",
                    "collection_name": "test_cost_optimized_rag",
                    "dimension": 384,
                },
                "chunking": {
                    "chunk_size": 500,
                    "chunk_overlap": 100,
                    "separators": ["\n\n", "\n", ".", " "],
                },
            }

            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_sparse_embedder = MagicMock()
            mock_sparse_cls.return_value = mock_sparse_embedder

            mock_db_instance = MagicMock()
            mock_db_cls.return_value = mock_db_instance

            pipeline = MilvusCostOptimizedRAGIndexingPipeline(config)
            assert pipeline is not None
            assert pipeline.text_splitter._chunk_size == 500
            assert pipeline.text_splitter._chunk_overlap == 100

        @patch("vectordb.langchain.cost_optimized_rag.indexing.milvus.MilvusVectorDB")
        @patch(
            "vectordb.langchain.cost_optimized_rag.indexing.milvus.EmbedderHelper.create_embedder"
        )
        @patch("vectordb.langchain.cost_optimized_rag.indexing.milvus.SparseEmbedder")
        def test_default_chunking_params(
            self, mock_sparse_cls, mock_embedder_helper, mock_db_cls, milvus_config
        ):
            """Test initialization with default chunking parameters."""
            mock_embedder = MagicMock()
            mock_embedder_helper.return_value = mock_embedder

            mock_sparse_embedder = MagicMock()
            mock_sparse_cls.return_value = mock_sparse_embedder

            mock_db_instance = MagicMock()
            mock_db_cls.return_value = mock_db_instance

            pipeline = MilvusCostOptimizedRAGIndexingPipeline(milvus_config)
            assert pipeline is not None
            assert pipeline.text_splitter._chunk_size == 1000
            assert pipeline.text_splitter._chunk_overlap == 200
