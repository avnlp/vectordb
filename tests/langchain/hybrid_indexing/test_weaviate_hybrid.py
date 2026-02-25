"""Tests for Weaviate hybrid indexing and search pipelines (LangChain)."""

from unittest.mock import MagicMock, patch

from haystack.dataclasses import Document as HaystackDocument
from langchain_core.documents import Document


class TestWeaviateHybridIndexing:
    """Unit tests for Weaviate hybrid indexing pipeline."""

    @patch("vectordb.langchain.hybrid_indexing.indexing.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.weaviate.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.hybrid_indexing.indexing.weaviate.SparseEmbedder")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.weaviate.DataloaderCatalog.create"
    )
    def test_indexing_initialization(
        self, mock_get_docs, mock_sparse_embedder, mock_embedder_helper, mock_db
    ):
        """Test pipeline initialization."""
        from vectordb.langchain.hybrid_indexing.indexing.weaviate import (
            WeaviateHybridIndexingPipeline,
        )

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "test-api-key",
                "collection_name": "test_hybrid",
                "dimension": 384,
            },
        }

        pipeline = WeaviateHybridIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.collection_name == "test_hybrid"
        assert pipeline.dimension == 384
        assert pipeline.dense_embedder is not None
        assert pipeline.sparse_embedder is not None

    @patch("vectordb.langchain.hybrid_indexing.indexing.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.weaviate.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.hybrid_indexing.indexing.weaviate.SparseEmbedder")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.weaviate.DataloaderCatalog.create"
    )
    def test_indexing_run_with_documents(
        self,
        mock_get_docs,
        mock_sparse_embedder_class,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
    ):
        """Test indexing with documents including sparse embeddings for reference."""
        from vectordb.langchain.hybrid_indexing.indexing.weaviate import (
            WeaviateHybridIndexingPipeline,
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

        # Mock sparse embedder (used for reference in Weaviate)
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
            "weaviate": {
                "url": "http://localhost:8080",
                "collection_name": "test_hybrid",
                "dimension": 384,
            },
        }

        pipeline = WeaviateHybridIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        assert result["db"] == "weaviate"
        assert result["collection_name"] == "test_hybrid"
        mock_db_inst.create_collection.assert_called_once()
        mock_db_inst.upsert.assert_called_once()

        # Verify sparse embeddings were generated (for reference)
        mock_sparse_embedder.embed_documents.assert_called_once()

    @patch("vectordb.langchain.hybrid_indexing.indexing.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.weaviate.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.hybrid_indexing.indexing.weaviate.SparseEmbedder")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.weaviate.DataloaderCatalog.create"
    )
    def test_indexing_run_no_documents(
        self, mock_get_docs, mock_sparse_embedder, mock_embedder_helper, mock_db
    ):
        """Test indexing with no documents."""
        from vectordb.langchain.hybrid_indexing.indexing.weaviate import (
            WeaviateHybridIndexingPipeline,
        )

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
                "collection_name": "test_hybrid",
            },
        }

        pipeline = WeaviateHybridIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0
        assert result["db"] == "weaviate"

    @patch("vectordb.langchain.hybrid_indexing.indexing.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.weaviate.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.hybrid_indexing.indexing.weaviate.SparseEmbedder")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.weaviate.DataloaderCatalog.create"
    )
    def test_upsert_data_structure(
        self,
        mock_get_docs,
        mock_sparse_embedder_class,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
    ):
        """Test that upsert data includes dense embeddings and metadata."""
        from vectordb.langchain.hybrid_indexing.indexing.weaviate import (
            WeaviateHybridIndexingPipeline,
        )

        sample_documents = [
            Document(
                page_content="Test document",
                metadata={"source": "test"},
            ),
        ]

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
            "weaviate": {
                "url": "http://localhost:8080",
                "collection_name": "test_hybrid",
            },
        }

        pipeline = WeaviateHybridIndexingPipeline(config)
        pipeline.run()

        # Check that upsert was called with correct data structure
        call_args = mock_db_inst.upsert.call_args
        upsert_data = call_args.kwargs.get(
            "data", call_args.args[0] if call_args.args else []
        )
        if upsert_data:
            assert "id" in upsert_data[0]
            assert "values" in upsert_data[0]
            assert "metadata" in upsert_data[0]
            # Weaviate uses BM25 natively, so no sparse_values in upsert


class TestWeaviateHybridSearch:
    """Unit tests for Weaviate hybrid search pipeline."""

    @patch("vectordb.langchain.hybrid_indexing.search.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.search.weaviate.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.hybrid_indexing.search.weaviate.SparseEmbedder")
    @patch("vectordb.langchain.hybrid_indexing.search.weaviate.RAGHelper.create_llm")
    def test_search_initialization(
        self, mock_llm_helper, mock_sparse_embedder, mock_embedder_helper, mock_db
    ):
        """Test search pipeline initialization."""
        from vectordb.langchain.hybrid_indexing.search.weaviate import (
            WeaviateHybridSearchPipeline,
        )

        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "collection_name": "test_hybrid",
                "alpha": 0.65,
            },
        }

        pipeline = WeaviateHybridSearchPipeline(config)
        assert pipeline.collection_name == "test_hybrid"
        assert pipeline.alpha == 0.65
        assert pipeline.llm is None

    @patch("vectordb.langchain.hybrid_indexing.search.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.search.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.hybrid_indexing.search.weaviate.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.hybrid_indexing.search.weaviate.SparseEmbedder")
    @patch("vectordb.langchain.hybrid_indexing.search.weaviate.RAGHelper.create_llm")
    def test_hybrid_search(
        self,
        mock_llm_helper,
        mock_sparse_embedder_class,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test hybrid search combining BM25 keyword and vector similarity."""
        from vectordb.langchain.hybrid_indexing.search.weaviate import (
            WeaviateHybridSearchPipeline,
        )

        sample_documents = [
            HaystackDocument(
                content="Python is a high-level programming language",
                meta={"source": "wiki"},
                id="1",
            ),
        ]

        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.hybrid_search.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        # Mock sparse embedder (for reference)
        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "collection_name": "test_hybrid",
                "alpha": 0.5,
            },
        }

        pipeline = WeaviateHybridSearchPipeline(config)
        result = pipeline.search("test query", top_k=10)

        assert result["query"] == "test query"
        assert len(result["documents"]) == 1
        assert "answer" not in result

        # Verify hybrid_search was called with query and embedding
        mock_db_inst.hybrid_search.assert_called_once()
        call_kwargs = mock_db_inst.hybrid_search.call_args.kwargs
        assert "vector" in call_kwargs
        assert "query" in call_kwargs
        assert call_kwargs["alpha"] == 0.5

    @patch("vectordb.langchain.hybrid_indexing.search.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.search.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.hybrid_indexing.search.weaviate.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.hybrid_indexing.search.weaviate.SparseEmbedder")
    @patch("vectordb.langchain.hybrid_indexing.search.weaviate.RAGHelper.create_llm")
    @patch("vectordb.langchain.hybrid_indexing.search.weaviate.RAGHelper.generate")
    def test_search_with_rag(
        self,
        mock_rag_generate,
        mock_llm_helper,
        mock_sparse_embedder_class,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test search with RAG generation."""
        from vectordb.langchain.hybrid_indexing.search.weaviate import (
            WeaviateHybridSearchPipeline,
        )

        sample_documents = [
            HaystackDocument(
                content="Python is a high-level programming language",
                meta={"source": "wiki"},
                id="1",
            ),
        ]

        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.hybrid_search.return_value = sample_documents
        mock_db.return_value = mock_db_inst

        mock_llm_inst = MagicMock()
        mock_llm_helper.return_value = mock_llm_inst
        mock_rag_generate.return_value = "Test answer"

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "collection_name": "test_hybrid",
            },
            "rag": {"enabled": True},
        }

        pipeline = WeaviateHybridSearchPipeline(config)
        result = pipeline.search("test query", top_k=10)

        assert result["query"] == "test query"
        assert result["answer"] == "Test answer"

    @patch("vectordb.langchain.hybrid_indexing.search.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.search.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.hybrid_indexing.search.weaviate.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.hybrid_indexing.search.weaviate.SparseEmbedder")
    @patch("vectordb.langchain.hybrid_indexing.search.weaviate.RAGHelper.create_llm")
    def test_search_with_filters(
        self,
        mock_llm_helper,
        mock_sparse_embedder_class,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test search with metadata filters."""
        from vectordb.langchain.hybrid_indexing.search.weaviate import (
            WeaviateHybridSearchPipeline,
        )

        sample_documents = [
            HaystackDocument(
                content="Python is a high-level programming language",
                meta={"source": "wiki"},
                id="1",
            ),
        ]

        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.hybrid_search.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "collection_name": "test_hybrid",
            },
        }

        pipeline = WeaviateHybridSearchPipeline(config)
        filters = {"source": "wiki"}
        result = pipeline.search("test query", top_k=10, filters=filters)

        assert result["query"] == "test query"
        mock_db_inst.hybrid_search.assert_called_once()
        call_kwargs = mock_db_inst.hybrid_search.call_args.kwargs
        assert call_kwargs["filters"] == filters

    @patch("vectordb.langchain.hybrid_indexing.search.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.search.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.hybrid_indexing.search.weaviate.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.hybrid_indexing.search.weaviate.SparseEmbedder")
    @patch("vectordb.langchain.hybrid_indexing.search.weaviate.RAGHelper.create_llm")
    def test_search_alpha_parameter(
        self,
        mock_llm_helper,
        mock_sparse_embedder_class,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test that alpha parameter controls fusion weight."""
        from vectordb.langchain.hybrid_indexing.search.weaviate import (
            WeaviateHybridSearchPipeline,
        )

        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.hybrid_search.return_value = []
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "collection_name": "test_hybrid",
                "alpha": 0.4,
            },
        }

        pipeline = WeaviateHybridSearchPipeline(config)
        pipeline.search("test query", top_k=10)

        call_kwargs = mock_db_inst.hybrid_search.call_args.kwargs
        assert call_kwargs["alpha"] == 0.4

    @patch("vectordb.langchain.hybrid_indexing.search.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.search.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.hybrid_indexing.search.weaviate.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.hybrid_indexing.search.weaviate.SparseEmbedder")
    @patch("vectordb.langchain.hybrid_indexing.search.weaviate.RAGHelper.create_llm")
    def test_weaviate_hybrid_uses_bm25(
        self,
        mock_llm_helper,
        mock_sparse_embedder_class,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test that Weaviate hybrid search uses BM25 + vector search."""
        from vectordb.langchain.hybrid_indexing.search.weaviate import (
            WeaviateHybridSearchPipeline,
        )

        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.hybrid_search.return_value = []
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "collection_name": "test_hybrid",
            },
        }

        pipeline = WeaviateHybridSearchPipeline(config)
        pipeline.search("machine learning algorithms", top_k=10)

        # Verify hybrid_search was called with query for BM25
        call_kwargs = mock_db_inst.hybrid_search.call_args.kwargs
        assert "query" in call_kwargs
        assert call_kwargs["query"] == "machine learning algorithms"
