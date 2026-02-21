"""Tests for Pinecone hybrid indexing and search pipelines (LangChain)."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


class TestPineconeHybridIndexing:
    """Unit tests for Pinecone hybrid indexing pipeline."""

    @patch("vectordb.langchain.hybrid_indexing.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.pinecone.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.hybrid_indexing.indexing.pinecone.SparseEmbedder")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.pinecone.DataloaderCatalog.create"
    )
    def test_indexing_initialization(
        self, mock_get_docs, mock_sparse_embedder, mock_embedder_helper, mock_db
    ):
        """Test pipeline initialization."""
        from vectordb.langchain.hybrid_indexing.indexing.pinecone import (
            PineconeHybridIndexingPipeline,
        )

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "test-hybrid",
                "namespace": "test-namespace",
                "dimension": 384,
            },
        }

        pipeline = PineconeHybridIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.index_name == "test-hybrid"
        assert pipeline.namespace == "test-namespace"
        assert pipeline.dimension == 384
        assert pipeline.dense_embedder is not None
        assert pipeline.sparse_embedder is not None

    @patch("vectordb.langchain.hybrid_indexing.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.pinecone.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.hybrid_indexing.indexing.pinecone.SparseEmbedder")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.pinecone.DataloaderCatalog.create"
    )
    def test_indexing_run_with_documents(
        self,
        mock_get_docs,
        mock_sparse_embedder_class,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
    ):
        """Test indexing with documents including sparse embeddings."""
        from vectordb.langchain.hybrid_indexing.indexing.pinecone import (
            PineconeHybridIndexingPipeline,
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
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "test-hybrid",
                "namespace": "test-namespace",
                "dimension": 384,
                "recreate": False,
                "metric": "cosine",
            },
        }

        pipeline = PineconeHybridIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        assert result["db"] == "pinecone"
        assert result["index_name"] == "test-hybrid"
        mock_db_inst.create_index.assert_called_once()
        mock_db_inst.upsert.assert_called_once()

        # Verify sparse embeddings were generated
        mock_sparse_embedder.embed_documents.assert_called_once()

    @patch("vectordb.langchain.hybrid_indexing.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.pinecone.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.hybrid_indexing.indexing.pinecone.SparseEmbedder")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.pinecone.DataloaderCatalog.create"
    )
    def test_indexing_run_no_documents(
        self, mock_get_docs, mock_sparse_embedder, mock_embedder_helper, mock_db
    ):
        """Test indexing with no documents."""
        from vectordb.langchain.hybrid_indexing.indexing.pinecone import (
            PineconeHybridIndexingPipeline,
        )

        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = []
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "test-hybrid",
                "namespace": "test-namespace",
            },
        }

        pipeline = PineconeHybridIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0
        assert result["db"] == "pinecone"

    @patch("vectordb.langchain.hybrid_indexing.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.pinecone.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.hybrid_indexing.indexing.pinecone.SparseEmbedder")
    @patch(
        "vectordb.langchain.hybrid_indexing.indexing.pinecone.DataloaderCatalog.create"
    )
    def test_hybrid_upsert_data_structure(
        self,
        mock_get_docs,
        mock_sparse_embedder_class,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
    ):
        """Test that upsert data includes both dense and sparse embeddings."""
        from vectordb.langchain.hybrid_indexing.indexing.pinecone import (
            PineconeHybridIndexingPipeline,
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
        sparse_embedding = {"indices": [1, 2], "values": [0.5, 0.3]}
        mock_sparse_embedder.embed_documents.return_value = [sparse_embedding]
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        mock_db_inst = MagicMock()
        mock_db_inst.upsert.return_value = 1
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "test-hybrid",
                "namespace": "test-namespace",
            },
        }

        pipeline = PineconeHybridIndexingPipeline(config)
        pipeline.run()

        # Check that upsert was called with correct data structure
        call_args = mock_db_inst.upsert.call_args
        upsert_data = call_args.kwargs.get(
            "data", call_args.args[0] if call_args.args else []
        )
        if upsert_data:
            assert "id" in upsert_data[0]
            assert "values" in upsert_data[0]
            assert "sparse_values" in upsert_data[0]
            assert "metadata" in upsert_data[0]
            assert upsert_data[0]["sparse_values"] == sparse_embedding


class TestPineconeHybridSearch:
    """Unit tests for Pinecone hybrid search pipeline."""

    @patch("vectordb.langchain.hybrid_indexing.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.hybrid_indexing.search.pinecone.SparseEmbedder")
    @patch("vectordb.langchain.hybrid_indexing.search.pinecone.RAGHelper.create_llm")
    def test_search_initialization(
        self, mock_llm_helper, mock_sparse_embedder, mock_embedder_helper, mock_db
    ):
        """Test search pipeline initialization."""
        from vectordb.langchain.hybrid_indexing.search.pinecone import (
            PineconeHybridSearchPipeline,
        )

        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "test-hybrid",
                "namespace": "test-namespace",
                "alpha": 0.7,
            },
        }

        pipeline = PineconeHybridSearchPipeline(config)
        assert pipeline.index_name == "test-hybrid"
        assert pipeline.namespace == "test-namespace"
        assert pipeline.alpha == 0.7
        assert pipeline.llm is None

    @patch("vectordb.langchain.hybrid_indexing.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.hybrid_indexing.search.pinecone.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.hybrid_indexing.search.pinecone.SparseEmbedder")
    @patch("vectordb.langchain.hybrid_indexing.search.pinecone.RAGHelper.create_llm")
    def test_hybrid_search(
        self,
        mock_llm_helper,
        mock_sparse_embedder_class,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test hybrid search with both dense and sparse embeddings."""
        from vectordb.langchain.hybrid_indexing.search.pinecone import (
            PineconeHybridSearchPipeline,
        )

        sample_documents = [
            Document(
                page_content="Python is a high-level programming language",
                metadata={"source": "wiki", "id": "1"},
            ),
        ]

        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.hybrid_search.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        # Mock sparse embedder
        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder.embed_query.return_value = {
            "indices": [1, 2],
            "values": [0.5, 0.3],
        }
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "test-hybrid",
                "namespace": "test-namespace",
                "alpha": 0.5,
            },
        }

        pipeline = PineconeHybridSearchPipeline(config)
        result = pipeline.search("test query", top_k=10)

        assert result["query"] == "test query"
        assert len(result["documents"]) == 1
        assert "answer" not in result

        # Verify hybrid_search was called with both embeddings
        mock_db_inst.hybrid_search.assert_called_once()
        call_kwargs = mock_db_inst.hybrid_search.call_args.kwargs
        assert "query_embedding" in call_kwargs
        assert "query_sparse_embedding" in call_kwargs
        assert call_kwargs["alpha"] == 0.5

    @patch("vectordb.langchain.hybrid_indexing.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.hybrid_indexing.search.pinecone.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.hybrid_indexing.search.pinecone.SparseEmbedder")
    @patch("vectordb.langchain.hybrid_indexing.search.pinecone.RAGHelper.create_llm")
    @patch("vectordb.langchain.hybrid_indexing.search.pinecone.RAGHelper.generate")
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
        from vectordb.langchain.hybrid_indexing.search.pinecone import (
            PineconeHybridSearchPipeline,
        )

        sample_documents = [
            Document(
                page_content="Python is a high-level programming language",
                metadata={"source": "wiki", "id": "1"},
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
        mock_sparse_embedder.embed_query.return_value = {
            "indices": [1],
            "values": [0.5],
        }
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "test-hybrid",
                "namespace": "test-namespace",
            },
            "rag": {"enabled": True},
        }

        pipeline = PineconeHybridSearchPipeline(config)
        result = pipeline.search("test query", top_k=10)

        assert result["query"] == "test query"
        assert result["answer"] == "Test answer"

    @patch("vectordb.langchain.hybrid_indexing.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.hybrid_indexing.search.pinecone.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.hybrid_indexing.search.pinecone.SparseEmbedder")
    @patch("vectordb.langchain.hybrid_indexing.search.pinecone.RAGHelper.create_llm")
    def test_search_with_filters(
        self,
        mock_llm_helper,
        mock_sparse_embedder_class,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test search with metadata filters."""
        from vectordb.langchain.hybrid_indexing.search.pinecone import (
            PineconeHybridSearchPipeline,
        )

        sample_documents = [
            Document(
                page_content="Python is a high-level programming language",
                metadata={"source": "wiki", "id": "1"},
            ),
        ]

        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.hybrid_search.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder.embed_query.return_value = {
            "indices": [1],
            "values": [0.5],
        }
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "test-hybrid",
                "namespace": "test-namespace",
            },
        }

        pipeline = PineconeHybridSearchPipeline(config)
        filters = {"source": "wiki"}
        result = pipeline.search("test query", top_k=10, filters=filters)

        assert result["query"] == "test query"
        mock_db_inst.hybrid_search.assert_called_once()
        call_kwargs = mock_db_inst.hybrid_search.call_args.kwargs
        assert call_kwargs["filters"] == filters

    @patch("vectordb.langchain.hybrid_indexing.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.hybrid_indexing.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.hybrid_indexing.search.pinecone.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.hybrid_indexing.search.pinecone.SparseEmbedder")
    @patch("vectordb.langchain.hybrid_indexing.search.pinecone.RAGHelper.create_llm")
    def test_search_alpha_parameter(
        self,
        mock_llm_helper,
        mock_sparse_embedder_class,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test that alpha parameter controls fusion weight."""
        from vectordb.langchain.hybrid_indexing.search.pinecone import (
            PineconeHybridSearchPipeline,
        )

        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.hybrid_search.return_value = []
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder.embed_query.return_value = {
            "indices": [1],
            "values": [0.5],
        }
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-api-key",
                "index_name": "test-hybrid",
                "namespace": "test-namespace",
                "alpha": 0.25,
            },
        }

        pipeline = PineconeHybridSearchPipeline(config)
        pipeline.search("test query", top_k=10)

        call_kwargs = mock_db_inst.hybrid_search.call_args.kwargs
        assert call_kwargs["alpha"] == 0.25
