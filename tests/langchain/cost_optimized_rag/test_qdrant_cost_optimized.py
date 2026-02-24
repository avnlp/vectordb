"""Tests for Qdrant cost-optimized RAG pipelines (LangChain)."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


class TestQdrantCostOptimizedIndexing:
    """Unit tests for Qdrant cost-optimized RAG indexing pipeline."""

    @patch("vectordb.langchain.cost_optimized_rag.indexing.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.qdrant.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.cost_optimized_rag.indexing.qdrant.SparseEmbedder")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.qdrant.DataloaderCatalog.create"
    )
    def test_indexing_initialization(
        self, mock_get_docs, mock_sparse_embedder, mock_embedder_helper, mock_db
    ):
        """Test pipeline initialization."""
        from vectordb.langchain.cost_optimized_rag.indexing.qdrant import (
            QdrantCostOptimizedRAGIndexingPipeline,
        )

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "collection_name": "test_cost_optimized",
                "dimension": 384,
            },
        }

        pipeline = QdrantCostOptimizedRAGIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.collection_name == "test_cost_optimized"
        assert pipeline.dimension == 384

    @patch("vectordb.langchain.cost_optimized_rag.indexing.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.qdrant.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.cost_optimized_rag.indexing.qdrant.SparseEmbedder")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.qdrant.DataloaderCatalog.create"
    )
    def test_indexing_run_with_documents(
        self,
        mock_get_docs,
        mock_sparse_embedder_class,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
    ):
        """Test indexing with documents."""
        from vectordb.langchain.cost_optimized_rag.indexing.qdrant import (
            QdrantCostOptimizedRAGIndexingPipeline,
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

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder.embed_documents.return_value = [
            {"indices": [1, 2], "values": [0.5, 0.5]}
        ] * 2
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        mock_db_inst = MagicMock()
        mock_db_inst.upsert.return_value = len(sample_documents)
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "collection_name": "test_cost_optimized",
                "dimension": 384,
            },
        }

        pipeline = QdrantCostOptimizedRAGIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.create_collection.assert_called_once()
        mock_db_inst.upsert.assert_called_once()

    @patch("vectordb.langchain.cost_optimized_rag.indexing.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.qdrant.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.cost_optimized_rag.indexing.qdrant.SparseEmbedder")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.qdrant.DataloaderCatalog.create"
    )
    def test_indexing_run_no_documents(
        self, mock_get_docs, mock_sparse_embedder, mock_embedder_helper, mock_db
    ):
        """Test indexing with no documents."""
        from vectordb.langchain.cost_optimized_rag.indexing.qdrant import (
            QdrantCostOptimizedRAGIndexingPipeline,
        )

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
                "collection_name": "test_cost_optimized",
            },
        }

        pipeline = QdrantCostOptimizedRAGIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0
        assert result["chunks_created"] == 0

    @patch("vectordb.langchain.cost_optimized_rag.indexing.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.qdrant.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.cost_optimized_rag.indexing.qdrant.SparseEmbedder")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.qdrant.DataloaderCatalog.create"
    )
    def test_indexing_with_api_key(
        self,
        mock_get_docs,
        mock_sparse_embedder_class,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
    ):
        """Test indexing with API key authentication."""
        from vectordb.langchain.cost_optimized_rag.indexing.qdrant import (
            QdrantCostOptimizedRAGIndexingPipeline,
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

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder.embed_documents.return_value = [
            {"indices": [1], "values": [1.0]}
        ]
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        mock_db_inst = MagicMock()
        mock_db_inst.upsert.return_value = 1
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": "test-api-key",
                "collection_name": "test_cost_optimized",
                "dimension": 384,
            },
        }

        pipeline = QdrantCostOptimizedRAGIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 1

    @patch("vectordb.langchain.cost_optimized_rag.indexing.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.qdrant.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.cost_optimized_rag.indexing.qdrant.SparseEmbedder")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.qdrant.DataloaderCatalog.create"
    )
    def test_indexing_stores_sparse_in_payload(
        self,
        mock_get_docs,
        mock_sparse_embedder_class,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
    ):
        """Test that indexing stores sparse vectors in payload."""
        from vectordb.langchain.cost_optimized_rag.indexing.qdrant import (
            QdrantCostOptimizedRAGIndexingPipeline,
        )

        sample_documents = [
            Document(
                page_content="Test document for hybrid search",
                metadata={"source": "test"},
            ),
        ]

        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader
        mock_embed_docs.return_value = (sample_documents, [[0.1] * 384])

        mock_sparse_embedder = MagicMock()
        sparse_vector = {"indices": [1, 2, 3], "values": [0.3, 0.4, 0.3]}
        mock_sparse_embedder.embed_documents.return_value = [sparse_vector]
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        mock_db_inst = MagicMock()
        mock_db_inst.upsert.return_value = 1
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "collection_name": "test_cost_optimized",
                "dimension": 384,
            },
        }

        pipeline = QdrantCostOptimizedRAGIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 1
        assert result["chunks_created"] == 1
        mock_db_inst.upsert.assert_called_once()


class TestQdrantCostOptimizedSearch:
    """Unit tests for Qdrant cost-optimized RAG search pipeline."""

    @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.qdrant.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.SparseEmbedder")
    @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.RAGHelper.create_llm")
    def test_search_initialization(
        self, mock_llm_helper, mock_sparse_embedder, mock_embedder_helper, mock_db
    ):
        """Test search pipeline initialization."""
        from vectordb.langchain.cost_optimized_rag.search.qdrant import (
            QdrantCostOptimizedRAGSearchPipeline,
        )

        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "collection_name": "test_cost_optimized",
            },
            "rag": {"enabled": False},
        }

        pipeline = QdrantCostOptimizedRAGSearchPipeline(config)
        assert pipeline.config == config
        assert pipeline.llm is None
        assert pipeline.rrf_k == 60

    @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.qdrant.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.SparseEmbedder")
    @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.RAGHelper.create_llm")
    def test_search_hybrid_execution(
        self,
        mock_llm_helper,
        mock_sparse_embedder_class,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test hybrid search execution using native Qdrant hybrid search."""
        from vectordb.langchain.cost_optimized_rag.search.qdrant import (
            QdrantCostOptimizedRAGSearchPipeline,
        )

        hybrid_docs = [
            Document(page_content="Hybrid result 1", metadata={"id": "1"}),
            Document(page_content="Hybrid result 2", metadata={"id": "2"}),
            Document(page_content="Hybrid result 3", metadata={"id": "3"}),
        ]

        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.search.return_value = hybrid_docs
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder.embed_query.return_value = {
            "indices": [1],
            "values": [1.0],
        }
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "collection_name": "test_cost_optimized",
            },
            "rag": {"enabled": False},
        }

        pipeline = QdrantCostOptimizedRAGSearchPipeline(config)
        result = pipeline.search("What is Python?", top_k=3)

        assert result["query"] == "What is Python?"
        mock_db_inst.search.assert_called_once()
        call_kwargs = mock_db_inst.search.call_args[1]
        assert call_kwargs["search_type"] == "hybrid"
        assert "dense" in call_kwargs["query_vector"]
        assert "sparse" in call_kwargs["query_vector"]

    @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.qdrant.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.SparseEmbedder")
    @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.RAGHelper.create_llm")
    @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.RAGHelper.generate")
    def test_search_with_rag_generation(
        self,
        mock_rag_generate,
        mock_llm_helper,
        mock_sparse_embedder_class,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test search with RAG answer generation."""
        from vectordb.langchain.cost_optimized_rag.search.qdrant import (
            QdrantCostOptimizedRAGSearchPipeline,
        )

        hybrid_docs = [
            Document(
                page_content="Python is a programming language", metadata={"id": "1"}
            ),
        ]

        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.search.return_value = hybrid_docs
        mock_db.return_value = mock_db_inst

        mock_llm = MagicMock()
        mock_llm_helper.return_value = mock_llm
        mock_rag_generate.return_value = "Python is a popular programming language."

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder.embed_query.return_value = {
            "indices": [1],
            "values": [1.0],
        }
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "collection_name": "test_cost_optimized",
            },
            "rag": {"enabled": True, "model": "gpt-3.5-turbo"},
        }

        pipeline = QdrantCostOptimizedRAGSearchPipeline(config)
        result = pipeline.search("What is Python?", top_k=1)

        assert "answer" in result
        assert result["answer"] == "Python is a popular programming language."
        mock_rag_generate.assert_called_once()
        mock_db_inst.search.assert_called_once()

    @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.qdrant.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.qdrant.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.SparseEmbedder")
    @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.RAGHelper.create_llm")
    def test_search_with_filters(
        self,
        mock_llm_helper,
        mock_sparse_embedder_class,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test search with metadata filters."""
        from vectordb.langchain.cost_optimized_rag.search.qdrant import (
            QdrantCostOptimizedRAGSearchPipeline,
        )

        hybrid_docs = [
            Document(
                page_content="Python programming", metadata={"category": "programming"}
            ),
        ]

        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.search.return_value = hybrid_docs
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder.embed_query.return_value = {
            "indices": [1],
            "values": [1.0],
        }
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "collection_name": "test_cost_optimized",
            },
            "rag": {"enabled": False},
        }

        pipeline = QdrantCostOptimizedRAGSearchPipeline(config)
        filters = {"category": "programming"}
        result = pipeline.search("Python", top_k=1, filters=filters)

        assert len(result["documents"]) == 1
        mock_db_inst.search.assert_called_once()
        call_kwargs = mock_db_inst.search.call_args[1]
        assert call_kwargs["filters"] == filters

    @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.QdrantVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.qdrant.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.SparseEmbedder")
    @patch("vectordb.langchain.cost_optimized_rag.search.qdrant.RAGHelper.create_llm")
    def test_search_custom_rrf_k(
        self, mock_llm_helper, mock_sparse_embedder, mock_embedder_helper, mock_db
    ):
        """Test search with custom RRF k parameter."""
        from vectordb.langchain.cost_optimized_rag.search.qdrant import (
            QdrantCostOptimizedRAGSearchPipeline,
        )

        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "qdrant": {
                "url": "http://localhost:6333",
                "collection_name": "test_cost_optimized",
            },
            "search": {
                "rrf_k": 120,
            },
            "rag": {"enabled": False},
        }

        pipeline = QdrantCostOptimizedRAGSearchPipeline(config)
        assert pipeline.rrf_k == 120
