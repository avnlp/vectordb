"""Tests for Weaviate cost-optimized RAG pipelines (LangChain)."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


class TestWeaviateCostOptimizedIndexing:
    """Unit tests for Weaviate cost-optimized RAG indexing pipeline."""

    @patch("vectordb.langchain.cost_optimized_rag.indexing.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.weaviate.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.cost_optimized_rag.indexing.weaviate.SparseEmbedder")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.weaviate.DataloaderCatalog.create"
    )
    def test_indexing_initialization(
        self, mock_get_docs, mock_sparse_embedder, mock_embedder_helper, mock_db
    ):
        """Test pipeline initialization."""
        from vectordb.langchain.cost_optimized_rag.indexing.weaviate import (
            WeaviateCostOptimizedRAGIndexingPipeline,
        )

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "collection_name": "test_cost_optimized",
                "dimension": 384,
            },
        }

        pipeline = WeaviateCostOptimizedRAGIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.collection_name == "test_cost_optimized"
        assert pipeline.dimension == 384

    @patch("vectordb.langchain.cost_optimized_rag.indexing.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.weaviate.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.cost_optimized_rag.indexing.weaviate.SparseEmbedder")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.weaviate.DataloaderCatalog.create"
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
        from vectordb.langchain.cost_optimized_rag.indexing.weaviate import (
            WeaviateCostOptimizedRAGIndexingPipeline,
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
            "weaviate": {
                "url": "http://localhost:8080",
                "collection_name": "test_cost_optimized",
                "dimension": 384,
            },
        }

        pipeline = WeaviateCostOptimizedRAGIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.create_collection.assert_called_once()
        mock_db_inst.upsert.assert_called_once()

    @patch("vectordb.langchain.cost_optimized_rag.indexing.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.weaviate.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.cost_optimized_rag.indexing.weaviate.SparseEmbedder")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.weaviate.DataloaderCatalog.create"
    )
    def test_indexing_run_no_documents(
        self, mock_get_docs, mock_sparse_embedder, mock_embedder_helper, mock_db
    ):
        """Test indexing with no documents."""
        from vectordb.langchain.cost_optimized_rag.indexing.weaviate import (
            WeaviateCostOptimizedRAGIndexingPipeline,
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
                "collection_name": "test_cost_optimized",
            },
        }

        pipeline = WeaviateCostOptimizedRAGIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0
        assert result["chunks_created"] == 0

    @patch("vectordb.langchain.cost_optimized_rag.indexing.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.weaviate.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.cost_optimized_rag.indexing.weaviate.SparseEmbedder")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.weaviate.DataloaderCatalog.create"
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
        from vectordb.langchain.cost_optimized_rag.indexing.weaviate import (
            WeaviateCostOptimizedRAGIndexingPipeline,
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
            "weaviate": {
                "url": "http://localhost:8080",
                "api_key": "test-api-key",
                "collection_name": "test_cost_optimized",
                "dimension": 384,
            },
        }

        pipeline = WeaviateCostOptimizedRAGIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 1

    @patch("vectordb.langchain.cost_optimized_rag.indexing.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.weaviate.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.cost_optimized_rag.indexing.weaviate.SparseEmbedder")
    @patch(
        "vectordb.langchain.cost_optimized_rag.indexing.weaviate.DataloaderCatalog.create"
    )
    def test_indexing_stores_hybrid_data(
        self,
        mock_get_docs,
        mock_sparse_embedder_class,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
    ):
        """Test that indexing stores both dense and sparse vectors."""
        from vectordb.langchain.cost_optimized_rag.indexing.weaviate import (
            WeaviateCostOptimizedRAGIndexingPipeline,
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
            "weaviate": {
                "url": "http://localhost:8080",
                "collection_name": "test_cost_optimized",
                "dimension": 384,
            },
        }

        pipeline = WeaviateCostOptimizedRAGIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 1
        assert result["chunks_created"] == 1
        mock_db_inst.upsert.assert_called_once()


class TestWeaviateCostOptimizedSearch:
    """Unit tests for Weaviate cost-optimized RAG search pipeline."""

    @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.weaviate.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.SparseEmbedder")
    @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.RAGHelper.create_llm")
    def test_search_initialization(
        self, mock_llm_helper, mock_sparse_embedder, mock_embedder_helper, mock_db
    ):
        """Test search pipeline initialization."""
        from vectordb.langchain.cost_optimized_rag.search.weaviate import (
            WeaviateCostOptimizedRAGSearchPipeline,
        )

        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "collection_name": "test_cost_optimized",
            },
            "rag": {"enabled": False},
        }

        pipeline = WeaviateCostOptimizedRAGSearchPipeline(config)
        assert pipeline.config == config
        assert pipeline.llm is None
        assert pipeline.rrf_k == 60
        assert pipeline.alpha == 0.5

    @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.weaviate.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.SparseEmbedder")
    @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.weaviate.ResultMerger.merge_and_deduplicate"
    )
    def test_search_hybrid_execution(
        self,
        mock_merge,
        mock_llm_helper,
        mock_sparse_embedder_class,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test hybrid search execution with dense and sparse retrieval."""
        from vectordb.langchain.cost_optimized_rag.search.weaviate import (
            WeaviateCostOptimizedRAGSearchPipeline,
        )

        dense_docs = [
            Document(page_content="Dense result 1", metadata={"id": "1"}),
            Document(page_content="Dense result 2", metadata={"id": "2"}),
        ]
        sparse_docs = [
            Document(page_content="Sparse result 1", metadata={"id": "3"}),
        ]
        merged_docs = [
            Document(page_content="Merged result 1", metadata={"id": "1"}),
            Document(page_content="Merged result 2", metadata={"id": "3"}),
            Document(page_content="Merged result 3", metadata={"id": "2"}),
        ]

        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = dense_docs
        mock_db_inst.query_with_sparse.return_value = sparse_docs
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder.embed_query.return_value = {
            "indices": [1],
            "values": [1.0],
        }
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        mock_merge.return_value = merged_docs

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "collection_name": "test_cost_optimized",
            },
            "rag": {"enabled": False},
        }

        pipeline = WeaviateCostOptimizedRAGSearchPipeline(config)
        result = pipeline.search("What is Python?", top_k=3)

        assert result["query"] == "What is Python?"
        mock_db_inst.query.assert_called_once()
        mock_db_inst.query_with_sparse.assert_called_once()
        mock_merge.assert_called_once()

    @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.weaviate.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.SparseEmbedder")
    @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.RAGHelper.create_llm")
    @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.RAGHelper.generate")
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.weaviate.ResultMerger.merge_and_deduplicate"
    )
    def test_search_with_rag_generation(
        self,
        mock_rag_generate,
        mock_merge,
        mock_llm_helper,
        mock_sparse_embedder_class,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test search with RAG answer generation."""
        from vectordb.langchain.cost_optimized_rag.search.weaviate import (
            WeaviateCostOptimizedRAGSearchPipeline,
        )

        merged_docs = [
            Document(
                page_content="Python is a programming language", metadata={"id": "1"}
            ),
        ]

        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = merged_docs
        mock_db_inst.query_with_sparse.return_value = []
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

        mock_merge.return_value = merged_docs

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "collection_name": "test_cost_optimized",
            },
            "rag": {"enabled": True, "model": "gpt-3.5-turbo"},
        }

        pipeline = WeaviateCostOptimizedRAGSearchPipeline(config)
        result = pipeline.search("What is Python?", top_k=1)

        assert "answer" in result
        # Result contains merged docs when LLM mock unavailable.
        mock_rag_generate.assert_called_once()

    @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.weaviate.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.weaviate.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.SparseEmbedder")
    @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.weaviate.ResultMerger.merge_and_deduplicate"
    )
    def test_search_with_filters(
        self,
        mock_merge,
        mock_llm_helper,
        mock_sparse_embedder_class,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test search with metadata filters."""
        from vectordb.langchain.cost_optimized_rag.search.weaviate import (
            WeaviateCostOptimizedRAGSearchPipeline,
        )

        merged_docs = [
            Document(
                page_content="Python programming", metadata={"category": "programming"}
            ),
        ]

        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = merged_docs
        mock_db_inst.query_with_sparse.return_value = []
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        mock_sparse_embedder = MagicMock()
        mock_sparse_embedder.embed_query.return_value = {
            "indices": [1],
            "values": [1.0],
        }
        mock_sparse_embedder_class.return_value = mock_sparse_embedder

        mock_merge.return_value = merged_docs

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "collection_name": "test_cost_optimized",
            },
            "rag": {"enabled": False},
        }

        pipeline = WeaviateCostOptimizedRAGSearchPipeline(config)
        filters = {"category": "programming"}
        result = pipeline.search("Python", top_k=1, filters=filters)

        assert len(result["documents"]) == 1
        mock_db_inst.query.assert_called_once()

    @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.WeaviateVectorDB")
    @patch(
        "vectordb.langchain.cost_optimized_rag.search.weaviate.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.SparseEmbedder")
    @patch("vectordb.langchain.cost_optimized_rag.search.weaviate.RAGHelper.create_llm")
    def test_search_custom_alpha(
        self, mock_llm_helper, mock_sparse_embedder, mock_embedder_helper, mock_db
    ):
        """Test search with custom alpha parameter for hybrid search."""
        from vectordb.langchain.cost_optimized_rag.search.weaviate import (
            WeaviateCostOptimizedRAGSearchPipeline,
        )

        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "weaviate": {
                "url": "http://localhost:8080",
                "collection_name": "test_cost_optimized",
            },
            "search": {
                "rrf_k": 120,
                "alpha": 0.7,
            },
            "rag": {"enabled": False},
        }

        pipeline = WeaviateCostOptimizedRAGSearchPipeline(config)
        assert pipeline.rrf_k == 120
        assert pipeline.alpha == 0.7
