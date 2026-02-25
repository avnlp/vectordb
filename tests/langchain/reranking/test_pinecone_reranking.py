"""Tests for Pinecone reranking pipelines (LangChain)."""

from unittest.mock import MagicMock, patch

from haystack.dataclasses import Document as HaystackDocument
from langchain_core.documents import Document


class TestPineconeRerankingIndexing:
    """Unit tests for Pinecone reranking indexing pipeline."""

    @patch("vectordb.langchain.reranking.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.reranking.indexing.pinecone.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.reranking.indexing.pinecone.DataloaderCatalog.create")
    def test_indexing_initialization(
        self, mock_get_docs, mock_embedder_helper, mock_db
    ):
        """Test pipeline initialization."""
        from vectordb.langchain.reranking.indexing.pinecone import (
            PineconeRerankingIndexingPipeline,
        )

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-index",
                "namespace": "test-namespace",
                "dimension": 384,
                "metric": "cosine",
            },
        }

        pipeline = PineconeRerankingIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.index_name == "test-index"
        assert pipeline.namespace == "test-namespace"

    @patch("vectordb.langchain.reranking.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.reranking.indexing.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.reranking.indexing.pinecone.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.reranking.indexing.pinecone.DataloaderCatalog.create")
    def test_indexing_run_with_documents(
        self,
        mock_get_docs,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
    ):
        """Test indexing with documents."""
        from vectordb.langchain.reranking.indexing.pinecone import (
            PineconeRerankingIndexingPipeline,
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

        mock_db_inst = MagicMock()
        mock_db_inst.upsert.return_value = len(sample_documents)
        mock_db_inst.create_index.return_value = True
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-index",
                "namespace": "test-namespace",
                "dimension": 384,
                "metric": "cosine",
            },
        }

        pipeline = PineconeRerankingIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.create_index.assert_called_once()
        mock_db_inst.upsert.assert_called_once()

    @patch("vectordb.langchain.reranking.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.reranking.indexing.pinecone.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.reranking.indexing.pinecone.DataloaderCatalog.create")
    def test_indexing_run_no_documents(
        self, mock_get_docs, mock_embedder_helper, mock_db
    ):
        """Test indexing with no documents."""
        from vectordb.langchain.reranking.indexing.pinecone import (
            PineconeRerankingIndexingPipeline,
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
                "api_key": "test-key",
                "index_name": "test-index",
                "namespace": "test-namespace",
                "dimension": 384,
                "metric": "cosine",
            },
        }

        pipeline = PineconeRerankingIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0

    @patch("vectordb.langchain.reranking.indexing.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.reranking.indexing.pinecone.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.reranking.indexing.pinecone.EmbedderHelper.embed_documents"
    )
    @patch("vectordb.langchain.reranking.indexing.pinecone.DataloaderCatalog.create")
    def test_indexing_with_recreate(
        self,
        mock_get_docs,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
    ):
        """Test indexing with recreate option."""
        from vectordb.langchain.reranking.indexing.pinecone import (
            PineconeRerankingIndexingPipeline,
        )

        sample_documents = [
            Document(
                page_content="Python is a high-level programming language",
                metadata={"source": "wiki", "id": "1"},
            ),
        ]

        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader
        mock_embed_docs.return_value = (sample_documents, [[0.1] * 384])

        mock_db_inst = MagicMock()
        mock_db_inst.upsert.return_value = len(sample_documents)
        mock_db_inst.create_index.return_value = True
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-index",
                "namespace": "test-namespace",
                "dimension": 384,
                "metric": "cosine",
                "recreate": True,
            },
        }

        pipeline = PineconeRerankingIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)
        mock_db_inst.create_index.assert_called_once()
        call_kwargs = mock_db_inst.create_index.call_args.kwargs
        assert call_kwargs["recreate"] is True


class TestPineconeRerankingSearch:
    """Unit tests for Pinecone reranking search pipeline."""

    @patch("vectordb.langchain.reranking.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.reranking.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.reranking.search.pinecone.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.reranking.search.pinecone.RerankerHelper.create_reranker"
    )
    def test_search_initialization(
        self, mock_reranker_helper, mock_llm_helper, mock_embedder_helper, mock_db
    ):
        """Test search pipeline initialization."""
        from vectordb.langchain.reranking.search.pinecone import (
            PineconeRerankingSearchPipeline,
        )

        mock_llm_helper.return_value = None
        mock_reranker_helper.return_value = MagicMock()

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-index",
                "namespace": "test-namespace",
            },
            "rag": {"enabled": False},
        }

        pipeline = PineconeRerankingSearchPipeline(config)
        assert pipeline.config == config
        assert pipeline.llm is None

    @patch("vectordb.langchain.reranking.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.reranking.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.reranking.search.pinecone.EmbedderHelper.embed_query")
    @patch("vectordb.langchain.reranking.search.pinecone.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.reranking.search.pinecone.RerankerHelper.create_reranker"
    )
    @patch("vectordb.langchain.reranking.search.pinecone.RerankerHelper.rerank")
    def test_search_execution(
        self,
        mock_rerank,
        mock_reranker_helper,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test search execution."""
        from vectordb.langchain.reranking.search.pinecone import (
            PineconeRerankingSearchPipeline,
        )

        sample_documents = [
            HaystackDocument(
                content="Python is a high-level programming language",
                meta={"source": "wiki"},
                id="1",
            ),
            HaystackDocument(
                content="Machine learning uses algorithms to learn from data",
                meta={"source": "wiki"},
                id="2",
            ),
        ]

        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        mock_reranker = MagicMock()
        mock_reranker_helper.return_value = mock_reranker
        mock_rerank.return_value = sample_documents[:1]

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-index",
                "namespace": "test-namespace",
            },
            "rag": {"enabled": False},
        }

        pipeline = PineconeRerankingSearchPipeline(config)
        result = pipeline.search("test query", top_k=10, rerank_k=5)

        assert result["query"] == "test query"
        assert len(result["documents"]) == 1
        assert "answer" not in result

    @patch("vectordb.langchain.reranking.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.reranking.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.reranking.search.pinecone.EmbedderHelper.embed_query")
    @patch("vectordb.langchain.reranking.search.pinecone.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.reranking.search.pinecone.RerankerHelper.create_reranker"
    )
    @patch("vectordb.langchain.reranking.search.pinecone.RAGHelper.generate")
    @patch("vectordb.langchain.reranking.search.pinecone.RerankerHelper.rerank")
    def test_search_with_rag(
        self,
        mock_rerank,
        mock_rag_generate,
        mock_reranker_helper,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test search with RAG generation."""
        from vectordb.langchain.reranking.search.pinecone import (
            PineconeRerankingSearchPipeline,
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
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst

        mock_llm_inst = MagicMock()
        mock_llm_helper.return_value = mock_llm_inst
        mock_rag_generate.return_value = "Test answer"

        mock_reranker = MagicMock()
        mock_reranker_helper.return_value = mock_reranker
        mock_rerank.return_value = sample_documents

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-index",
                "namespace": "test-namespace",
            },
            "rag": {"enabled": True},
        }

        pipeline = PineconeRerankingSearchPipeline(config)
        result = pipeline.search("test query", top_k=10, rerank_k=5)

        assert result["query"] == "test query"
        assert result["answer"] == "Test answer"

    @patch("vectordb.langchain.reranking.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.reranking.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.reranking.search.pinecone.EmbedderHelper.embed_query")
    @patch("vectordb.langchain.reranking.search.pinecone.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.reranking.search.pinecone.RerankerHelper.create_reranker"
    )
    @patch("vectordb.langchain.reranking.search.pinecone.RerankerHelper.rerank")
    def test_search_with_filters(
        self,
        mock_rerank,
        mock_reranker_helper,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test search with filters."""
        from vectordb.langchain.reranking.search.pinecone import (
            PineconeRerankingSearchPipeline,
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
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        mock_reranker = MagicMock()
        mock_reranker_helper.return_value = mock_reranker
        mock_rerank.return_value = sample_documents

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-index",
                "namespace": "test-namespace",
            },
            "rag": {"enabled": False},
        }

        pipeline = PineconeRerankingSearchPipeline(config)
        filters = {"source": "wiki"}
        result = pipeline.search("test query", top_k=10, rerank_k=5, filters=filters)

        assert result["query"] == "test query"
        mock_db_inst.query.assert_called_once()
        call_kwargs = mock_db_inst.query.call_args.kwargs
        assert call_kwargs["filter"] == filters

    @patch("vectordb.langchain.reranking.search.pinecone.PineconeVectorDB")
    @patch(
        "vectordb.langchain.reranking.search.pinecone.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.reranking.search.pinecone.EmbedderHelper.embed_query")
    @patch("vectordb.langchain.reranking.search.pinecone.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.reranking.search.pinecone.RerankerHelper.create_reranker"
    )
    @patch("vectordb.langchain.reranking.search.pinecone.RerankerHelper.rerank")
    def test_search_empty_results(
        self,
        mock_rerank,
        mock_reranker_helper,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
    ):
        """Test search with empty results."""
        from vectordb.langchain.reranking.search.pinecone import (
            PineconeRerankingSearchPipeline,
        )

        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = []
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        mock_reranker = MagicMock()
        mock_reranker_helper.return_value = mock_reranker
        mock_rerank.return_value = []

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "pinecone": {
                "api_key": "test-key",
                "index_name": "test-index",
                "namespace": "test-namespace",
            },
            "rag": {"enabled": False},
        }

        pipeline = PineconeRerankingSearchPipeline(config)
        result = pipeline.search("test query", top_k=10, rerank_k=5)

        assert result["query"] == "test query"
        assert result["documents"] == []
