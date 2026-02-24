"""Tests for Weaviate parent document retrieval pipeline (indexing and search)."""

from unittest.mock import MagicMock, patch

from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore


class TestWeaviateParentDocIndexing:
    """Unit tests for Weaviate parent document indexing pipeline."""

    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.weaviate.load_parent_doc_config"
    )
    @patch("vectordb.haystack.parent_document_retrieval.indexing.weaviate.setup_logger")
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.weaviate.SentenceTransformersDocumentEmbedder"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.weaviate.HierarchicalDocumentSplitter"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.weaviate.WeaviateVectorDB"
    )
    def test_indexing_init_loads_config(
        self,
        mock_db: MagicMock,
        mock_splitter: MagicMock,
        mock_embedder: MagicMock,
        mock_logger: MagicMock,
        mock_load_config: MagicMock,
        weaviate_config: dict,
    ) -> None:
        """Test indexing pipeline initialization loads config correctly."""
        from vectordb.haystack.parent_document_retrieval.indexing.weaviate import (
            WeaviateParentDocIndexingPipeline,
        )

        mock_load_config.return_value = weaviate_config

        pipeline = WeaviateParentDocIndexingPipeline("config.yaml")

        assert pipeline.config == weaviate_config
        mock_load_config.assert_called_once_with("config.yaml")
        mock_db.assert_called_once()

    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.weaviate.load_parent_doc_config"
    )
    @patch("vectordb.haystack.parent_document_retrieval.indexing.weaviate.setup_logger")
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.weaviate.SentenceTransformersDocumentEmbedder"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.weaviate.HierarchicalDocumentSplitter"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.weaviate.WeaviateVectorDB"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.weaviate.DataloaderCatalog"
    )
    def test_indexing_run_creates_hierarchy(
        self,
        mock_registry: MagicMock,
        mock_db_class: MagicMock,
        mock_splitter_class: MagicMock,
        mock_embedder_class: MagicMock,
        mock_logger: MagicMock,
        mock_load_config: MagicMock,
        weaviate_config: dict,
        sample_documents: list[Document],
        sample_parent_documents: list[Document],
        sample_leaf_documents: list[Document],
    ) -> None:
        """Test indexing run creates document hierarchy."""
        from vectordb.haystack.parent_document_retrieval.indexing.weaviate import (
            WeaviateParentDocIndexingPipeline,
        )

        mock_load_config.return_value = weaviate_config
        # Set up proper mock chain for DataloaderCatalog
        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_registry.create.return_value = mock_loader

        mock_splitter = MagicMock()
        all_docs = sample_parent_documents + sample_leaf_documents
        mock_splitter.run.return_value = {"documents": all_docs}
        mock_splitter_class.return_value = mock_splitter

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"documents": sample_leaf_documents}
        mock_embedder_class.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db

        pipeline = WeaviateParentDocIndexingPipeline("config.yaml")
        result = pipeline.run()

        assert result["num_parents"] == len(sample_parent_documents)
        assert result["num_leaves"] == len(sample_leaf_documents)
        mock_db.upsert.assert_called_once()

    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.weaviate.load_parent_doc_config"
    )
    @patch("vectordb.haystack.parent_document_retrieval.indexing.weaviate.setup_logger")
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.weaviate.SentenceTransformersDocumentEmbedder"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.weaviate.HierarchicalDocumentSplitter"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.weaviate.WeaviateVectorDB"
    )
    def test_indexing_creates_collection(
        self,
        mock_db_class: MagicMock,
        mock_splitter: MagicMock,
        mock_embedder: MagicMock,
        mock_logger: MagicMock,
        mock_load_config: MagicMock,
        weaviate_config: dict,
    ) -> None:
        """Test indexing creates collection with correct dimension."""
        from vectordb.haystack.parent_document_retrieval.indexing.weaviate import (
            WeaviateParentDocIndexingPipeline,
        )

        mock_load_config.return_value = weaviate_config
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db

        WeaviateParentDocIndexingPipeline("config.yaml")

        mock_db.create_collection.assert_called_once()
        call_kwargs = mock_db.create_collection.call_args[1]
        assert call_kwargs["dimension"] == 1024

    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.weaviate.load_parent_doc_config"
    )
    @patch("vectordb.haystack.parent_document_retrieval.indexing.weaviate.setup_logger")
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.weaviate.SentenceTransformersDocumentEmbedder"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.weaviate.HierarchicalDocumentSplitter"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.weaviate.WeaviateVectorDB"
    )
    def test_indexing_initializes_splitter_with_config(
        self,
        mock_db: MagicMock,
        mock_splitter_class: MagicMock,
        mock_embedder: MagicMock,
        mock_logger: MagicMock,
        mock_load_config: MagicMock,
        weaviate_config: dict,
    ) -> None:
        """Test splitter is initialized with config values."""
        from vectordb.haystack.parent_document_retrieval.indexing.weaviate import (
            WeaviateParentDocIndexingPipeline,
        )

        mock_load_config.return_value = weaviate_config

        WeaviateParentDocIndexingPipeline("config.yaml")

        mock_splitter_class.assert_called_once()
        call_kwargs = mock_splitter_class.call_args[1]
        assert call_kwargs["block_sizes"] == {100, 25}
        assert call_kwargs["split_overlap"] == 5
        assert call_kwargs["split_by"] == "word"

    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.weaviate.load_parent_doc_config"
    )
    @patch("vectordb.haystack.parent_document_retrieval.indexing.weaviate.setup_logger")
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.weaviate.SentenceTransformersDocumentEmbedder"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.weaviate.HierarchicalDocumentSplitter"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.weaviate.WeaviateVectorDB"
    )
    def test_indexing_has_parent_store(
        self,
        mock_db: MagicMock,
        mock_splitter: MagicMock,
        mock_embedder: MagicMock,
        mock_logger: MagicMock,
        mock_load_config: MagicMock,
        weaviate_config: dict,
    ) -> None:
        """Test indexing pipeline creates parent store."""
        from vectordb.haystack.parent_document_retrieval.indexing.weaviate import (
            WeaviateParentDocIndexingPipeline,
        )

        mock_load_config.return_value = weaviate_config

        pipeline = WeaviateParentDocIndexingPipeline("config.yaml")

        assert isinstance(pipeline.parent_store, InMemoryDocumentStore)


class TestWeaviateParentDocSearch:
    """Unit tests for Weaviate parent document search pipeline."""

    @patch(
        "vectordb.haystack.parent_document_retrieval.search.weaviate.load_parent_doc_config"
    )
    @patch("vectordb.haystack.parent_document_retrieval.search.weaviate.setup_logger")
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.weaviate.SentenceTransformersTextEmbedder"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.weaviate.WeaviateVectorDB"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.weaviate.AutoMergingRetriever"
    )
    def test_search_init_loads_config(
        self,
        mock_merger: MagicMock,
        mock_db: MagicMock,
        mock_embedder: MagicMock,
        mock_logger: MagicMock,
        mock_load_config: MagicMock,
        weaviate_config: dict,
        sample_parent_store: InMemoryDocumentStore,
    ) -> None:
        """Test search pipeline initialization loads config correctly."""
        from vectordb.haystack.parent_document_retrieval.search.weaviate import (
            WeaviateParentDocSearchPipeline,
        )

        mock_load_config.return_value = weaviate_config

        pipeline = WeaviateParentDocSearchPipeline("config.yaml", sample_parent_store)

        assert pipeline.config == weaviate_config
        assert pipeline.parent_store == sample_parent_store
        mock_db.assert_called_once()

    @patch(
        "vectordb.haystack.parent_document_retrieval.search.weaviate.load_parent_doc_config"
    )
    @patch("vectordb.haystack.parent_document_retrieval.search.weaviate.setup_logger")
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.weaviate.SentenceTransformersTextEmbedder"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.weaviate.WeaviateVectorDB"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.weaviate.AutoMergingRetriever"
    )
    def test_search_calls_query(
        self,
        mock_merger_class: MagicMock,
        mock_db_class: MagicMock,
        mock_embedder_class: MagicMock,
        mock_logger: MagicMock,
        mock_load_config: MagicMock,
        weaviate_config: dict,
        sample_parent_store: InMemoryDocumentStore,
        sample_leaf_documents: list[Document],
        sample_embedding: list[float],
    ) -> None:
        """Test search method calls query on database."""
        from vectordb.haystack.parent_document_retrieval.search.weaviate import (
            WeaviateParentDocSearchPipeline,
        )

        mock_load_config.return_value = weaviate_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.query.return_value = sample_leaf_documents
        mock_db_class.return_value = mock_db

        mock_merger = MagicMock()
        mock_merger.run.return_value = {"documents": sample_leaf_documents[:2]}
        mock_merger_class.return_value = mock_merger

        pipeline = WeaviateParentDocSearchPipeline("config.yaml", sample_parent_store)
        result = pipeline.search("test query", top_k=5)

        mock_db.query.assert_called_once()
        assert "documents" in result
        assert "num_leaves_matched" in result

    @patch(
        "vectordb.haystack.parent_document_retrieval.search.weaviate.load_parent_doc_config"
    )
    @patch("vectordb.haystack.parent_document_retrieval.search.weaviate.setup_logger")
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.weaviate.SentenceTransformersTextEmbedder"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.weaviate.WeaviateVectorDB"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.weaviate.AutoMergingRetriever"
    )
    def test_search_oversamples_leaves(
        self,
        mock_merger_class: MagicMock,
        mock_db_class: MagicMock,
        mock_embedder_class: MagicMock,
        mock_logger: MagicMock,
        mock_load_config: MagicMock,
        weaviate_config: dict,
        sample_parent_store: InMemoryDocumentStore,
        sample_embedding: list[float],
    ) -> None:
        """Test search oversamples leaves (top_k * 3)."""
        from vectordb.haystack.parent_document_retrieval.search.weaviate import (
            WeaviateParentDocSearchPipeline,
        )

        mock_load_config.return_value = weaviate_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.query.return_value = []
        mock_db_class.return_value = mock_db

        mock_merger = MagicMock()
        mock_merger.run.return_value = {"documents": []}
        mock_merger_class.return_value = mock_merger

        pipeline = WeaviateParentDocSearchPipeline("config.yaml", sample_parent_store)
        pipeline.search("test query", top_k=5)

        call_kwargs = mock_db.query.call_args[1]
        assert call_kwargs["top_k"] == 15  # 5 * 3

    @patch(
        "vectordb.haystack.parent_document_retrieval.search.weaviate.load_parent_doc_config"
    )
    @patch("vectordb.haystack.parent_document_retrieval.search.weaviate.setup_logger")
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.weaviate.SentenceTransformersTextEmbedder"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.weaviate.WeaviateVectorDB"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.weaviate.AutoMergingRetriever"
    )
    def test_search_uses_auto_merger(
        self,
        mock_merger_class: MagicMock,
        mock_db_class: MagicMock,
        mock_embedder_class: MagicMock,
        mock_logger: MagicMock,
        mock_load_config: MagicMock,
        weaviate_config: dict,
        sample_parent_store: InMemoryDocumentStore,
        sample_leaf_documents: list[Document],
        sample_embedding: list[float],
    ) -> None:
        """Test search uses auto-merging retriever."""
        from vectordb.haystack.parent_document_retrieval.search.weaviate import (
            WeaviateParentDocSearchPipeline,
        )

        mock_load_config.return_value = weaviate_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.query.return_value = sample_leaf_documents
        mock_db_class.return_value = mock_db

        merged_docs = [Document(content="Merged parent content")]
        mock_merger = MagicMock()
        mock_merger.run.return_value = {"documents": merged_docs}
        mock_merger_class.return_value = mock_merger

        pipeline = WeaviateParentDocSearchPipeline("config.yaml", sample_parent_store)
        result = pipeline.search("test query")

        mock_merger.run.assert_called_once_with(documents=sample_leaf_documents)
        assert result["documents"] == merged_docs

    @patch(
        "vectordb.haystack.parent_document_retrieval.search.weaviate.load_parent_doc_config"
    )
    @patch("vectordb.haystack.parent_document_retrieval.search.weaviate.setup_logger")
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.weaviate.SentenceTransformersTextEmbedder"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.weaviate.WeaviateVectorDB"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.weaviate.AutoMergingRetriever"
    )
    def test_search_initializes_auto_merger_with_threshold(
        self,
        mock_merger_class: MagicMock,
        mock_db: MagicMock,
        mock_embedder: MagicMock,
        mock_logger: MagicMock,
        mock_load_config: MagicMock,
        weaviate_config: dict,
        sample_parent_store: InMemoryDocumentStore,
    ) -> None:
        """Test auto merger is initialized with config threshold."""
        from vectordb.haystack.parent_document_retrieval.search.weaviate import (
            WeaviateParentDocSearchPipeline,
        )

        mock_load_config.return_value = weaviate_config

        WeaviateParentDocSearchPipeline("config.yaml", sample_parent_store)

        mock_merger_class.assert_called_once()
        call_kwargs = mock_merger_class.call_args[1]
        assert call_kwargs["threshold"] == 0.5
        assert call_kwargs["document_store"] == sample_parent_store

    @patch(
        "vectordb.haystack.parent_document_retrieval.search.weaviate.load_parent_doc_config"
    )
    @patch("vectordb.haystack.parent_document_retrieval.search.weaviate.setup_logger")
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.weaviate.SentenceTransformersTextEmbedder"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.weaviate.WeaviateVectorDB"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.weaviate.AutoMergingRetriever"
    )
    def test_search_returns_correct_structure(
        self,
        mock_merger_class: MagicMock,
        mock_db_class: MagicMock,
        mock_embedder_class: MagicMock,
        mock_logger: MagicMock,
        mock_load_config: MagicMock,
        weaviate_config: dict,
        sample_parent_store: InMemoryDocumentStore,
        sample_leaf_documents: list[Document],
        sample_embedding: list[float],
    ) -> None:
        """Test search returns dict with correct keys."""
        from vectordb.haystack.parent_document_retrieval.search.weaviate import (
            WeaviateParentDocSearchPipeline,
        )

        mock_load_config.return_value = weaviate_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.query.return_value = sample_leaf_documents
        mock_db_class.return_value = mock_db

        mock_merger = MagicMock()
        mock_merger.run.return_value = {"documents": sample_leaf_documents[:2]}
        mock_merger_class.return_value = mock_merger

        pipeline = WeaviateParentDocSearchPipeline("config.yaml", sample_parent_store)
        result = pipeline.search("test query", top_k=5)

        assert "query" in result
        assert result["query"] == "test query"
        assert "documents" in result
        assert "num_leaves_matched" in result
        assert "num_parents_returned" in result
        assert result["num_leaves_matched"] == len(sample_leaf_documents)
        assert result["num_parents_returned"] == 2
