"""Tests for Chroma parent document retrieval pipeline (indexing and search)."""

from unittest.mock import MagicMock, patch

from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore


class TestChromaParentDocIndexing:
    """Unit tests for Chroma parent document indexing pipeline."""

    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.chroma.load_parent_doc_config"
    )
    @patch("vectordb.haystack.parent_document_retrieval.indexing.chroma.setup_logger")
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.chroma.SentenceTransformersDocumentEmbedder"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.chroma.HierarchicalDocumentSplitter"
    )
    @patch("vectordb.haystack.parent_document_retrieval.indexing.chroma.ChromaVectorDB")
    def test_indexing_init_loads_config(
        self,
        mock_db: MagicMock,
        mock_splitter: MagicMock,
        mock_embedder: MagicMock,
        mock_logger: MagicMock,
        mock_load_config: MagicMock,
        chroma_config: dict,
    ) -> None:
        """Test indexing pipeline initialization loads config correctly."""
        from vectordb.haystack.parent_document_retrieval.indexing.chroma import (
            ChromaParentDocIndexingPipeline,
        )

        mock_load_config.return_value = chroma_config

        pipeline = ChromaParentDocIndexingPipeline("config.yaml")

        assert pipeline.config == chroma_config
        mock_load_config.assert_called_once_with("config.yaml")
        mock_db.assert_called_once()

    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.chroma.load_parent_doc_config"
    )
    @patch("vectordb.haystack.parent_document_retrieval.indexing.chroma.setup_logger")
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.chroma.SentenceTransformersDocumentEmbedder"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.chroma.HierarchicalDocumentSplitter"
    )
    @patch("vectordb.haystack.parent_document_retrieval.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.chroma.DataloaderCatalog"
    )
    def test_indexing_run_creates_hierarchy(
        self,
        mock_registry: MagicMock,
        mock_db_class: MagicMock,
        mock_splitter_class: MagicMock,
        mock_embedder_class: MagicMock,
        mock_logger: MagicMock,
        mock_load_config: MagicMock,
        chroma_config: dict,
        sample_documents: list[Document],
        sample_parent_documents: list[Document],
        sample_leaf_documents: list[Document],
    ) -> None:
        """Test indexing run creates document hierarchy."""
        from vectordb.haystack.parent_document_retrieval.indexing.chroma import (
            ChromaParentDocIndexingPipeline,
        )

        mock_load_config.return_value = chroma_config
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

        pipeline = ChromaParentDocIndexingPipeline("config.yaml")
        result = pipeline.run()

        assert result["num_parents"] == len(sample_parent_documents)
        assert result["num_leaves"] == len(sample_leaf_documents)
        mock_db.upsert.assert_called_once()

    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.chroma.load_parent_doc_config"
    )
    @patch("vectordb.haystack.parent_document_retrieval.indexing.chroma.setup_logger")
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.chroma.SentenceTransformersDocumentEmbedder"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.chroma.HierarchicalDocumentSplitter"
    )
    @patch("vectordb.haystack.parent_document_retrieval.indexing.chroma.ChromaVectorDB")
    def test_indexing_creates_collection(
        self,
        mock_db_class: MagicMock,
        mock_splitter: MagicMock,
        mock_embedder: MagicMock,
        mock_logger: MagicMock,
        mock_load_config: MagicMock,
        chroma_config: dict,
    ) -> None:
        """Test indexing creates collection with correct dimension."""
        from vectordb.haystack.parent_document_retrieval.indexing.chroma import (
            ChromaParentDocIndexingPipeline,
        )

        mock_load_config.return_value = chroma_config
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db

        ChromaParentDocIndexingPipeline("config.yaml")

        mock_db.create_collection.assert_called_once()
        call_kwargs = mock_db.create_collection.call_args[1]
        assert call_kwargs["dimension"] == 1024

    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.chroma.load_parent_doc_config"
    )
    @patch("vectordb.haystack.parent_document_retrieval.indexing.chroma.setup_logger")
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.chroma.SentenceTransformersDocumentEmbedder"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.chroma.HierarchicalDocumentSplitter"
    )
    @patch("vectordb.haystack.parent_document_retrieval.indexing.chroma.ChromaVectorDB")
    def test_indexing_initializes_splitter_with_config(
        self,
        mock_db: MagicMock,
        mock_splitter_class: MagicMock,
        mock_embedder: MagicMock,
        mock_logger: MagicMock,
        mock_load_config: MagicMock,
        chroma_config: dict,
    ) -> None:
        """Test splitter is initialized with config values."""
        from vectordb.haystack.parent_document_retrieval.indexing.chroma import (
            ChromaParentDocIndexingPipeline,
        )

        mock_load_config.return_value = chroma_config

        ChromaParentDocIndexingPipeline("config.yaml")

        mock_splitter_class.assert_called_once()
        call_kwargs = mock_splitter_class.call_args[1]
        assert call_kwargs["block_sizes"] == {100, 25}
        assert call_kwargs["split_overlap"] == 5
        assert call_kwargs["split_by"] == "word"

    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.chroma.load_parent_doc_config"
    )
    @patch("vectordb.haystack.parent_document_retrieval.indexing.chroma.setup_logger")
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.chroma.SentenceTransformersDocumentEmbedder"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.chroma.HierarchicalDocumentSplitter"
    )
    @patch("vectordb.haystack.parent_document_retrieval.indexing.chroma.ChromaVectorDB")
    def test_indexing_has_parent_store(
        self,
        mock_db: MagicMock,
        mock_splitter: MagicMock,
        mock_embedder: MagicMock,
        mock_logger: MagicMock,
        mock_load_config: MagicMock,
        chroma_config: dict,
    ) -> None:
        """Test indexing pipeline creates parent store."""
        from vectordb.haystack.parent_document_retrieval.indexing.chroma import (
            ChromaParentDocIndexingPipeline,
        )

        mock_load_config.return_value = chroma_config

        pipeline = ChromaParentDocIndexingPipeline("config.yaml")

        assert isinstance(pipeline.parent_store, InMemoryDocumentStore)

    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.chroma.load_parent_doc_config"
    )
    @patch("vectordb.haystack.parent_document_retrieval.indexing.chroma.setup_logger")
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.chroma.SentenceTransformersDocumentEmbedder"
    )
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.chroma.HierarchicalDocumentSplitter"
    )
    @patch("vectordb.haystack.parent_document_retrieval.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.haystack.parent_document_retrieval.indexing.chroma.DataloaderCatalog"
    )
    def test_indexing_stores_parents_in_memory(
        self,
        mock_registry: MagicMock,
        mock_db_class: MagicMock,
        mock_splitter_class: MagicMock,
        mock_embedder_class: MagicMock,
        mock_logger: MagicMock,
        mock_load_config: MagicMock,
        chroma_config: dict,
        sample_documents: list[Document],
        sample_parent_documents: list[Document],
        sample_leaf_documents: list[Document],
    ) -> None:
        """Test indexing stores parent documents in InMemoryDocumentStore."""
        from vectordb.haystack.parent_document_retrieval.indexing.chroma import (
            ChromaParentDocIndexingPipeline,
        )

        mock_load_config.return_value = chroma_config
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

        pipeline = ChromaParentDocIndexingPipeline("config.yaml")
        pipeline.run()

        stored_docs = pipeline.parent_store.filter_documents()
        assert len(stored_docs) == len(sample_parent_documents)


class TestChromaParentDocSearch:
    """Unit tests for Chroma parent document search pipeline."""

    @patch(
        "vectordb.haystack.parent_document_retrieval.search.chroma.load_parent_doc_config"
    )
    @patch("vectordb.haystack.parent_document_retrieval.search.chroma.setup_logger")
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.chroma.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.parent_document_retrieval.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.chroma.AutoMergingRetriever"
    )
    def test_search_init_loads_config(
        self,
        mock_merger: MagicMock,
        mock_db: MagicMock,
        mock_embedder: MagicMock,
        mock_logger: MagicMock,
        mock_load_config: MagicMock,
        chroma_config: dict,
        sample_parent_store: InMemoryDocumentStore,
    ) -> None:
        """Test search pipeline initialization loads config correctly."""
        from vectordb.haystack.parent_document_retrieval.search.chroma import (
            ChromaParentDocSearchPipeline,
        )

        mock_load_config.return_value = chroma_config

        pipeline = ChromaParentDocSearchPipeline("config.yaml", sample_parent_store)

        assert pipeline.config == chroma_config
        assert pipeline.parent_store == sample_parent_store
        mock_db.assert_called_once()

    @patch(
        "vectordb.haystack.parent_document_retrieval.search.chroma.load_parent_doc_config"
    )
    @patch("vectordb.haystack.parent_document_retrieval.search.chroma.setup_logger")
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.chroma.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.parent_document_retrieval.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.chroma.AutoMergingRetriever"
    )
    def test_search_calls_query(
        self,
        mock_merger_class: MagicMock,
        mock_db_class: MagicMock,
        mock_embedder_class: MagicMock,
        mock_logger: MagicMock,
        mock_load_config: MagicMock,
        chroma_config: dict,
        sample_parent_store: InMemoryDocumentStore,
        sample_leaf_documents: list[Document],
        sample_embedding: list[float],
    ) -> None:
        """Test search method calls query on database."""
        from vectordb.haystack.parent_document_retrieval.search.chroma import (
            ChromaParentDocSearchPipeline,
        )

        mock_load_config.return_value = chroma_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.query.return_value = sample_leaf_documents
        mock_db_class.return_value = mock_db

        mock_merger = MagicMock()
        mock_merger.run.return_value = {"documents": sample_leaf_documents[:2]}
        mock_merger_class.return_value = mock_merger

        pipeline = ChromaParentDocSearchPipeline("config.yaml", sample_parent_store)
        result = pipeline.search("test query", top_k=5)

        mock_db.query.assert_called_once()
        assert "documents" in result
        assert "num_leaves_matched" in result

    @patch(
        "vectordb.haystack.parent_document_retrieval.search.chroma.load_parent_doc_config"
    )
    @patch("vectordb.haystack.parent_document_retrieval.search.chroma.setup_logger")
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.chroma.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.parent_document_retrieval.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.chroma.AutoMergingRetriever"
    )
    def test_search_oversamples_leaves(
        self,
        mock_merger_class: MagicMock,
        mock_db_class: MagicMock,
        mock_embedder_class: MagicMock,
        mock_logger: MagicMock,
        mock_load_config: MagicMock,
        chroma_config: dict,
        sample_parent_store: InMemoryDocumentStore,
        sample_embedding: list[float],
    ) -> None:
        """Test search oversamples leaves (top_k * 3)."""
        from vectordb.haystack.parent_document_retrieval.search.chroma import (
            ChromaParentDocSearchPipeline,
        )

        mock_load_config.return_value = chroma_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.query.return_value = []
        mock_db_class.return_value = mock_db

        mock_merger = MagicMock()
        mock_merger.run.return_value = {"documents": []}
        mock_merger_class.return_value = mock_merger

        pipeline = ChromaParentDocSearchPipeline("config.yaml", sample_parent_store)
        pipeline.search("test query", top_k=5)

        call_kwargs = mock_db.query.call_args[1]
        assert call_kwargs["top_k"] == 15  # 5 * 3

    @patch(
        "vectordb.haystack.parent_document_retrieval.search.chroma.load_parent_doc_config"
    )
    @patch("vectordb.haystack.parent_document_retrieval.search.chroma.setup_logger")
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.chroma.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.parent_document_retrieval.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.chroma.AutoMergingRetriever"
    )
    def test_search_uses_auto_merger(
        self,
        mock_merger_class: MagicMock,
        mock_db_class: MagicMock,
        mock_embedder_class: MagicMock,
        mock_logger: MagicMock,
        mock_load_config: MagicMock,
        chroma_config: dict,
        sample_parent_store: InMemoryDocumentStore,
        sample_leaf_documents: list[Document],
        sample_embedding: list[float],
    ) -> None:
        """Test search uses auto-merging retriever."""
        from vectordb.haystack.parent_document_retrieval.search.chroma import (
            ChromaParentDocSearchPipeline,
        )

        mock_load_config.return_value = chroma_config

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

        pipeline = ChromaParentDocSearchPipeline("config.yaml", sample_parent_store)
        result = pipeline.search("test query")

        mock_merger.run.assert_called_once_with(documents=sample_leaf_documents)
        assert result["documents"] == merged_docs

    @patch(
        "vectordb.haystack.parent_document_retrieval.search.chroma.load_parent_doc_config"
    )
    @patch("vectordb.haystack.parent_document_retrieval.search.chroma.setup_logger")
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.chroma.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.parent_document_retrieval.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.chroma.AutoMergingRetriever"
    )
    def test_search_initializes_auto_merger_with_threshold(
        self,
        mock_merger_class: MagicMock,
        mock_db: MagicMock,
        mock_embedder: MagicMock,
        mock_logger: MagicMock,
        mock_load_config: MagicMock,
        chroma_config: dict,
        sample_parent_store: InMemoryDocumentStore,
    ) -> None:
        """Test auto merger is initialized with config threshold."""
        from vectordb.haystack.parent_document_retrieval.search.chroma import (
            ChromaParentDocSearchPipeline,
        )

        mock_load_config.return_value = chroma_config

        ChromaParentDocSearchPipeline("config.yaml", sample_parent_store)

        mock_merger_class.assert_called_once()
        call_kwargs = mock_merger_class.call_args[1]
        assert call_kwargs["threshold"] == 0.5
        assert call_kwargs["document_store"] == sample_parent_store

    @patch(
        "vectordb.haystack.parent_document_retrieval.search.chroma.load_parent_doc_config"
    )
    @patch("vectordb.haystack.parent_document_retrieval.search.chroma.setup_logger")
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.chroma.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.parent_document_retrieval.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.chroma.AutoMergingRetriever"
    )
    def test_search_returns_correct_structure(
        self,
        mock_merger_class: MagicMock,
        mock_db_class: MagicMock,
        mock_embedder_class: MagicMock,
        mock_logger: MagicMock,
        mock_load_config: MagicMock,
        chroma_config: dict,
        sample_parent_store: InMemoryDocumentStore,
        sample_leaf_documents: list[Document],
        sample_embedding: list[float],
    ) -> None:
        """Test search returns dict with correct keys."""
        from vectordb.haystack.parent_document_retrieval.search.chroma import (
            ChromaParentDocSearchPipeline,
        )

        mock_load_config.return_value = chroma_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.query.return_value = sample_leaf_documents
        mock_db_class.return_value = mock_db

        mock_merger = MagicMock()
        mock_merger.run.return_value = {"documents": sample_leaf_documents[:2]}
        mock_merger_class.return_value = mock_merger

        pipeline = ChromaParentDocSearchPipeline("config.yaml", sample_parent_store)
        result = pipeline.search("test query", top_k=5)

        assert "query" in result
        assert result["query"] == "test query"
        assert "documents" in result
        assert "num_leaves_matched" in result
        assert "num_parents_returned" in result
        assert result["num_leaves_matched"] == len(sample_leaf_documents)
        assert result["num_parents_returned"] == 2

    @patch(
        "vectordb.haystack.parent_document_retrieval.search.chroma.load_parent_doc_config"
    )
    @patch("vectordb.haystack.parent_document_retrieval.search.chroma.setup_logger")
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.chroma.SentenceTransformersTextEmbedder"
    )
    @patch("vectordb.haystack.parent_document_retrieval.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.haystack.parent_document_retrieval.search.chroma.AutoMergingRetriever"
    )
    def test_search_limits_results_to_top_k(
        self,
        mock_merger_class: MagicMock,
        mock_db_class: MagicMock,
        mock_embedder_class: MagicMock,
        mock_logger: MagicMock,
        mock_load_config: MagicMock,
        chroma_config: dict,
        sample_parent_store: InMemoryDocumentStore,
        sample_leaf_documents: list[Document],
        sample_embedding: list[float],
    ) -> None:
        """Test search limits merged results to top_k."""
        from vectordb.haystack.parent_document_retrieval.search.chroma import (
            ChromaParentDocSearchPipeline,
        )

        mock_load_config.return_value = chroma_config

        mock_embedder = MagicMock()
        mock_embedder.run.return_value = {"embedding": sample_embedding}
        mock_embedder_class.return_value = mock_embedder

        mock_db = MagicMock()
        mock_db.query.return_value = sample_leaf_documents
        mock_db_class.return_value = mock_db

        many_docs = [Document(content=f"Doc {i}") for i in range(10)]
        mock_merger = MagicMock()
        mock_merger.run.return_value = {"documents": many_docs}
        mock_merger_class.return_value = mock_merger

        pipeline = ChromaParentDocSearchPipeline("config.yaml", sample_parent_store)
        result = pipeline.search("test query", top_k=3)

        assert len(result["documents"]) == 3
        assert result["num_parents_returned"] == 3
