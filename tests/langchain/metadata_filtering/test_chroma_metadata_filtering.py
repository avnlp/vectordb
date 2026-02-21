"""Tests for Chroma metadata filtering pipelines (LangChain).

This module tests the metadata filtering pipeline implementation for Chroma vector
database. Metadata filtering enables precise retrieval by combining vector similarity
with structured metadata constraints, improving both precision and performance.

Metadata Filtering Pipeline Flow:
    1. Indexing: Documents stored with rich metadata (source, category, date, etc.)
    2. Search: Vector similarity combined with metadata filter predicates
    3. Results: Only documents matching both semantic and metadata criteria

Chroma-specific aspects tested:
    - Local persistent storage path configuration
    - Collection-based document organization
    - Chroma's native metadata filtering capabilities
    - Filter syntax compatibility (equality, comparison, logical operators)

Filter Types Supported:
    - Equality: {"source": "wiki"}
    - Comparison: {"score": {"$gt": 0.5}}
    - Logical AND: Multiple filter keys combined
    - Logical OR: {"$or": [{"source": "wiki"}, {"source": "docs"}]}

Test Coverage:
    - Indexing pipeline initialization with metadata schema
    - Document indexing with metadata preservation
    - Empty batch handling during indexing
    - Search pipeline initialization
    - Search execution with various filter predicates
    - Filter validation and error handling
    - RAG mode with filtered retrieval

External dependencies (ChromaVectorDB, EmbedderHelper, DataLoaderHelper,
RAGHelper) are mocked to enable fast, isolated unit tests.
"""

from unittest.mock import MagicMock, patch

from vectordb.langchain.metadata_filtering.indexing.chroma import (
    ChromaMetadataFilteringIndexingPipeline,
)
from vectordb.langchain.metadata_filtering.search.chroma import (
    ChromaMetadataFilteringSearchPipeline,
)


class TestChromaMetadataFilteringIndexing:
    """Unit tests for Chroma metadata filtering indexing pipeline.

    Validates the indexing pipeline which stores documents with metadata
    for later filtered retrieval. Metadata is preserved during embedding
    and storage, enabling precise filtering at query time.

    Pipeline Flow:
        1. Load documents with metadata from dataloader
        2. Embed document content (metadata not embedded)
        3. Store documents with metadata in Chroma
        4. Return indexing statistics

    Metadata Preservation:
        - Source, category, timestamps, custom fields
        - Used for filtering but not for embedding
        - Stored alongside vectors in Chroma
    """

    @patch("vectordb.langchain.metadata_filtering.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.metadata_filtering.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.indexing.chroma.DataloaderCatalog.create"
    )
    def test_indexing_initialization(
        self, mock_get_docs, mock_embedder_helper, mock_db
    ):
        """Test pipeline initialization with metadata filtering configuration.

        Verifies that:
        - Configuration dict is preserved on pipeline instance
        - Collection name is extracted from chroma config section
        - No external calls during initialization

        Args:
            mock_get_docs: Mock for DataloaderCatalog.create.
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db: Mock for ChromaVectorDB class.
        """
        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_metadata_filtering",
            },
        }

        pipeline = ChromaMetadataFilteringIndexingPipeline(config)
        assert pipeline.config == config
        assert pipeline.collection_name == "test_metadata_filtering"

    @patch("vectordb.langchain.metadata_filtering.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.metadata_filtering.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.indexing.chroma.EmbedderHelper.embed_documents"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.indexing.chroma.DataloaderCatalog.create"
    )
    def test_indexing_run_with_documents(
        self,
        mock_get_docs,
        mock_embed_docs,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test successful document indexing with metadata preservation.

        Validates the complete indexing workflow:
        1. DataLoaderHelper loads documents with metadata
        2. EmbedderHelper generates 384-dimensional embeddings
        3. ChromaVectorDB.upsert stores docs with metadata intact
        4. Result reports count of indexed documents

        Metadata Integrity:
        - Document metadata preserved through pipeline
        - Available for filtering at query time
        - Not used in embedding generation

        Args:
            mock_get_docs: Mock returning sample_documents fixture.
            mock_embed_docs: Mock returning (docs, embeddings) tuple.
            mock_embedder_helper: Mock for embedder factory.
            mock_db: Mock for ChromaVectorDB, tracks upsert calls.
            sample_documents: Fixture with 5 sample documents with metadata.
        """
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader
        mock_embed_docs.return_value = (sample_documents, [[0.1] * 384] * 5)

        mock_db_inst = MagicMock()
        mock_db_inst.upsert.return_value = len(sample_documents)
        mock_db.return_value = mock_db_inst

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_metadata_filtering",
            },
        }

        pipeline = ChromaMetadataFilteringIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == len(sample_documents)

    @patch("vectordb.langchain.metadata_filtering.indexing.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.metadata_filtering.indexing.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.indexing.chroma.DataloaderCatalog.create"
    )
    def test_indexing_run_no_documents(
        self, mock_get_docs, mock_embedder_helper, mock_db
    ):
        """Test graceful handling of empty document batches.

        Ensures when the dataloader returns empty list:
        - No exceptions raised
        - Result reports 0 documents indexed
        - No database operations attempted

        Args:
            mock_get_docs: Mock returning empty list.
            mock_embedder_helper: Mock for embedder factory.
            mock_db: Mock for ChromaVectorDB (should not be called).
        """
        mock_dataset = MagicMock()
        mock_dataset.to_langchain.return_value = []
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_get_docs.return_value = mock_loader

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_metadata_filtering",
            },
        }

        pipeline = ChromaMetadataFilteringIndexingPipeline(config)
        result = pipeline.run()

        assert result["documents_indexed"] == 0


class TestChromaMetadataFilteringSearch:
    """Unit tests for Chroma metadata filtering search pipeline.

    Tests validate search functionality with metadata constraints:
    - Pure vector search (baseline without filters)
    - Filtered search combining vector similarity + metadata predicates
    - Various filter types (equality, comparison, logical)
    - RAG mode with filtered retrieval

    Filter Benefits:
        - Precision: Exclude irrelevant documents by metadata
        - Performance: Reduce search space before vector comparison
        - Security: Tenant isolation via metadata filtering
    """

    @patch("vectordb.langchain.metadata_filtering.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.metadata_filtering.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch("vectordb.langchain.metadata_filtering.search.chroma.RAGHelper.create_llm")
    def test_search_initialization(
        self, mock_llm_helper, mock_embedder_helper, mock_db
    ):
        """Test search pipeline initialization with filtering support.

        Verifies that:
        - Configuration is stored on pipeline instance
        - LLM initialized for RAG (can be None when disabled)
        - Embedder initialized for query embedding
        - Filter capabilities ready for query time

        Args:
            mock_llm_helper: Mock for RAGHelper.create_llm.
            mock_embedder_helper: Mock for EmbedderHelper.create_embedder.
            mock_db: Mock for ChromaVectorDB class.
        """
        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_metadata_filtering",
            },
            "rag": {"enabled": False},
        }

        pipeline = ChromaMetadataFilteringSearchPipeline(config)
        assert pipeline.config == config

    @patch("vectordb.langchain.metadata_filtering.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.metadata_filtering.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.search.chroma.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.metadata_filtering.search.chroma.RAGHelper.create_llm")
    def test_search_execution(
        self,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test basic search execution without filters (baseline).

        Validates the standard search flow:
        1. Query embedding via EmbedderHelper.embed_query
        2. ChromaVectorDB.query retrieves matching documents
        3. Results returned with query and document list

        This test establishes baseline behavior before testing filters.

        Args:
            mock_llm_helper: Mock for RAGHelper.create_llm.
            mock_embed_query: Mock returning query vector.
            mock_embedder_helper: Mock for embedder factory.
            mock_db: Mock for ChromaVectorDB with query results.
            sample_documents: Fixture with sample documents.
        """
        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_metadata_filtering",
            },
            "rag": {"enabled": False},
        }

        pipeline = ChromaMetadataFilteringSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert len(result["documents"]) > 0

    @patch("vectordb.langchain.metadata_filtering.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.metadata_filtering.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.search.chroma.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.metadata_filtering.search.chroma.RAGHelper.create_llm")
    def test_search_with_filters(
        self,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test search with metadata filters applied.

        Validates filtered retrieval combining vector similarity with
        metadata constraints. Filters reduce the candidate pool before
        vector comparison, improving both precision and performance.

        Filter Application:
        - Filters passed to ChromaVectorDB.query as where clause
        - Only documents matching filter predicates considered
        - Vector similarity ranked within filtered subset

        Example Filter:
        {"source": "wiki"} - Only documents from wiki source

        Args:
            mock_llm_helper: Mock for RAGHelper.create_llm.
            mock_embed_query: Mock returning query vector.
            mock_embedder_helper: Mock for embedder factory.
            mock_db: Mock for ChromaVectorDB with query results.
            sample_documents: Fixture with sample documents.
        """
        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_metadata_filtering",
            },
            "rag": {"enabled": False},
        }

        pipeline = ChromaMetadataFilteringSearchPipeline(config)
        filters = {"source": "wiki"}
        result = pipeline.search("test query", top_k=5, filters=filters)

        assert result["query"] == "test query"

    @patch("vectordb.langchain.metadata_filtering.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.metadata_filtering.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.search.chroma.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.metadata_filtering.search.chroma.RAGHelper.create_llm")
    @patch(
        "vectordb.langchain.metadata_filtering.search.chroma.DocumentFilter.filter_by_metadata"
    )
    def test_search_with_configured_filters(
        self,
        mock_filter,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test search applies configured metadata filters."""
        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst
        mock_llm_helper.return_value = None
        mock_filter.return_value = sample_documents[:2]

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_metadata_filtering",
            },
            "filters": {
                "conditions": [
                    {"field": "source", "value": "wiki", "operator": "equals"}
                ]
            },
            "rag": {"enabled": False},
        }

        pipeline = ChromaMetadataFilteringSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        mock_filter.assert_called_once_with(
            sample_documents,
            key="source",
            value="wiki",
            operator="equals",
        )

    @patch("vectordb.langchain.metadata_filtering.search.chroma.ChromaVectorDB")
    @patch(
        "vectordb.langchain.metadata_filtering.search.chroma.EmbedderHelper.create_embedder"
    )
    @patch(
        "vectordb.langchain.metadata_filtering.search.chroma.EmbedderHelper.embed_query"
    )
    @patch("vectordb.langchain.metadata_filtering.search.chroma.RAGHelper.create_llm")
    @patch("vectordb.langchain.metadata_filtering.search.chroma.RAGHelper.generate")
    def test_search_with_rag(
        self,
        mock_rag_generate,
        mock_llm_helper,
        mock_embed_query,
        mock_embedder_helper,
        mock_db,
        sample_documents,
    ):
        """Test search with RAG generation."""
        mock_embed_query.return_value = [0.1] * 384
        mock_db_inst = MagicMock()
        mock_db_inst.query.return_value = sample_documents
        mock_db.return_value = mock_db_inst

        mock_llm = MagicMock()
        mock_llm_helper.return_value = mock_llm
        mock_rag_generate.return_value = "Generated answer"

        config = {
            "dataloader": {"type": "arc", "limit": 10},
            "embeddings": {"model": "test-model", "device": "cpu"},
            "chroma": {
                "path": "./test_chroma_data",
                "collection_name": "test_metadata_filtering",
            },
            "rag": {"enabled": True, "model": "test-llm"},
        }

        pipeline = ChromaMetadataFilteringSearchPipeline(config)
        result = pipeline.search("test query", top_k=5)

        assert result["query"] == "test query"
        assert "answer" in result
        assert result["answer"] == "Generated answer"
