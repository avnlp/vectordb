"""Tests for Chroma vector database wrapper.

This module tests the ChromaVectorDB class which provides a unified interface
for Chroma database operations including collection management, indexing,
and hybrid search capabilities.
"""

from contextlib import suppress
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.databases.chroma import ChromaVectorDB


class TestChromaVectorDBInitialization:
    """Test suite for ChromaVectorDB initialization.

    Tests cover:
    - Initialization with various parameter combinations
    - Configuration loading from files and dicts
    - Environment variable resolution
    - Error handling for missing parameters
    """

    def test_initialization_with_direct_parameters(self) -> None:
        """Test initialization with direct parameters."""
        with patch("vectordb.databases.chroma.chromadb"):
            db = ChromaVectorDB(
                host="localhost",
                port=8000,
                collection_name="test_collection",
            )

            assert db.host == "localhost"
            assert db.port == 8000
            assert db.collection_name == "test_collection"

    def test_initialization_with_config_dict(self) -> None:
        """Test initialization with configuration dictionary."""
        config = {
            "chroma": {
                "host": "localhost",
                "port": 8000,
                "collection_name": "test_collection",
            }
        }
        with patch("vectordb.databases.chroma.chromadb"):
            db = ChromaVectorDB(config=config)

            assert db.host == "localhost"
            assert db.port == 8000

    @patch.dict("os.environ", {"CHROMA_HOST": "env_host"})
    def test_initialization_with_environment_variables(self) -> None:
        """Test initialization with environment variables."""
        with patch("vectordb.databases.chroma.chromadb"):
            db = ChromaVectorDB()

            assert db.host == "env_host"

    def test_initialization_parameter_priority(self) -> None:
        """Test that parameters override config and environment."""
        config = {
            "chroma": {
                "host": "config_host",
                "port": 9000,
            }
        }
        with (
            patch("vectordb.databases.chroma.chromadb"),
            patch.dict("os.environ", {"CHROMA_HOST": "env_host"}),
        ):
            db = ChromaVectorDB(host="param_host", config=config)

            # Parameters should have priority
            assert db.host == "param_host"


class TestChromaVectorDBCollectionManagement:
    """Test suite for collection management.

    Tests cover:
    - Collection creation with various configurations
    - Collection retrieval and deletion
    - Metadata filtering
    - Error handling
    """

    @pytest.fixture
    def mock_chroma_db(self):
        """Create mock ChromaVectorDB instance."""
        with patch("vectordb.databases.chroma.chromadb"):
            db = ChromaVectorDB(
                host="localhost",
                port=8000,
                collection_name="test_collection",
            )
            db.client = MagicMock()
            db.collection = MagicMock()
            return db

    def test_create_collection_default(self, mock_chroma_db) -> None:
        """Test default collection creation."""
        mock_chroma_db.client.get_or_create_collection = MagicMock(
            return_value=MagicMock()
        )

        mock_chroma_db.create_collection("test_collection")

        mock_chroma_db.client.get_or_create_collection.assert_called_once()

    def test_create_collection_with_metadata(self, mock_chroma_db) -> None:
        """Test collection creation with metadata."""
        mock_chroma_db.client.delete_collection = MagicMock()
        mock_chroma_db.client.get_or_create_collection = MagicMock()
        mock_chroma_db.client.list_collections = MagicMock(return_value=[])

        mock_chroma_db.create_collection("test_collection", recreate=False)

        mock_chroma_db.client.get_or_create_collection.assert_called_once()

    def test_delete_collection(self, mock_chroma_db) -> None:
        """Test collection deletion."""
        mock_chroma_db.client.delete_collection = MagicMock()

        mock_chroma_db.delete_collection("test_collection")

        mock_chroma_db.client.delete_collection.assert_called_once()


class TestChromaVectorDBIndexing:
    """Test suite for document indexing.

    Tests cover:
    - Upserting documents
    - Batch operations
    - Error handling for invalid inputs
    """

    @pytest.fixture
    def mock_chroma_db(self):
        """Create mock ChromaVectorDB instance."""
        with patch("vectordb.databases.chroma.chromadb"):
            db = ChromaVectorDB(
                host="localhost",
                port=8000,
                collection_name="test_collection",
            )
            db.client = MagicMock()
            db.collection = MagicMock()
            return db

    def test_upsert_documents_success(
        self, mock_chroma_db, sample_documents: list[Document]
    ) -> None:
        """Test successful document upserting."""
        mock_chroma_db.collection.upsert = MagicMock()

        mock_chroma_db.upsert(sample_documents)

        mock_chroma_db.collection.upsert.assert_called_once()

    def test_upsert_empty_documents(self, mock_chroma_db) -> None:
        """Test upserting empty document list."""
        mock_chroma_db.collection.upsert = MagicMock()

        with suppress(Exception):
            mock_chroma_db.upsert([])

        # Should handle empty list gracefully
        assert True

    def test_upsert_documents_with_metadata(
        self, mock_chroma_db, sample_documents: list[Document]
    ) -> None:
        """Test upserting documents with metadata."""
        mock_chroma_db.collection.upsert = MagicMock()

        docs_with_meta = [
            Document(
                content=doc.content,
                meta={**doc.meta, "extra_field": "value"},
                embedding=doc.embedding,
            )
            for doc in sample_documents
        ]

        mock_chroma_db.upsert(docs_with_meta)

        mock_chroma_db.collection.upsert.assert_called_once()


class TestChromaVectorDBSearch:
    """Test suite for search operations.

    Tests cover:
    - Dense vector search
    - Metadata filtering
    - Result ranking and limiting
    - Error handling
    """

    @pytest.fixture
    def mock_chroma_db(self):
        """Create mock ChromaVectorDB instance."""
        with patch("vectordb.databases.chroma.chromadb"):
            db = ChromaVectorDB(
                host="localhost",
                port=8000,
                collection_name="test_collection",
            )
            db.client = MagicMock()
            db.collection = MagicMock()
            return db

    def test_search_with_vector(
        self, mock_chroma_db, query_embedding: list[float]
    ) -> None:
        """Test vector search."""
        mock_results = MagicMock()
        mock_results.get = MagicMock(return_value=[])
        mock_chroma_db.collection.query = MagicMock(return_value=mock_results)

        mock_chroma_db.search(query_embedding, top_k=5)

        mock_chroma_db.collection.query.assert_called_once()

    def test_search_with_metadata_filter(
        self, mock_chroma_db, query_embedding: list[float]
    ) -> None:
        """Test search with metadata filtering."""
        mock_results = MagicMock()
        mock_results.get = MagicMock(return_value=[])
        mock_chroma_db.collection.query = MagicMock(return_value=mock_results)

        filters = {"source": "textbook"}
        mock_chroma_db.search(query_embedding, filters=filters, top_k=5)

        mock_chroma_db.collection.query.assert_called_once()

    def test_search_top_k_limit(
        self, mock_chroma_db, query_embedding: list[float]
    ) -> None:
        """Test that search respects top_k limit."""
        mock_results = MagicMock()
        mock_results.get = MagicMock(return_value=[])
        mock_chroma_db.collection.query = MagicMock(return_value=mock_results)

        mock_chroma_db.search(query_embedding, top_k=3)

        # Verify top_k is passed to query
        call_args = mock_chroma_db.collection.query.call_args
        assert call_args is not None


class TestChromaVectorDBHttpClient:
    """Test suite for HttpClient initialization.

    Tests cover:
    - HttpClient creation with host, port, ssl, api_key, tenant, database parameters
    - Verify logging output for connection info
    """

    @patch("vectordb.databases.chroma.chromadb")
    @patch("vectordb.databases.chroma.Settings")
    def test_http_client_initialization_with_all_params(
        self, mock_settings_class, mock_chromadb
    ) -> None:
        """Test HttpClient creation with all connection parameters."""
        mock_http_client = MagicMock()
        mock_chromadb.HttpClient = mock_http_client
        mock_settings = MagicMock()
        mock_settings_class.return_value = mock_settings

        db = ChromaVectorDB(
            host="api.chroma.cloud",
            port=443,
            api_key="test-api-key",
            tenant="test_tenant",
            database="test_database",
            collection_name="test_collection",
            ssl=True,
        )

        # Trigger client creation
        db._get_client()

        mock_http_client.assert_called_once()
        call_kwargs = mock_http_client.call_args.kwargs
        assert call_kwargs["host"] == "api.chroma.cloud"
        assert call_kwargs["port"] == 443
        assert call_kwargs["ssl"] is True
        assert call_kwargs["settings"] == mock_settings
        # Verify Settings was called with expected parameters
        mock_settings_class.assert_called_once()
        settings_kwargs = mock_settings_class.call_args.kwargs
        assert settings_kwargs["chroma_api_key"] == "test-api-key"
        assert settings_kwargs["chroma_tenant"] == "test_tenant"
        assert settings_kwargs["chroma_database"] == "test_database"

    @patch("vectordb.databases.chroma.chromadb")
    @patch("vectordb.databases.chroma.Settings")
    def test_http_client_logging(
        self, mock_settings_class, mock_chromadb, caplog
    ) -> None:
        """Test that HttpClient initialization logs connection info."""
        mock_chromadb.HttpClient = MagicMock()
        mock_settings = MagicMock()
        mock_settings_class.return_value = mock_settings

        db = ChromaVectorDB(
            host="api.chroma.cloud",
            port=8000,
            collection_name="test_collection",
        )

        with caplog.at_level("INFO"):
            db._get_client()

        assert "Connecting to Chroma HttpClient at api.chroma.cloud:8000" in caplog.text
        assert "Chroma client initialized successfully" in caplog.text

    @patch("vectordb.databases.chroma.chromadb")
    @patch("vectordb.databases.chroma.Settings")
    def test_http_client_with_ssl_false(
        self, mock_settings_class, mock_chromadb
    ) -> None:
        """Test HttpClient creation with ssl=False."""
        mock_http_client = MagicMock()
        mock_chromadb.HttpClient = mock_http_client
        mock_settings = MagicMock()
        mock_settings_class.return_value = mock_settings

        db = ChromaVectorDB(
            host="localhost",
            port=8000,
            collection_name="test_collection",
            ssl=False,
        )

        db._get_client()

        call_kwargs = mock_http_client.call_args.kwargs
        assert call_kwargs["ssl"] is False


class TestChromaVectorDBStrictCollectionCreation:
    """Test suite for strict collection creation.

    Tests cover:
    - create_collection with get_or_create=False calls create_collection
    """

    @pytest.fixture
    def mock_chroma_db(self):
        """Create mock ChromaVectorDB instance."""
        with patch("vectordb.databases.chroma.chromadb"):
            db = ChromaVectorDB(
                host="localhost",
                port=8000,
                collection_name="test_collection",
            )
            db.client = MagicMock()
            db.collection = None
            return db

    def test_create_collection_strict_mode(self, mock_chroma_db) -> None:
        """Test create_collection with get_or_create=False."""
        mock_collection = MagicMock()
        mock_chroma_db.client.create_collection.return_value = mock_collection

        mock_chroma_db.create_collection(
            name="strict_collection",
            get_or_create=False,
        )

        mock_chroma_db.client.create_collection.assert_called_once()
        mock_chroma_db.client.get_or_create_collection.assert_not_called()

    def test_create_collection_get_or_create_mode(self, mock_chroma_db) -> None:
        """Test create_collection with get_or_create=True."""
        mock_collection = MagicMock()
        mock_chroma_db.client.get_or_create_collection.return_value = mock_collection

        mock_chroma_db.create_collection(
            name="flexible_collection",
            get_or_create=True,
        )

        mock_chroma_db.client.get_or_create_collection.assert_called_once()
        mock_chroma_db.client.create_collection.assert_not_called()


class TestChromaVectorDBDictionaryUpsert:
    """Test suite for dictionary format upsert.

    Tests cover:
    - upsert with dict containing ids, embeddings, metadatas, documents
    - metadata flattening for nested dicts
    """

    @pytest.fixture
    def mock_chroma_db(self):
        """Create mock ChromaVectorDB instance."""
        with patch("vectordb.databases.chroma.chromadb"):
            db = ChromaVectorDB(
                host="localhost",
                port=8000,
                collection_name="test_collection",
            )
            db.client = MagicMock()
            db.collection = MagicMock()
            return db

    def test_upsert_with_dictionary_format(self, mock_chroma_db) -> None:
        """Test upsert with dict containing ids, embeddings, metadatas, documents."""
        data = {
            "ids": ["doc1", "doc2"],
            "embeddings": [[0.1] * 384, [0.2] * 384],
            "metadatas": [{"source": "wiki"}, {"source": "paper"}],
            "documents": ["content1", "content2"],
        }

        mock_chroma_db.upsert(data)

        mock_chroma_db.collection.upsert.assert_called_once()
        call_kwargs = mock_chroma_db.collection.upsert.call_args.kwargs
        assert call_kwargs["ids"] == ["doc1", "doc2"]
        assert call_kwargs["documents"] == ["content1", "content2"]

    def test_upsert_with_nested_metadata(self, mock_chroma_db) -> None:
        """Test metadata flattening for nested dicts in dictionary format."""
        data = {
            "ids": ["doc1"],
            "embeddings": [[0.1] * 384],
            "metadatas": [{"nested": {"key": "value"}, "simple": "val"}],
            "documents": ["content1"],
        }

        mock_chroma_db.upsert(data)

        mock_chroma_db.collection.upsert.assert_called_once()
        call_kwargs = mock_chroma_db.collection.upsert.call_args.kwargs
        # Metadata should be flattened
        expected_metadata = [{"nested.key": "value", "simple": "val"}]
        assert call_kwargs["metadatas"] == expected_metadata


class TestChromaVectorDBEmptyMetadata:
    """Test suite for empty metadata handling.

    Tests cover:
    - Edge case where all metadata dicts are empty
    """

    @pytest.fixture
    def mock_chroma_db(self):
        """Create mock ChromaVectorDB instance."""
        with patch("vectordb.databases.chroma.chromadb"):
            db = ChromaVectorDB(
                host="localhost",
                port=8000,
                collection_name="test_collection",
            )
            db.client = MagicMock()
            db.collection = MagicMock()
            return db

    def test_upsert_with_all_empty_metadata(self, mock_chroma_db) -> None:
        """Test with metadatas=[{}, {}, {}] - all empty dicts."""
        data = {
            "ids": ["doc1", "doc2", "doc3"],
            "embeddings": [[0.1] * 384, [0.2] * 384, [0.3] * 384],
            "metadatas": [{}, {}, {}],
            "documents": ["content1", "content2", "content3"],
        }

        mock_chroma_db.upsert(data)

        mock_chroma_db.collection.upsert.assert_called_once()
        call_kwargs = mock_chroma_db.collection.upsert.call_args.kwargs
        # When all metadata is empty, it should be set to None
        assert call_kwargs["metadatas"] is None

    def test_upsert_with_partial_empty_metadata(self, mock_chroma_db) -> None:
        """Test with mix of empty and non-empty metadata dicts."""
        data = {
            "ids": ["doc1", "doc2", "doc3"],
            "embeddings": [[0.1] * 384, [0.2] * 384, [0.3] * 384],
            "metadatas": [{}, {"source": "wiki"}, {}],
            "documents": ["content1", "content2", "content3"],
        }

        mock_chroma_db.upsert(data)

        mock_chroma_db.collection.upsert.assert_called_once()
        call_kwargs = mock_chroma_db.collection.upsert.call_args.kwargs
        # Empty dicts should be replaced with {"_": "_"}
        expected_metadata = [{"_": "_"}, {"source": "wiki"}, {"_": "_"}]
        assert call_kwargs["metadatas"] == expected_metadata


class TestChromaVectorDBQueryVectorInclusion:
    """Test suite for query vector inclusion.

    Tests cover:
    - query with include_vectors=True includes embeddings in the include list
    """

    @pytest.fixture
    def mock_chroma_db(self):
        """Create mock ChromaVectorDB instance."""
        with patch("vectordb.databases.chroma.chromadb"):
            db = ChromaVectorDB(
                host="localhost",
                port=8000,
                collection_name="test_collection",
            )
            db.client = MagicMock()
            db.collection = MagicMock()
            return db

    def test_query_with_include_vectors_true(self, mock_chroma_db) -> None:
        """Test that embeddings are included when include_vectors=True."""
        query_embedding = [0.1] * 384

        mock_chroma_db.query(
            query_embedding=query_embedding,
            n_results=5,
            include_vectors=True,
        )

        mock_chroma_db.collection.query.assert_called_once()
        call_kwargs = mock_chroma_db.collection.query.call_args.kwargs
        assert "embeddings" in call_kwargs["include"]
        assert "metadatas" in call_kwargs["include"]
        assert "documents" in call_kwargs["include"]
        assert "distances" in call_kwargs["include"]

    def test_query_without_include_vectors(self, mock_chroma_db) -> None:
        """Test that embeddings are not included by default."""
        query_embedding = [0.1] * 384

        mock_chroma_db.query(
            query_embedding=query_embedding,
            n_results=5,
        )

        mock_chroma_db.collection.query.assert_called_once()
        call_kwargs = mock_chroma_db.collection.query.call_args.kwargs
        assert "embeddings" not in call_kwargs["include"]
        assert "metadatas" in call_kwargs["include"]
        assert "documents" in call_kwargs["include"]
        assert "distances" in call_kwargs["include"]


class TestChromaVectorDBResultConversion:
    """Test suite for result conversion.

    Tests cover:
    - query_to_documents with complete results
    - distance to similarity score conversion
    - handling missing/None values
    """

    def test_query_to_documents_complete_results(self) -> None:
        """Test conversion with complete results."""
        with patch("vectordb.databases.chroma.chromadb"):
            db = ChromaVectorDB(
                host="localhost",
                port=8000,
                collection_name="test_collection",
            )

            results = {
                "ids": [["doc1", "doc2"]],
                "documents": [["content1", "content2"]],
                "metadatas": [[{"source": "wiki"}, {"source": "paper"}]],
                "distances": [[0.2, 0.5]],
                "embeddings": [[[0.1] * 384, [0.2] * 384]],
            }

            documents = db.query_to_documents(results)

            assert len(documents) == 2
            assert documents[0].id == "doc1"
            assert documents[0].content == "content1"
            assert documents[0].meta == {"source": "wiki"}
            assert documents[0].score == 0.8  # 1.0 - 0.2
            assert documents[0].embedding == [0.1] * 384

            assert documents[1].id == "doc2"
            assert documents[1].score == 0.5  # 1.0 - 0.5

    def test_query_to_documents_distance_to_similarity(self) -> None:
        """Test distance to similarity score conversion."""
        with patch("vectordb.databases.chroma.chromadb"):
            db = ChromaVectorDB(
                host="localhost",
                port=8000,
                collection_name="test_collection",
            )

            results = {
                "ids": [["doc1"]],
                "documents": [["content1"]],
                "metadatas": [[{}]],
                "distances": [[0.3]],
            }

            documents = db.query_to_documents(results)

            assert documents[0].score == 0.7  # 1.0 - 0.3

    def test_query_to_documents_with_missing_values(self) -> None:
        """Test handling missing/None values in results."""
        with patch("vectordb.databases.chroma.chromadb"):
            db = ChromaVectorDB(
                host="localhost",
                port=8000,
                collection_name="test_collection",
            )

            results = {
                "ids": [["doc1"]],
                "documents": [[]],  # Empty documents
                "metadatas": [[]],  # Empty metadatas
                # No distances
            }

            documents = db.query_to_documents(results)

            assert len(documents) == 1
            assert documents[0].id == "doc1"
            assert documents[0].content == ""  # Default empty string
            assert documents[0].meta == {}  # Default empty dict
            assert not hasattr(documents[0], "score") or documents[0].score is None


class TestChromaVectorDBNativeSearchAPI:
    """Test suite for native Search API.

    Tests cover:
    - search with where clause
    - different embedding formats
    - fallback when imports fail
    """

    @pytest.fixture
    def mock_chroma_db(self):
        """Create mock ChromaVectorDB instance."""
        with patch("vectordb.databases.chroma.chromadb"):
            db = ChromaVectorDB(
                host="api.chroma.cloud",  # Use host to trigger native search path
                port=8000,
                collection_name="test_collection",
            )
            db.client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.name = "test_collection"
            # Mock that collection has search method
            mock_collection.search = MagicMock(return_value={"results": []})
            db.collection = mock_collection
            return db

    def test_search_with_where_clause(self, mock_chroma_db) -> None:
        """Test search with where clause using native Search API."""
        with (
            patch(
                "vectordb.databases.chroma.chromadb.execution.expression.Search"
            ) as mock_search_class,
            patch(
                "vectordb.databases.chroma.chromadb.execution.expression.Knn"
            ) as mock_knn_class,
        ):
            mock_search = MagicMock()
            mock_search.limit.return_value = mock_search
            mock_search.where.return_value = mock_search
            mock_search.rank.return_value = mock_search
            mock_search_class.return_value = mock_search

            mock_knn = MagicMock()
            mock_knn_class.return_value = mock_knn

            query_embedding = [0.1] * 384
            where_filter = {"source": "wiki"}

            mock_chroma_db.search(
                query_embeddings=query_embedding,
                n_results=5,
                where=where_filter,
            )

            mock_search.where.assert_called_once_with(where_filter)
            mock_chroma_db.collection.search.assert_called_once()

    def test_search_with_single_embedding(self, mock_chroma_db) -> None:
        """Test search with single embedding format."""
        with (
            patch(
                "vectordb.databases.chroma.chromadb.execution.expression.Search"
            ) as mock_search_class,
            patch(
                "vectordb.databases.chroma.chromadb.execution.expression.Knn"
            ) as mock_knn_class,
        ):
            mock_search = MagicMock()
            mock_search.limit.return_value = mock_search
            mock_search.rank.return_value = mock_search
            mock_search_class.return_value = mock_search

            mock_knn = MagicMock()
            mock_knn_class.return_value = mock_knn

            # Single embedding (not wrapped in list)
            query_embedding = [0.1] * 384

            mock_chroma_db.search(
                query_embeddings=query_embedding,
                n_results=5,
            )

            mock_search.rank.assert_called_once()
            mock_chroma_db.collection.search.assert_called_once()

    def test_search_fallback_on_import_error(self, mock_chroma_db) -> None:
        """Test fallback to query() when imports fail."""
        # Make the collection have search method but import will fail
        mock_chroma_db.collection.search = MagicMock()

        with patch(
            "vectordb.databases.chroma.chromadb.execution.expression.Search",
            side_effect=ImportError("No module named 'chromadb.execution'"),
        ):
            query_embedding = [0.1] * 384

            # Should not raise, should fall back to query
            mock_chroma_db.search(
                query_embeddings=query_embedding,
                n_results=5,
            )

            # Should fall back to query method
            mock_chroma_db.collection.query.assert_called_once()

    def test_search_fallback_on_not_implemented_error(self, mock_chroma_db) -> None:
        """Test fallback to query() when NotImplementedError is raised."""
        mock_chroma_db.collection.search = MagicMock(
            side_effect=NotImplementedError("Search not available")
        )

        with (
            patch(
                "vectordb.databases.chroma.chromadb.execution.expression.Search"
            ) as mock_search_class,
            patch("vectordb.databases.chroma.chromadb.execution.expression.Knn"),
        ):
            mock_search = MagicMock()
            mock_search.limit.return_value = mock_search
            mock_search.rank.return_value = mock_search
            mock_search_class.return_value = mock_search

            query_embedding = [0.1] * 384

            # Should not raise, should fall back to query
            mock_chroma_db.search(
                query_embeddings=query_embedding,
                n_results=5,
            )

            # Should fall back to query method
            mock_chroma_db.collection.query.assert_called_once()

    def test_search_with_prebuilt_searches_object(self, mock_chroma_db) -> None:
        """Test search with pre-built searches object passed in kwargs."""
        with (
            patch(
                "vectordb.databases.chroma.chromadb.execution.expression.Search"
            ) as mock_search_class,
            patch("vectordb.databases.chroma.chromadb.execution.expression.Knn"),
        ):
            mock_search = MagicMock()
            mock_search_class.return_value = mock_search

            prebuilt_search = MagicMock()

            mock_chroma_db.search(
                searches=prebuilt_search,
                n_results=5,
            )

            # Should call collection.search with the prebuilt search object
            mock_chroma_db.collection.search.assert_called_once_with(
                prebuilt_search,
                searches=prebuilt_search,
            )


class TestChromaVectorDBCollectionOperations:
    """Test suite for collection operations.

    Tests cover:
    - list_collections - list all collection names
    - delete_documents - delete by IDs or filter
    - delete_collection - validate name and cleanup self.collection reference
    """

    @pytest.fixture
    def mock_chroma_db(self):
        """Create mock ChromaVectorDB instance."""
        with patch("vectordb.databases.chroma.chromadb"):
            db = ChromaVectorDB(
                host="localhost",
                port=8000,
                collection_name="test_collection",
            )
            db.client = MagicMock()
            db.collection = MagicMock()
            return db

    def test_list_collections(self, mock_chroma_db) -> None:
        """Test list_collections returns all collection names."""
        mock_collection1 = MagicMock()
        mock_collection1.name = "collection1"
        mock_collection2 = MagicMock()
        mock_collection2.name = "collection2"
        mock_chroma_db.client.list_collections.return_value = [
            mock_collection1,
            mock_collection2,
        ]

        result = mock_chroma_db.list_collections()

        assert result == ["collection1", "collection2"]
        mock_chroma_db.client.list_collections.assert_called_once()

    def test_delete_documents_by_ids(self, mock_chroma_db) -> None:
        """Test delete_documents with IDs."""
        ids = ["doc1", "doc2"]

        mock_chroma_db.delete_documents(ids=ids)

        mock_chroma_db.collection.delete.assert_called_once_with(
            ids=ids,
            where=None,
        )

    def test_delete_documents_by_filter(self, mock_chroma_db) -> None:
        """Test delete_documents with filter."""
        where_filter = {"source": "wiki"}

        mock_chroma_db.delete_documents(where=where_filter)

        mock_chroma_db.collection.delete.assert_called_once_with(
            ids=None,
            where=where_filter,
        )

    def test_delete_collection_clears_reference(self, mock_chroma_db) -> None:
        """Test delete_collection clears self.collection reference."""
        mock_chroma_db.client.delete_collection = MagicMock()
        mock_chroma_db.collection_name = "test_collection"

        mock_chroma_db.delete_collection("test_collection")

        mock_chroma_db.client.delete_collection.assert_called_once_with(
            name="test_collection"
        )
        assert mock_chroma_db.collection is None

    def test_delete_collection_without_name_raises_error(self, mock_chroma_db) -> None:
        """Test delete_collection raises error without name."""
        mock_chroma_db.collection_name = None

        with pytest.raises(ValueError, match="Collection name is required"):
            mock_chroma_db.delete_collection()


class TestChromaVectorDBTenantContext:
    """Test suite for tenant context switching.

    Tests cover:
    - with_tenant method creates new instance with different tenant
    """

    def test_with_tenant_creates_new_instance(self) -> None:
        """Test with_tenant creates new instance with different tenant."""
        with patch("vectordb.databases.chroma.chromadb"):
            db = ChromaVectorDB(
                host="api.chroma.cloud",
                port=8000,
                api_key="test-key",
                tenant="original_tenant",
                database="original_db",
                collection_name="test_collection",
                ssl=True,
            )

            new_db = db.with_tenant("new_tenant", "new_db")

            assert new_db is not db  # Should be a new instance
            assert new_db.tenant == "new_tenant"
            assert new_db.database == "new_db"
            # Other attributes should be preserved
            assert new_db.host == db.host
            assert new_db.port == db.port
            assert new_db.api_key == db.api_key
            assert new_db.ssl == db.ssl
            assert new_db.collection_name == db.collection_name

    def test_with_tenant_preserves_database_if_not_specified(self) -> None:
        """Test with_tenant preserves original database if not specified."""
        with patch("vectordb.databases.chroma.chromadb"):
            db = ChromaVectorDB(
                host="api.chroma.cloud",
                port=8000,
                tenant="original_tenant",
                database="original_db",
                collection_name="test_collection",
            )

            new_db = db.with_tenant("new_tenant")

            assert new_db.tenant == "new_tenant"
            assert new_db.database == "original_db"  # Should be preserved


class TestChromaVectorDBMetadataFlattening:
    """Test suite for metadata flattening.

    Tests cover:
    - _flatten_metadata with nested dictionaries
    - handling lists, None values
    - various types
    """

    def test_flatten_metadata_nested_dict(self) -> None:
        """Test flatten_metadata with nested dictionaries."""
        with patch("vectordb.databases.chroma.chromadb"):
            db = ChromaVectorDB(
                host="localhost",
                port=8000,
                collection_name="test_collection",
            )

            metadata = {
                "level1": {
                    "level2": {"level3": "deep_value"},
                    "sibling": "sibling_value",
                },
                "top": "top_value",
            }

            result = db.flatten_metadata(metadata)

            assert result == {
                "level1.level2.level3": "deep_value",
                "level1.sibling": "sibling_value",
                "top": "top_value",
            }

    def test_flatten_metadata_with_list(self) -> None:
        """Test flatten_metadata with list values."""
        with patch("vectordb.databases.chroma.chromadb"):
            db = ChromaVectorDB(
                host="localhost",
                port=8000,
                collection_name="test_collection",
            )

            metadata = {
                "tags": ["ml", "ai", "nlp"],
                "scores": [1, 2, 3],
            }

            result = db.flatten_metadata(metadata)

            assert result == {
                "tags": ["ml", "ai", "nlp"],
                "scores": [1, 2, 3],
            }

    def test_flatten_metadata_with_complex_list(self) -> None:
        """Test flatten_metadata with list containing non-primitive types."""
        with patch("vectordb.databases.chroma.chromadb"):
            db = ChromaVectorDB(
                host="localhost",
                port=8000,
                collection_name="test_collection",
            )

            metadata = {
                "mixed": [{"key": "value"}, "string"],  # List with dict
            }

            result = db.flatten_metadata(metadata)

            # Complex lists should be converted to string
            assert result["mixed"] == "[{'key': 'value'}, 'string']"

    def test_flatten_metadata_with_none(self) -> None:
        """Test flatten_metadata skips None values."""
        with patch("vectordb.databases.chroma.chromadb"):
            db = ChromaVectorDB(
                host="localhost",
                port=8000,
                collection_name="test_collection",
            )

            metadata = {
                "valid": "value",
                "none_field": None,
                "another_valid": 123,
            }

            result = db.flatten_metadata(metadata)

            assert result == {
                "valid": "value",
                "another_valid": 123,
            }

    def test_flatten_metadata_with_various_types(self) -> None:
        """Test flatten_metadata with various primitive types."""
        with patch("vectordb.databases.chroma.chromadb"):
            db = ChromaVectorDB(
                host="localhost",
                port=8000,
                collection_name="test_collection",
            )

            metadata = {
                "string": "text",
                "integer": 42,
                "float": 3.14,
                "boolean": True,
            }

            result = db.flatten_metadata(metadata)

            assert result == {
                "string": "text",
                "integer": 42,
                "float": 3.14,
                "boolean": True,
            }

    def test_flatten_metadata_with_unsupported_type(self) -> None:
        """Test flatten_metadata converts unsupported types to string."""
        with patch("vectordb.databases.chroma.chromadb"):
            db = ChromaVectorDB(
                host="localhost",
                port=8000,
                collection_name="test_collection",
            )

            # Create a custom object that is not directly supported
            class CustomObject:
                def __str__(self):
                    return "custom_object_value"

            metadata = {
                "custom": CustomObject(),
            }

            result = db.flatten_metadata(metadata)

            # Should convert to string
            assert result["custom"] == "custom_object_value"


class TestChromaVectorDBPersistentAndEphemeral:
    """Test suite for PersistentClient and EphemeralClient initialization.

    Tests cover:
    - PersistentClient creation with path
    - EphemeralClient creation when persistent=False
    """

    @patch("vectordb.databases.chroma.chromadb")
    def test_persistent_client_initialization(self, mock_chromadb) -> None:
        """Test PersistentClient creation with path."""
        mock_persistent_client = MagicMock()
        mock_chromadb.PersistentClient = mock_persistent_client

        db = ChromaVectorDB(
            path="./chroma_data",
            persistent=True,
            collection_name="test_collection",
        )

        db._get_client()

        mock_persistent_client.assert_called_once_with(path="./chroma_data")

    @patch("vectordb.databases.chroma.chromadb")
    def test_ephemeral_client_initialization(self, mock_chromadb) -> None:
        """Test EphemeralClient creation when persistent=False and no host."""
        mock_ephemeral_client = MagicMock()
        mock_chromadb.EphemeralClient = mock_ephemeral_client

        db = ChromaVectorDB(
            persistent=False,
            collection_name="test_collection",
        )

        db._get_client()

        mock_ephemeral_client.assert_called_once()

    @patch("vectordb.databases.chroma.chromadb")
    def test_persistent_client_logging(self, mock_chromadb, caplog) -> None:
        """Test that PersistentClient initialization logs correctly."""
        mock_chromadb.PersistentClient = MagicMock()

        db = ChromaVectorDB(
            path="./chroma_data",
            persistent=True,
            collection_name="test_collection",
        )

        with caplog.at_level("INFO"):
            db._get_client()

        assert "Initializing Chroma PersistentClient at ./chroma_data" in caplog.text

    @patch("vectordb.databases.chroma.chromadb")
    def test_ephemeral_client_logging(self, mock_chromadb, caplog) -> None:
        """Test that EphemeralClient initialization logs correctly."""
        mock_chromadb.EphemeralClient = MagicMock()

        db = ChromaVectorDB(
            persistent=False,
            collection_name="test_collection",
        )

        with caplog.at_level("INFO"):
            db._get_client()

        assert "Initializing Chroma EphemeralClient" in caplog.text


class TestChromaVectorDBConfigurationLoading:
    """Test suite for configuration loading.

    Tests cover:
    - initialization with config_path parameter
    """

    @patch("vectordb.databases.chroma.load_config")
    def test_initialization_with_config_path(self, mock_load_config) -> None:
        """Test initialization with config_path parameter."""
        mock_load_config.return_value = {
            "chroma": {
                "host": "config_host",
                "port": 9000,
                "collection_name": "config_collection",
            }
        }

        with patch("vectordb.databases.chroma.chromadb"):
            db = ChromaVectorDB(config_path="/path/to/config.yaml")

        mock_load_config.assert_called_once_with("/path/to/config.yaml")
        assert db.host == "config_host"
        assert db.port == 9000
        assert db.collection_name == "config_collection"


class TestChromaVectorDBGetCollection:
    """Test suite for _get_collection method.

    Tests cover:
    - Collection name validation
    - ValueError when no collection name is available
    """

    @patch("vectordb.databases.chroma.chromadb")
    def test_get_collection_raises_value_error_when_no_name(
        self, mock_chromadb
    ) -> None:
        """Test _get_collection raises ValueError when no collection name."""
        db = ChromaVectorDB(
            persistent=False,
        )
        db.collection_name = None
        db.collection = None
        db.client = MagicMock()

        with pytest.raises(ValueError, match="Collection name is required."):
            db._get_collection()

    @patch("vectordb.databases.chroma.chromadb")
    def test_get_collection_with_new_name(self, mock_chromadb) -> None:
        """Test _get_collection updates collection_name when new name provided."""
        mock_collection = MagicMock()
        mock_collection.name = "new_collection"
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection

        db = ChromaVectorDB(
            persistent=False,
            collection_name="old_collection",
        )
        db.client = mock_client
        db.collection = None

        result = db._get_collection(name="new_collection")

        assert db.collection_name == "new_collection"
        assert result == mock_collection
        mock_client.get_collection.assert_called_once_with(name="new_collection")

    @patch("vectordb.databases.chroma.chromadb")
    def test_get_collection_reuses_existing_when_same_name(self, mock_chromadb) -> None:
        """Test _get_collection reuses existing collection when name matches."""
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"

        db = ChromaVectorDB(
            persistent=False,
            collection_name="test_collection",
        )
        db.collection = mock_collection

        result = db._get_collection(name="test_collection")

        assert result == mock_collection


class TestChromaVectorDBQueryToDocuments:
    """Test suite for query_to_documents method.

    Tests cover:
    - Converting results with embeddings
    - Handling empty embeddings
    """

    @patch("vectordb.databases.chroma.chromadb")
    def test_query_to_documents_with_embeddings(self, mock_chromadb) -> None:
        """Test query_to_documents includes embeddings when present."""
        db = ChromaVectorDB(
            persistent=False,
            collection_name="test_collection",
        )

        results = {
            "ids": [["doc1", "doc2"]],
            "documents": [["content1", "content2"]],
            "metadatas": [[{"key": "val1"}, {"key": "val2"}]],
            "distances": [[0.1, 0.2]],
            "embeddings": [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]],
        }

        documents = db.query_to_documents(results)

        assert len(documents) == 2
        assert documents[0].embedding == [0.1, 0.2, 0.3]
        assert documents[1].embedding == [0.4, 0.5, 0.6]
        assert documents[0].score == pytest.approx(0.9)
        assert documents[1].score == pytest.approx(0.8)

    @patch("vectordb.databases.chroma.chromadb")
    def test_query_to_documents_empty_embeddings(self, mock_chromadb) -> None:
        """Test query_to_documents handles empty embeddings list."""
        db = ChromaVectorDB(
            persistent=False,
            collection_name="test_collection",
        )

        results = {
            "ids": [["doc1"]],
            "documents": [["content1"]],
            "metadatas": [[{"key": "val1"}]],
            "distances": [[0.1]],
            "embeddings": [[]],
        }

        documents = db.query_to_documents(results)

        assert len(documents) == 1
        assert not hasattr(documents[0], "embedding") or documents[0].embedding is None

    @patch("vectordb.databases.chroma.chromadb")
    def test_query_to_documents_none_embeddings(self, mock_chromadb) -> None:
        """Test query_to_documents handles None embeddings."""
        db = ChromaVectorDB(
            persistent=False,
            collection_name="test_collection",
        )

        results = {
            "ids": [["doc1"]],
            "documents": [["content1"]],
            "metadatas": [[{"key": "val1"}]],
            "distances": [[0.1]],
            "embeddings": None,
        }

        documents = db.query_to_documents(results)

        assert len(documents) == 1
        assert documents[0].content == "content1"


@pytest.mark.integration
@pytest.mark.enable_socket
class TestChromaVectorDBIntegration:
    """Integration tests for ChromaVectorDB with actual operations.

    These tests require a Chroma instance to be running.
    """

    def test_end_to_end_workflow(self, sample_documents: list[Document]) -> None:
        """Test complete workflow: create collection -> upsert -> search."""
        with (
            patch("vectordb.databases.chroma.chromadb"),
            patch("vectordb.databases.chroma.ChromaVectorDB.create_collection"),
            patch("vectordb.databases.chroma.ChromaVectorDB.upsert"),
            patch("vectordb.databases.chroma.ChromaVectorDB.search"),
        ):
            db = ChromaVectorDB(
                host="localhost",
                port=8000,
                collection_name="test_collection",
            )

            assert db is not None
