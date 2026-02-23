"""Tests for Chroma metadata filtering pipeline."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import yaml


class TestChromaMetadataFilteringPipeline:
    """Tests for ChromaMetadataFilteringPipeline class."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_collection",
            },
            "embeddings": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384,
            },
            "metadata_filtering": {
                "test_query": "test query",
                "schema": [
                    {
                        "field": "category",
                        "type": "string",
                        "operators": ["eq", "ne", "contains"],
                        "description": "Document category",
                    },
                    {
                        "field": "year",
                        "type": "integer",
                        "operators": ["eq", "gt", "gte", "lt", "lte", "range"],
                        "description": "Publication year",
                    },
                ],
                "test_filters": [
                    {
                        "name": "tech_filter",
                        "conditions": [
                            {"field": "category", "operator": "eq", "value": "tech"}
                        ],
                    }
                ],
            },
            "dataloader": {"limit": 10},
            "logging": {"level": "DEBUG"},
        }

    @pytest.fixture
    def config_file(self, sample_config):
        """Create temporary config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_config, f)
            f.flush()
            yield f.name
        os.unlink(f.name)

    def test_init_without_embedder(self, config_file):
        """Test pipeline initialization without embedder."""
        with patch(
            "vectordb.haystack.metadata_filtering.chroma.chromadb.PersistentClient"
        ) as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance

            from vectordb.haystack.metadata_filtering.chroma import (
                ChromaMetadataFilteringPipeline,
            )

            pipeline = ChromaMetadataFilteringPipeline(config_file)

            mock_client.assert_called_once()
            assert pipeline.client is not None

    def test_connect_uses_persist_directory(self, config_file):
        """Test that _connect uses persist directory from config."""
        with patch(
            "vectordb.haystack.metadata_filtering.chroma.chromadb.PersistentClient"
        ) as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance

            from vectordb.haystack.metadata_filtering.chroma import (
                ChromaMetadataFilteringPipeline,
            )

            ChromaMetadataFilteringPipeline(config_file)

            mock_client.assert_called_once_with(path="./test_chroma_data")

    def test_connect_uses_ephemeral_client_when_persist_dir_empty(self):
        """Test _connect uses Client when persist dir is empty."""
        import yaml

        config = {
            "chroma": {"persist_directory": "", "collection_name": "test_collection"},
            "embeddings": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384,
            },
            "metadata_filtering": {"schema": [], "test_filters": []},
            "dataloader": {"limit": 10},
            "logging": {"level": "DEBUG"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()
            config_path = f.name

        try:
            with (
                patch.dict(os.environ, {"CHROMA_PERSIST_DIR": ""}, clear=False),
                patch(
                    "vectordb.haystack.metadata_filtering.chroma.chromadb.Client"
                ) as mock_client,
                patch(
                    "vectordb.haystack.metadata_filtering.chroma.chromadb.PersistentClient"
                ) as mock_persistent,
            ):
                from vectordb.haystack.metadata_filtering.chroma import (
                    ChromaMetadataFilteringPipeline,
                )

                ChromaMetadataFilteringPipeline(config_path)

                mock_client.assert_called_once()
                mock_persistent.assert_not_called()
        finally:
            os.unlink(config_path)

    def test_create_collection(self, config_file):
        """Test _create_collection method."""
        with patch(
            "vectordb.haystack.metadata_filtering.chroma.chromadb.PersistentClient"
        ) as mock_client:
            mock_client_instance = MagicMock()
            mock_collection = MagicMock()
            mock_client_instance.get_or_create_collection.return_value = mock_collection
            mock_client.return_value = mock_client_instance

            from vectordb.haystack.metadata_filtering.chroma import (
                ChromaMetadataFilteringPipeline,
            )

            pipeline = ChromaMetadataFilteringPipeline(config_file)
            result = pipeline._create_collection("test_collection")

            mock_client_instance.get_or_create_collection.assert_called_once_with(
                name="test_collection", metadata={"hnsw:space": "cosine"}
            )
            assert result == mock_collection
            assert pipeline.collection_name == "test_collection"

    def test_create_collection_no_client_raises(self, config_file):
        """Test that creating collection without client raises ValueError."""
        with patch(
            "vectordb.haystack.metadata_filtering.chroma.chromadb.PersistentClient"
        ) as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance

            from vectordb.haystack.metadata_filtering.chroma import (
                ChromaMetadataFilteringPipeline,
            )

            pipeline = ChromaMetadataFilteringPipeline(config_file)
            pipeline.client = None

            with pytest.raises(ValueError, match="Client not initialized"):
                pipeline._create_collection("test_collection")

    def test_index_documents(self, config_file):
        """Test _index_documents method."""
        with patch(
            "vectordb.haystack.metadata_filtering.chroma.chromadb.PersistentClient"
        ) as mock_client:
            mock_client_instance = MagicMock()
            mock_collection = MagicMock()
            mock_client_instance.get_or_create_collection.return_value = mock_collection
            mock_client.return_value = mock_client_instance

            from haystack import Document

            from vectordb.haystack.metadata_filtering.chroma import (
                ChromaMetadataFilteringPipeline,
            )

            pipeline = ChromaMetadataFilteringPipeline(config_file)
            pipeline._create_collection("test")

            documents = [
                Document(
                    content="Test document 1",
                    meta={"category": "tech", "year": 2024},
                    embedding=[0.1, 0.2, 0.3],
                ),
                Document(
                    content="Test document 2",
                    meta={"category": "science", "year": 2023},
                    embedding=[0.4, 0.5, 0.6],
                ),
            ]

            count = pipeline._index_documents(documents)

            mock_collection.add.assert_called_once()
            call_args = mock_collection.add.call_args
            assert len(call_args.kwargs["ids"]) == 2
            assert len(call_args.kwargs["embeddings"]) == 2
            assert call_args.kwargs["documents"] == [
                "Test document 1",
                "Test document 2",
            ]
            assert count == 2

    def test_index_documents_skips_without_embedding(self, config_file):
        """Test that documents without embeddings are skipped."""
        with patch(
            "vectordb.haystack.metadata_filtering.chroma.chromadb.PersistentClient"
        ) as mock_client:
            mock_client_instance = MagicMock()
            mock_collection = MagicMock()
            mock_client_instance.get_or_create_collection.return_value = mock_collection
            mock_client.return_value = mock_client_instance

            from haystack import Document

            from vectordb.haystack.metadata_filtering.chroma import (
                ChromaMetadataFilteringPipeline,
            )

            pipeline = ChromaMetadataFilteringPipeline(config_file)
            pipeline._create_collection("test")

            documents = [
                Document(
                    content="With embedding",
                    meta={},
                    embedding=[0.1, 0.2, 0.3],
                ),
                Document(
                    content="Without embedding",
                    meta={},
                    embedding=None,
                ),
            ]

            count = pipeline._index_documents(documents)

            mock_collection.add.assert_called_once()
            call_args = mock_collection.add.call_args
            assert len(call_args.kwargs["ids"]) == 1
            assert count == 1

    def test_index_documents_no_collection_raises(self, config_file):
        """Test that indexing without collection raises ValueError."""
        with patch(
            "vectordb.haystack.metadata_filtering.chroma.chromadb.PersistentClient"
        ) as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance

            from haystack import Document

            from vectordb.haystack.metadata_filtering.chroma import (
                ChromaMetadataFilteringPipeline,
            )

            pipeline = ChromaMetadataFilteringPipeline(config_file)

            with pytest.raises(ValueError, match="Collection not initialized"):
                pipeline._index_documents(
                    [Document(content="test", embedding=[0.1, 0.2, 0.3])]
                )

    def test_pre_filter_with_filter(self, config_file):
        """Test _pre_filter with filter."""
        with patch(
            "vectordb.haystack.metadata_filtering.chroma.chromadb.PersistentClient"
        ) as mock_client:
            mock_client_instance = MagicMock()
            mock_collection = MagicMock()
            mock_collection.get.return_value = {"ids": ["1", "2", "3"]}
            mock_client_instance.get_or_create_collection.return_value = mock_collection
            mock_client.return_value = mock_client_instance

            from vectordb.haystack.metadata_filtering.chroma import (
                ChromaMetadataFilteringPipeline,
            )

            pipeline = ChromaMetadataFilteringPipeline(config_file)
            pipeline._create_collection("test")

            filter_dict = {"category": {"$eq": "tech"}}
            count = pipeline._pre_filter(filter_dict)

            mock_collection.get.assert_called_once_with(where=filter_dict, include=[])
            assert count == 3

    def test_pre_filter_without_filter(self, config_file):
        """Test _pre_filter without filter."""
        with patch(
            "vectordb.haystack.metadata_filtering.chroma.chromadb.PersistentClient"
        ) as mock_client:
            mock_client_instance = MagicMock()
            mock_collection = MagicMock()
            mock_collection.get.return_value = {"ids": ["1", "2"]}
            mock_client_instance.get_or_create_collection.return_value = mock_collection
            mock_client.return_value = mock_client_instance

            from vectordb.haystack.metadata_filtering.chroma import (
                ChromaMetadataFilteringPipeline,
            )

            pipeline = ChromaMetadataFilteringPipeline(config_file)
            pipeline._create_collection("test")

            count = pipeline._pre_filter({})

            mock_collection.get.assert_called_once_with(include=[])
            assert count == 2

    def test_pre_filter_no_collection_returns_zero(self, config_file):
        """Test that pre-filter returns 0 without collection."""
        with patch(
            "vectordb.haystack.metadata_filtering.chroma.chromadb.PersistentClient"
        ) as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance

            from vectordb.haystack.metadata_filtering.chroma import (
                ChromaMetadataFilteringPipeline,
            )

            pipeline = ChromaMetadataFilteringPipeline(config_file)
            pipeline.collection = None

            count = pipeline._pre_filter({})

            assert count == 0

    def test_pre_filter_exception_returns_zero(self, config_file):
        """Test that pre-filter returns 0 on exception."""
        with patch(
            "vectordb.haystack.metadata_filtering.chroma.chromadb.PersistentClient"
        ) as mock_client:
            mock_client_instance = MagicMock()
            mock_collection = MagicMock()
            mock_collection.get.side_effect = Exception("Chroma error")
            mock_client_instance.get_or_create_collection.return_value = mock_collection
            mock_client.return_value = mock_client_instance

            from vectordb.haystack.metadata_filtering.chroma import (
                ChromaMetadataFilteringPipeline,
            )

            pipeline = ChromaMetadataFilteringPipeline(config_file)
            pipeline._create_collection("test")

            count = pipeline._pre_filter({"category": "tech"})

            assert count == 0

    def test_vector_search(self, config_file):
        """Test _vector_search method."""
        with patch(
            "vectordb.haystack.metadata_filtering.chroma.chromadb.PersistentClient"
        ) as mock_client:
            mock_client_instance = MagicMock()
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "ids": [["1", "2"]],
                "documents": [["doc1 content", "doc2 content"]],
                "metadatas": [[{"category": "tech"}, {"category": "science"}]],
                "distances": [[0.2, 0.4]],
            }
            mock_client_instance.get_or_create_collection.return_value = mock_collection
            mock_client.return_value = mock_client_instance

            from vectordb.haystack.metadata_filtering.chroma import (
                ChromaMetadataFilteringPipeline,
            )

            pipeline = ChromaMetadataFilteringPipeline(config_file)
            pipeline._create_collection("test")

            query_embedding = [0.1, 0.2, 0.3]
            results = pipeline._vector_search(query_embedding, {}, top_k=2)

            assert len(results) == 2
            assert results[0]["id"] == "1"
            assert results[0]["score"] == 0.8  # 1 - 0.2
            assert results[0]["content"] == "doc1 content"
            assert results[1]["id"] == "2"
            assert results[1]["score"] == 0.6  # 1 - 0.4

    def test_vector_search_with_filter(self, config_file):
        """Test _vector_search with filter."""
        with patch(
            "vectordb.haystack.metadata_filtering.chroma.chromadb.PersistentClient"
        ) as mock_client:
            mock_client_instance = MagicMock()
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "ids": [["1"]],
                "documents": [["doc1 content"]],
                "metadatas": [[{"category": "tech"}]],
                "distances": [[0.3]],
            }
            mock_client_instance.get_or_create_collection.return_value = mock_collection
            mock_client.return_value = mock_client_instance

            from vectordb.haystack.metadata_filtering.chroma import (
                ChromaMetadataFilteringPipeline,
            )

            pipeline = ChromaMetadataFilteringPipeline(config_file)
            pipeline._create_collection("test")

            query_embedding = [0.1, 0.2, 0.3]
            filter_dict = {"category": {"$eq": "tech"}}
            pipeline._vector_search(query_embedding, filter_dict, top_k=10)

            mock_collection.query.assert_called_once()
            call_args = mock_collection.query.call_args
            assert call_args.kwargs["where"] == filter_dict

    def test_vector_search_no_collection_raises(self, config_file):
        """Test that vector search raises error without collection."""
        with patch(
            "vectordb.haystack.metadata_filtering.chroma.chromadb.PersistentClient"
        ) as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance

            from vectordb.haystack.metadata_filtering.chroma import (
                ChromaMetadataFilteringPipeline,
            )

            pipeline = ChromaMetadataFilteringPipeline(config_file)
            pipeline.collection = None

            with pytest.raises(ValueError, match="Collection not initialized"):
                pipeline._vector_search([0.1, 0.2, 0.3], {})

    def test_run_without_documents(self, config_file):
        """Test run method without documents."""
        with patch(
            "vectordb.haystack.metadata_filtering.chroma.chromadb.PersistentClient"
        ) as mock_client:
            mock_client_instance = MagicMock()
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_collection.query.return_value = {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }
            mock_client_instance.get_or_create_collection.return_value = mock_collection
            mock_client.return_value = mock_client_instance

            from vectordb.haystack.metadata_filtering.chroma import (
                ChromaMetadataFilteringPipeline,
            )

            pipeline = ChromaMetadataFilteringPipeline(config_file)

            # Run without documents (documents list is empty in run method)
            with patch.object(pipeline, "_init_embedder"):
                results = pipeline.run()

            # Should return empty results
            assert len(results) == 0


class TestChromaMetadataFilteringPipelineUnit:
    """Unit tests for Chroma pipeline with comprehensive mocking."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "chroma": {
                "persist_directory": "./test_chroma_data",
                "collection_name": "test_collection",
            },
            "embeddings": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384,
            },
            "metadata_filtering": {
                "test_query": "test query",
                "schema": [],
                "test_filters": [],
            },
            "dataloader": {"limit": 10},
            "logging": {"level": "DEBUG"},
        }

    def test_default_dimension(self, sample_config):
        """Test that default dimension is 384."""
        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_config, f)
            f.flush()
            config_path = f.name

        try:
            with patch(
                "vectordb.haystack.metadata_filtering.chroma.chromadb.PersistentClient"
            ):
                from vectordb.haystack.metadata_filtering.chroma import (
                    ChromaMetadataFilteringPipeline,
                )

                pipeline = ChromaMetadataFilteringPipeline(config_path)
                assert pipeline.dimension == 384
        finally:
            os.unlink(config_path)

    def test_logger_setup(self, sample_config):
        """Test logger is properly set up."""
        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_config, f)
            f.flush()
            config_path = f.name

        try:
            with patch(
                "vectordb.haystack.metadata_filtering.chroma.chromadb.PersistentClient"
            ):
                from vectordb.haystack.metadata_filtering.chroma import (
                    ChromaMetadataFilteringPipeline,
                )

                pipeline = ChromaMetadataFilteringPipeline(config_path)
                assert pipeline.logger is not None
        finally:
            os.unlink(config_path)
