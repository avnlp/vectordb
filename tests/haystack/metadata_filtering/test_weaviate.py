"""Tests for Weaviate metadata filtering pipeline."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import yaml


class TestWeaviateMetadataFilteringPipeline:
    """Tests for WeaviateMetadataFilteringPipeline class."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "weaviate": {
                "url": "http://localhost:8080",
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
                    {
                        "field": "score",
                        "type": "float",
                        "operators": ["gte", "lte"],
                        "description": "Relevance score",
                    },
                    {
                        "field": "active",
                        "type": "boolean",
                        "operators": ["eq"],
                        "description": "Active status",
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
            "collection": {"name": "TestCollection"},
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

    def test_init_creates_client(self, config_file):
        """Test pipeline initialization creates client."""
        with patch(
            "vectordb.haystack.metadata_filtering.weaviate.weaviate.connect_to_local"
        ) as mock_connect:
            mock_client = MagicMock()
            mock_connect.return_value = mock_client

            from vectordb.haystack.metadata_filtering.weaviate import (
                WeaviateMetadataFilteringPipeline,
            )

            WeaviateMetadataFilteringPipeline(config_file)

            mock_connect.assert_called_once()

    def test_connect_with_api_key(self, sample_config):
        """Test connection with API key uses cloud."""
        sample_config["weaviate"]["api_key"] = "test-api-key"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_config, f)
            f.flush()
            config_path = f.name

        try:
            with patch(
                "vectordb.haystack.metadata_filtering.weaviate.weaviate.connect_to_weaviate_cloud"
            ) as mock_cloud:
                mock_client = MagicMock()
                mock_cloud.return_value = mock_client

                from vectordb.haystack.metadata_filtering.weaviate import (
                    WeaviateMetadataFilteringPipeline,
                )

                WeaviateMetadataFilteringPipeline(config_path)

                mock_cloud.assert_called_once()
        finally:
            os.unlink(config_path)

    def test_connect_without_api_key(self, config_file):
        """Test connection without API key uses local."""
        with patch(
            "vectordb.haystack.metadata_filtering.weaviate.weaviate.connect_to_local"
        ) as mock_connect:
            mock_client = MagicMock()
            mock_connect.return_value = mock_client

            from vectordb.haystack.metadata_filtering.weaviate import (
                WeaviateMetadataFilteringPipeline,
            )

            WeaviateMetadataFilteringPipeline(config_file)

            mock_connect.assert_called_once()

    def test_connect_uses_env_url(self, sample_config):
        """Test connection uses WEAVIATE_URL environment variable."""
        sample_config["weaviate"].pop("url", None)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_config, f)
            f.flush()
            config_path = f.name

        try:
            with (
                patch.dict(
                    os.environ, {"WEAVIATE_URL": "http://env-host:8080"}, clear=False
                ),
                patch(
                    "vectordb.haystack.metadata_filtering.weaviate.weaviate.connect_to_local"
                ) as mock_connect,
            ):
                mock_client = MagicMock()
                mock_connect.return_value = mock_client

                from vectordb.haystack.metadata_filtering.weaviate import (
                    WeaviateMetadataFilteringPipeline,
                )

                WeaviateMetadataFilteringPipeline(config_path)

                mock_connect.assert_called_once_with(host="env-host")
        finally:
            os.unlink(config_path)

    def test_create_collection_new(self, config_file):
        """Test creating a new collection."""
        with patch(
            "vectordb.haystack.metadata_filtering.weaviate.weaviate.connect_to_local"
        ) as mock_connect:
            mock_client = MagicMock()
            mock_client.collections.exists.return_value = False
            mock_collection = MagicMock()
            mock_client.collections.create.return_value = mock_collection
            mock_connect.return_value = mock_client

            from vectordb.haystack.metadata_filtering.weaviate import (
                WeaviateMetadataFilteringPipeline,
            )

            pipeline = WeaviateMetadataFilteringPipeline(config_file)
            pipeline._create_collection("TestCollection")

            mock_client.collections.create.assert_called_once()
            call_kwargs = mock_client.collections.create.call_args.kwargs
            assert call_kwargs["name"] == "TestCollection"

    def test_create_collection_existing(self, config_file):
        """Test getting existing collection."""
        with patch(
            "vectordb.haystack.metadata_filtering.weaviate.weaviate.connect_to_local"
        ) as mock_connect:
            mock_client = MagicMock()
            mock_client.collections.exists.return_value = True
            mock_collection = MagicMock()
            mock_client.collections.get.return_value = mock_collection
            mock_connect.return_value = mock_client

            from vectordb.haystack.metadata_filtering.weaviate import (
                WeaviateMetadataFilteringPipeline,
            )

            pipeline = WeaviateMetadataFilteringPipeline(config_file)
            pipeline._create_collection("TestCollection")

            mock_client.collections.get.assert_called_once_with("TestCollection")
            mock_client.collections.create.assert_not_called()

    def test_create_collection_no_client_raises(self, config_file):
        """Test that creating collection without client raises ValueError."""
        with patch(
            "vectordb.haystack.metadata_filtering.weaviate.weaviate.connect_to_local"
        ) as mock_connect:
            mock_client = MagicMock()
            mock_connect.return_value = mock_client

            from vectordb.haystack.metadata_filtering.weaviate import (
                WeaviateMetadataFilteringPipeline,
            )

            pipeline = WeaviateMetadataFilteringPipeline(config_file)
            pipeline.client = None

            with pytest.raises(ValueError, match="Client not initialized"):
                pipeline._create_collection("TestCollection")

    def test_index_documents(self, config_file):
        """Test _index_documents method."""
        with patch(
            "vectordb.haystack.metadata_filtering.weaviate.weaviate.connect_to_local"
        ) as mock_connect:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_batch = MagicMock()
            mock_client.collections.exists.return_value = False
            mock_client.collections.create.return_value = mock_collection
            mock_collection.batch.dynamic.return_value.__enter__ = MagicMock(
                return_value=mock_batch
            )
            mock_collection.batch.dynamic.return_value.__exit__ = MagicMock(
                return_value=False
            )
            mock_connect.return_value = mock_client

            from haystack import Document

            from vectordb.haystack.metadata_filtering.weaviate import (
                WeaviateMetadataFilteringPipeline,
            )

            pipeline = WeaviateMetadataFilteringPipeline(config_file)
            pipeline._create_collection("TestCollection")

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

            assert mock_batch.add_object.call_count == 2
            assert count == 2

    def test_index_documents_skips_without_embedding(self, config_file):
        """Test that documents without embeddings are skipped."""
        with patch(
            "vectordb.haystack.metadata_filtering.weaviate.weaviate.connect_to_local"
        ) as mock_connect:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_batch = MagicMock()
            mock_client.collections.exists.return_value = False
            mock_client.collections.create.return_value = mock_collection
            mock_collection.batch.dynamic.return_value.__enter__ = MagicMock(
                return_value=mock_batch
            )
            mock_collection.batch.dynamic.return_value.__exit__ = MagicMock(
                return_value=False
            )
            mock_connect.return_value = mock_client

            from haystack import Document

            from vectordb.haystack.metadata_filtering.weaviate import (
                WeaviateMetadataFilteringPipeline,
            )

            pipeline = WeaviateMetadataFilteringPipeline(config_file)
            pipeline._create_collection("TestCollection")

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

            assert mock_batch.add_object.call_count == 1
            assert count == 1

    def test_index_documents_no_collection_raises(self, config_file):
        """Test that indexing without collection raises ValueError."""
        with patch(
            "vectordb.haystack.metadata_filtering.weaviate.weaviate.connect_to_local"
        ) as mock_connect:
            mock_client = MagicMock()
            mock_connect.return_value = mock_client

            from haystack import Document

            from vectordb.haystack.metadata_filtering.weaviate import (
                WeaviateMetadataFilteringPipeline,
            )

            pipeline = WeaviateMetadataFilteringPipeline(config_file)
            pipeline.collection = None

            with pytest.raises(ValueError, match="Collection not initialized"):
                pipeline._index_documents(
                    [Document(content="test", embedding=[0.1, 0.2, 0.3])]
                )

    def test_pre_filter_with_filter(self, config_file):
        """Test _pre_filter with filter."""
        with patch(
            "vectordb.haystack.metadata_filtering.weaviate.weaviate.connect_to_local"
        ) as mock_connect:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_aggregate = MagicMock()
            mock_aggregate.total_count = 10
            mock_collection.aggregate.over_all.return_value = mock_aggregate
            mock_client.collections.exists.return_value = True
            mock_client.collections.get.return_value = mock_collection
            mock_connect.return_value = mock_client

            from vectordb.haystack.metadata_filtering.weaviate import (
                WeaviateMetadataFilteringPipeline,
            )

            pipeline = WeaviateMetadataFilteringPipeline(config_file)
            pipeline._create_collection("TestCollection")

            mock_filter = MagicMock()
            count = pipeline._pre_filter(mock_filter)

            mock_collection.aggregate.over_all.assert_called_once()
            assert count == 10

    def test_pre_filter_without_filter(self, config_file):
        """Test _pre_filter without filter."""
        with patch(
            "vectordb.haystack.metadata_filtering.weaviate.weaviate.connect_to_local"
        ) as mock_connect:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_aggregate = MagicMock()
            mock_aggregate.total_count = 5
            mock_collection.aggregate.over_all.return_value = mock_aggregate
            mock_client.collections.exists.return_value = True
            mock_client.collections.get.return_value = mock_collection
            mock_connect.return_value = mock_client

            from vectordb.haystack.metadata_filtering.weaviate import (
                WeaviateMetadataFilteringPipeline,
            )

            pipeline = WeaviateMetadataFilteringPipeline(config_file)
            pipeline._create_collection("TestCollection")

            count = pipeline._pre_filter(None)

            mock_collection.aggregate.over_all.assert_called_once_with(total_count=True)
            assert count == 5

    def test_pre_filter_no_collection_returns_zero(self, config_file):
        """Test that pre-filter returns 0 without collection."""
        with patch(
            "vectordb.haystack.metadata_filtering.weaviate.weaviate.connect_to_local"
        ) as mock_connect:
            mock_client = MagicMock()
            mock_connect.return_value = mock_client

            from vectordb.haystack.metadata_filtering.weaviate import (
                WeaviateMetadataFilteringPipeline,
            )

            pipeline = WeaviateMetadataFilteringPipeline(config_file)
            pipeline.collection = None

            count = pipeline._pre_filter(None)

            assert count == 0

    def test_pre_filter_exception_returns_zero(self, config_file):
        """Test that pre-filter returns 0 on exception."""
        with patch(
            "vectordb.haystack.metadata_filtering.weaviate.weaviate.connect_to_local"
        ) as mock_connect:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.aggregate.over_all.side_effect = Exception("Weaviate error")
            mock_client.collections.exists.return_value = True
            mock_client.collections.get.return_value = mock_collection
            mock_connect.return_value = mock_client

            from vectordb.haystack.metadata_filtering.weaviate import (
                WeaviateMetadataFilteringPipeline,
            )

            pipeline = WeaviateMetadataFilteringPipeline(config_file)
            pipeline._create_collection("TestCollection")

            mock_filter = MagicMock()
            count = pipeline._pre_filter(mock_filter)

            assert count == 0

    def test_vector_search(self, config_file):
        """Test _vector_search method."""
        with patch(
            "vectordb.haystack.metadata_filtering.weaviate.weaviate.connect_to_local"
        ) as mock_connect:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_query = MagicMock()
            mock_objects = MagicMock()
            mock_objects.properties = {"content": "doc1 content", "category": "tech"}
            mock_objects.metadata.distance = 0.2
            mock_objects.metadata.score = 0.8
            mock_query.objects = [mock_objects]
            mock_collection.query.near_vector.return_value = mock_query
            mock_client.collections.exists.return_value = True
            mock_client.collections.get.return_value = mock_collection
            mock_connect.return_value = mock_client

            from vectordb.haystack.metadata_filtering.weaviate import (
                WeaviateMetadataFilteringPipeline,
            )

            pipeline = WeaviateMetadataFilteringPipeline(config_file)
            pipeline._create_collection("TestCollection")

            query_embedding = [0.1, 0.2, 0.3]
            results = pipeline._vector_search(query_embedding, None, top_k=10)

            assert len(results) == 1
            assert "id" in results[0]
            assert results[0]["score"] == 0.8  # 1 - 0.2
            assert results[0]["content"] == "doc1 content"
            assert results[0]["metadata"]["category"] == "tech"

    def test_vector_search_with_filter(self, config_file):
        """Test _vector_search with filter."""
        with patch(
            "vectordb.haystack.metadata_filtering.weaviate.weaviate.connect_to_local"
        ) as mock_connect:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_query = MagicMock()
            mock_query.objects = []
            mock_collection.query.near_vector.return_value = mock_query
            mock_client.collections.exists.return_value = True
            mock_client.collections.get.return_value = mock_collection
            mock_connect.return_value = mock_client

            from vectordb.haystack.metadata_filtering.weaviate import (
                WeaviateMetadataFilteringPipeline,
            )

            pipeline = WeaviateMetadataFilteringPipeline(config_file)
            pipeline._create_collection("TestCollection")

            query_embedding = [0.1, 0.2, 0.3]
            mock_filter = MagicMock()
            pipeline._vector_search(query_embedding, mock_filter, top_k=10)

            mock_collection.query.near_vector.assert_called_once()
            call_args = mock_collection.query.near_vector.call_args
            assert call_args.kwargs["filters"] == mock_filter

    def test_vector_search_no_collection_raises(self, config_file):
        """Test that vector search raises error without collection."""
        with patch(
            "vectordb.haystack.metadata_filtering.weaviate.weaviate.connect_to_local"
        ) as mock_connect:
            mock_client = MagicMock()
            mock_connect.return_value = mock_client

            from vectordb.haystack.metadata_filtering.weaviate import (
                WeaviateMetadataFilteringPipeline,
            )

            pipeline = WeaviateMetadataFilteringPipeline(config_file)
            pipeline.collection = None

            with pytest.raises(ValueError, match="Collection not initialized"):
                pipeline._vector_search([0.1, 0.2, 0.3], None)

    def test_run_without_documents(self, config_file):
        """Test run method without documents."""
        with patch(
            "vectordb.haystack.metadata_filtering.weaviate.weaviate.connect_to_local"
        ) as mock_connect:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_aggregate = MagicMock()
            mock_aggregate.total_count = 0
            mock_collection.aggregate.over_all.return_value = mock_aggregate
            mock_query = MagicMock()
            mock_query.objects = []
            mock_collection.query.near_vector.return_value = mock_query
            mock_client.collections.exists.return_value = True
            mock_client.collections.get.return_value = mock_collection
            mock_connect.return_value = mock_client

            from vectordb.haystack.metadata_filtering.weaviate import (
                WeaviateMetadataFilteringPipeline,
            )

            pipeline = WeaviateMetadataFilteringPipeline(config_file)

            # Run without documents (documents list is empty in run method)
            with patch.object(pipeline, "_init_embedder"):
                results = pipeline.run()

            # Should return empty results
            assert len(results) == 0


class TestWeaviateMetadataFilteringPipelineUnit:
    """Unit tests for Weaviate pipeline with comprehensive mocking."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "weaviate": {
                "url": "http://localhost:8080",
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
            "collection": {"name": "TestCollection"},
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
                "vectordb.haystack.metadata_filtering.weaviate.weaviate.connect_to_local"
            ):
                from vectordb.haystack.metadata_filtering.weaviate import (
                    WeaviateMetadataFilteringPipeline,
                )

                pipeline = WeaviateMetadataFilteringPipeline(config_path)
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
                "vectordb.haystack.metadata_filtering.weaviate.weaviate.connect_to_local"
            ):
                from vectordb.haystack.metadata_filtering.weaviate import (
                    WeaviateMetadataFilteringPipeline,
                )

                pipeline = WeaviateMetadataFilteringPipeline(config_path)
                assert pipeline.logger is not None
        finally:
            os.unlink(config_path)

    def test_del_closes_client(self, sample_config):
        """Test that __del__ closes client connection."""
        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_config, f)
            f.flush()
            config_path = f.name

        try:
            with patch(
                "vectordb.haystack.metadata_filtering.weaviate.weaviate.connect_to_local"
            ) as mock_connect:
                mock_client = MagicMock()
                mock_connect.return_value = mock_client

                from vectordb.haystack.metadata_filtering.weaviate import (
                    WeaviateMetadataFilteringPipeline,
                )

                pipeline = WeaviateMetadataFilteringPipeline(config_path)

                # Simulate deletion
                pipeline.__del__()

                mock_client.close.assert_called_once()
        finally:
            os.unlink(config_path)

    def test_del_handles_exception(self, sample_config):
        """Test that __del__ handles exceptions gracefully."""
        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_config, f)
            f.flush()
            config_path = f.name

        try:
            with patch(
                "vectordb.haystack.metadata_filtering.weaviate.weaviate.connect_to_local"
            ) as mock_connect:
                mock_client = MagicMock()
                mock_client.close.side_effect = Exception("Close failed")
                mock_connect.return_value = mock_client

                from vectordb.haystack.metadata_filtering.weaviate import (
                    WeaviateMetadataFilteringPipeline,
                )

                pipeline = WeaviateMetadataFilteringPipeline(config_path)

                # Should not raise
                pipeline.__del__()
        finally:
            os.unlink(config_path)
