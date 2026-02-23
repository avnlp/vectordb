"""Unit tests for Milvus metadata filtering pipeline."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import yaml
from haystack import Document


class TestMilvusMetadataFilteringPipelineUnit:
    """Unit tests for MilvusMetadataFilteringPipeline with comprehensive mocking."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "milvus": {
                "host": "localhost",
                "port": "19530",
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
                        "operators": ["eq"],
                        "description": "Document category",
                    }
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

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            Document(
                content="Test document 1",
                meta={"category": "tech", "year": 2024},
                embedding=[0.1] * 384,
            ),
            Document(
                content="Test document 2",
                meta={"category": "science", "year": 2023},
                embedding=[0.2] * 384,
            ),
        ]

    def test_init_sets_default_dimension(self, config_file):
        """Test that __init__ sets default dimension to 384."""
        with patch("pymilvus.connections.connect"):
            from vectordb.haystack.metadata_filtering.milvus import (
                MilvusMetadataFilteringPipeline,
            )

            pipeline = MilvusMetadataFilteringPipeline(config_file)
            assert pipeline.dimension == 384
            assert pipeline.collection is None

    def test_connect_uses_config_values(self, config_file):
        """Test _connect uses host and port from config."""
        with patch("pymilvus.connections.connect") as mock_connect:
            from vectordb.haystack.metadata_filtering.milvus import (
                MilvusMetadataFilteringPipeline,
            )

            pipeline = MilvusMetadataFilteringPipeline(config_file)
            pipeline._connect()

            # Called once in __init__ and once explicitly
            assert mock_connect.call_count == 2
            mock_connect.assert_called_with(
                alias="default", host="localhost", port="19530"
            )

    def test_connect_uses_default_values(self, config_file):
        """Test _connect uses default values when not in config."""
        config = {"milvus": {}, "logging": {"level": "DEBUG"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()
            config_path = f.name

        try:
            with patch("pymilvus.connections.connect") as mock_connect:
                from vectordb.haystack.metadata_filtering.milvus import (
                    MilvusMetadataFilteringPipeline,
                )

                pipeline = MilvusMetadataFilteringPipeline(config_path)
                pipeline._connect()

                # Called once in __init__ and once explicitly
                assert mock_connect.call_count == 2
                mock_connect.assert_called_with(
                    alias="default", host="localhost", port="19530"
                )
        finally:
            os.unlink(config_path)

    def test_index_documents_raises_when_collection_none(self, config_file):
        """Test _index_documents raises ValueError when collection not initialized."""
        with patch("pymilvus.connections.connect"):
            from vectordb.haystack.metadata_filtering.milvus import (
                MilvusMetadataFilteringPipeline,
            )

            pipeline = MilvusMetadataFilteringPipeline(config_file)
            pipeline.collection = None

            with pytest.raises(ValueError, match="Collection not initialized"):
                pipeline._index_documents([Document(content="test")])

    def test_index_documents_inserts_data(self, config_file, sample_documents):
        """Test _index_documents inserts embeddings, contents, and metadata."""
        with patch("pymilvus.connections.connect"):
            mock_collection = MagicMock()

            from vectordb.haystack.metadata_filtering.milvus import (
                MilvusMetadataFilteringPipeline,
            )

            pipeline = MilvusMetadataFilteringPipeline(config_file)
            pipeline.collection = mock_collection

            count = pipeline._index_documents(sample_documents)

            assert count == 2
            mock_collection.insert.assert_called_once()
            call_args = mock_collection.insert.call_args[0][0]
            assert len(call_args) == 3  # embeddings, contents, metadatas
            mock_collection.flush.assert_called_once()

    def test_pre_filter_returns_zero_when_collection_none(self, config_file):
        """Test _pre_filter returns 0 when collection is None."""
        with patch("pymilvus.connections.connect"):
            from vectordb.haystack.metadata_filtering.milvus import (
                MilvusMetadataFilteringPipeline,
            )

            pipeline = MilvusMetadataFilteringPipeline(config_file)
            pipeline.collection = None

            count = pipeline._pre_filter("category == 'tech'")
            assert count == 0

    def test_pre_filter_returns_total_when_no_filter_expr(self, config_file):
        """Test _pre_filter returns total count when filter_expr is empty."""
        with patch("pymilvus.connections.connect"):
            mock_collection = MagicMock()
            mock_collection.num_entities = 100

            from vectordb.haystack.metadata_filtering.milvus import (
                MilvusMetadataFilteringPipeline,
            )

            pipeline = MilvusMetadataFilteringPipeline(config_file)
            pipeline.collection = mock_collection

            count = pipeline._pre_filter("")
            assert count == 100
            mock_collection.query.assert_not_called()

    def test_pre_filter_counts_with_filter(self, config_file):
        """Test _pre_filter counts matching documents with filter expression."""
        with patch("pymilvus.connections.connect"):
            mock_collection = MagicMock()
            mock_collection.query.return_value = [{"id": 1}, {"id": 2}]

            from vectordb.haystack.metadata_filtering.milvus import (
                MilvusMetadataFilteringPipeline,
            )

            pipeline = MilvusMetadataFilteringPipeline(config_file)
            pipeline.collection = mock_collection

            count = pipeline._pre_filter("category == 'tech'")

            assert count == 2
            mock_collection.query.assert_called_once_with(
                expr="category == 'tech'", output_fields=["id"]
            )

    def test_vector_search_raises_when_collection_none(self, config_file):
        """Test _vector_search raises ValueError when collection not initialized."""
        with patch("pymilvus.connections.connect"):
            from vectordb.haystack.metadata_filtering.milvus import (
                MilvusMetadataFilteringPipeline,
            )

            pipeline = MilvusMetadataFilteringPipeline(config_file)
            pipeline.collection = None

            with pytest.raises(ValueError, match="Collection not initialized"):
                pipeline._vector_search([0.1] * 384, "", top_k=10)

    def test_vector_search_returns_formatted_results(self, config_file):
        """Test _vector_search returns formatted search results."""
        with patch("pymilvus.connections.connect"):
            mock_hit = MagicMock()
            mock_hit.id = 1
            mock_hit.score = 0.95
            mock_hit.entity.get.side_effect = lambda key, default="": {
                "content": "doc content",
                "metadata": {"category": "tech"},
            }.get(key, default)

            mock_collection = MagicMock()
            mock_collection.search.return_value = [[mock_hit]]

            from vectordb.haystack.metadata_filtering.milvus import (
                MilvusMetadataFilteringPipeline,
            )

            pipeline = MilvusMetadataFilteringPipeline(config_file)
            pipeline.collection = mock_collection

            results = pipeline._vector_search(
                [0.1] * 384, "category == 'tech'", top_k=5
            )

            assert len(results) == 1
            assert results[0]["id"] == 1
            assert results[0]["score"] == 0.95
            assert results[0]["content"] == "doc content"
            assert results[0]["metadata"]["category"] == "tech"

    def test_vector_search_with_none_filter(self, config_file):
        """Test _vector_search passes None filter when filter_expr is empty."""
        with patch("pymilvus.connections.connect"):
            mock_collection = MagicMock()
            mock_collection.search.return_value = [[]]

            from vectordb.haystack.metadata_filtering.milvus import (
                MilvusMetadataFilteringPipeline,
            )

            pipeline = MilvusMetadataFilteringPipeline(config_file)
            pipeline.collection = mock_collection

            pipeline._vector_search([0.1] * 384, "", top_k=10)

            call_kwargs = mock_collection.search.call_args.kwargs
            assert call_kwargs["expr"] is None

    def test_run_skips_indexing_when_no_documents(self, config_file):
        """Test run skips indexing when documents list is empty."""
        with (
            patch("pymilvus.connections.connect"),
            patch(
                "vectordb.haystack.metadata_filtering.milvus.MilvusMetadataFilteringPipeline._create_collection"
            ) as mock_create,
            patch(
                "vectordb.haystack.metadata_filtering.milvus.MilvusMetadataFilteringPipeline._index_documents"
            ) as mock_index,
            patch(
                "vectordb.haystack.metadata_filtering.milvus.MilvusMetadataFilteringPipeline._init_embedder"
            ),
            patch(
                "vectordb.haystack.metadata_filtering.milvus.parse_filter_from_config"
            ) as mock_parse,
            patch(
                "vectordb.haystack.metadata_filtering.milvus.MilvusFilterExpressionBuilder"
            ) as mock_builder,
        ):
            mock_collection = MagicMock()
            mock_collection.num_entities = 0
            mock_collection.query.return_value = []
            mock_collection.search.return_value = [[]]
            mock_create.return_value = mock_collection
            mock_parse.return_value = MagicMock()
            mock_builder.return_value.build.return_value = ""

            from vectordb.haystack.metadata_filtering.milvus import (
                MilvusMetadataFilteringPipeline,
            )

            pipeline = MilvusMetadataFilteringPipeline(config_file)
            pipeline.embedder = None

            results = pipeline.run()

            mock_index.assert_not_called()
            assert len(results) == 0

    def test_run_uses_zero_vector_when_no_embedder(self, config_file):
        """Test run uses zero vector when embedder is not initialized."""
        with (
            patch("pymilvus.connections.connect"),
            patch(
                "vectordb.haystack.metadata_filtering.milvus.MilvusMetadataFilteringPipeline._create_collection"
            ) as mock_create,
            patch(
                "vectordb.haystack.metadata_filtering.milvus.MilvusMetadataFilteringPipeline._index_documents"
            ),
            patch(
                "vectordb.haystack.metadata_filtering.milvus.MilvusMetadataFilteringPipeline._init_embedder"
            ),
            patch(
                "vectordb.haystack.metadata_filtering.milvus.parse_filter_from_config"
            ) as mock_parse,
            patch(
                "vectordb.haystack.metadata_filtering.milvus.MilvusFilterExpressionBuilder"
            ) as mock_builder,
        ):
            mock_collection = MagicMock()
            mock_collection.num_entities = 10
            mock_collection.query.return_value = [{"id": i} for i in range(5)]
            mock_collection.search.return_value = [[]]
            mock_create.return_value = mock_collection
            mock_parse.return_value = MagicMock()
            mock_builder.return_value.build.return_value = "category == 'tech'"

            from vectordb.haystack.metadata_filtering.milvus import (
                MilvusMetadataFilteringPipeline,
            )

            pipeline = MilvusMetadataFilteringPipeline(config_file)
            pipeline.embedder = None
            pipeline.dimension = 384

            with patch(
                "vectordb.haystack.metadata_filtering.milvus.MilvusMetadataFilteringPipeline._vector_search"
            ) as mock_search:
                mock_search.return_value = []
                pipeline.run()

                call_args = mock_search.call_args
                query_embedding = call_args[0][0]
                assert len(query_embedding) == 384
                assert all(v == 0.0 for v in query_embedding)

    def test_run_timing_only_on_first_result(self, config_file):
        """Test that timing is only set on first result."""
        with (
            patch("pymilvus.connections.connect"),
            patch(
                "vectordb.haystack.metadata_filtering.milvus.MilvusMetadataFilteringPipeline._create_collection"
            ) as mock_create,
            patch(
                "vectordb.haystack.metadata_filtering.milvus.MilvusMetadataFilteringPipeline._index_documents"
            ),
            patch(
                "vectordb.haystack.metadata_filtering.milvus.MilvusMetadataFilteringPipeline._init_embedder"
            ),
            patch(
                "vectordb.haystack.metadata_filtering.milvus.parse_filter_from_config"
            ) as mock_parse,
            patch(
                "vectordb.haystack.metadata_filtering.milvus.MilvusFilterExpressionBuilder"
            ) as mock_builder,
        ):
            mock_collection = MagicMock()
            mock_collection.num_entities = 10
            mock_collection.query.return_value = [{"id": i} for i in range(5)]
            mock_create.return_value = mock_collection
            mock_parse.return_value = MagicMock()
            mock_builder.return_value.build.return_value = ""

            from vectordb.haystack.metadata_filtering.milvus import (
                MilvusMetadataFilteringPipeline,
            )

            pipeline = MilvusMetadataFilteringPipeline(config_file)
            pipeline.embedder = None

            mock_hit1 = MagicMock()
            mock_hit1.id = 1
            mock_hit1.score = 0.9
            mock_hit1.entity.get.side_effect = lambda key, default="": {
                "content": "doc1",
                "metadata": {},
            }.get(key, default)

            mock_hit2 = MagicMock()
            mock_hit2.id = 2
            mock_hit2.score = 0.8
            mock_hit2.entity.get.side_effect = lambda key, default="": {
                "content": "doc2",
                "metadata": {},
            }.get(key, default)

            mock_collection.search.return_value = [[mock_hit1, mock_hit2]]

            results = pipeline.run()

            assert len(results) == 2
            assert results[0].timing is not None
            assert results[1].timing is None

    def test_run_initializes_text_embedder_for_query(self, config_file):
        """Test run initializes SentenceTransformersTextEmbedder for query."""
        with (
            patch("pymilvus.connections.connect"),
            patch(
                "vectordb.haystack.metadata_filtering.milvus.MilvusMetadataFilteringPipeline._create_collection"
            ) as mock_create,
            patch(
                "vectordb.haystack.metadata_filtering.milvus.MilvusMetadataFilteringPipeline._index_documents"
            ),
            patch(
                "vectordb.haystack.metadata_filtering.milvus.MilvusMetadataFilteringPipeline._init_embedder"
            ),
            patch(
                "vectordb.haystack.metadata_filtering.milvus.parse_filter_from_config"
            ) as mock_parse,
            patch(
                "vectordb.haystack.metadata_filtering.milvus.MilvusFilterExpressionBuilder"
            ) as mock_builder,
            patch(
                "haystack.components.embedders.SentenceTransformersTextEmbedder"
            ) as mock_embedder_class,
        ):
            mock_collection = MagicMock()
            mock_collection.num_entities = 10
            mock_collection.query.return_value = [{"id": i} for i in range(5)]
            mock_collection.search.return_value = [[]]
            mock_create.return_value = mock_collection
            mock_parse.return_value = MagicMock()
            mock_builder.return_value.build.return_value = ""

            mock_text_embedder = MagicMock()
            mock_text_embedder.run.return_value = {"embedding": [0.1] * 384}
            mock_embedder_class.return_value = mock_text_embedder

            from vectordb.haystack.metadata_filtering.milvus import (
                MilvusMetadataFilteringPipeline,
            )

            pipeline = MilvusMetadataFilteringPipeline(config_file)
            pipeline.embedder = MagicMock()

            pipeline.run()

            mock_embedder_class.assert_called_once_with(
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            mock_text_embedder.warm_up.assert_called_once()
            mock_text_embedder.run.assert_called_once_with(text="test query")
