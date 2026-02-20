"""Tests for QdrantMetadataFilteringPipeline."""

from unittest.mock import MagicMock, mock_open, patch

import pytest
from haystack import Document
from qdrant_client.http.models import Distance, VectorParams

from vectordb.haystack.metadata_filtering.qdrant import QdrantMetadataFilteringPipeline
from vectordb.haystack.metadata_filtering.vectordb_pipeline_type import (
    FilteredQueryResult,
)


@pytest.fixture
def mock_config():
    return {
        "qdrant": {"host": "localhost", "port": 6333, "api_key": "test-key"},
        "collection": {
            "name": "test_collection",
        },
        "embeddings": {"dimension": 128, "model": "test-model"},
        "dataloader": {"limit": 10},
        "metadata_filtering": {
            "schema": [
                {"field": "category", "type": "string", "operators": ["eq"]},
            ],
            "test_filters": [
                {
                    "name": "test_filter",
                    "conditions": [
                        {"field": "category", "operator": "eq", "value": "news"}
                    ],
                }
            ],
            "filter_query": "category == 'news'",
            "test_query": "test query",
        },
        "logging": {"name": "test_logger", "level": "INFO"},
    }


@pytest.fixture
def mock_qdrant_client():
    with patch(
        "vectordb.haystack.metadata_filtering.qdrant.QdrantClient"
    ) as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # Setup mock behavior
        mock_client.get_collections.return_value.collections = []
        mock_client.scroll.return_value = ([1, 2, 3], None)

        # Mock search results
        mock_hit = MagicMock()
        mock_hit.id = 1
        mock_hit.score = 0.92
        mock_hit.payload = {"content": "test content", "category": "news"}
        mock_client.search.return_value = [mock_hit]

        mock_client.get_collection.return_value.points_count = 100

        yield {"cls": mock_client_cls, "instance": mock_client}


@pytest.fixture
def mock_embedders():
    with (
        patch(
            "vectordb.haystack.metadata_filtering.base.SentenceTransformersDocumentEmbedder"
        ) as mock_doc_emb,
        patch(
            "haystack.components.embedders.SentenceTransformersTextEmbedder"
        ) as mock_text_emb,
    ):
        mock_doc_instance = MagicMock()
        mock_doc_emb.return_value = mock_doc_instance
        mock_doc_instance.run.return_value = {"documents": []}

        mock_text_instance = MagicMock()
        mock_text_emb.return_value = mock_text_instance
        mock_text_instance.run.return_value = {"embedding": [0.1] * 128}

        yield {
            "doc_embedder": mock_doc_emb,
            "text_embedder": mock_text_emb,
            "doc_instance": mock_doc_instance,
            "text_instance": mock_text_instance,
        }


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_initialization(mock_yaml_load, mock_file, mock_config, mock_qdrant_client):
    mock_yaml_load.return_value = mock_config

    pipeline = QdrantMetadataFilteringPipeline("config.yaml")

    assert pipeline.config == mock_config
    mock_qdrant_client["cls"].assert_called_with(
        host="localhost", port=6333, api_key="test-key"
    )


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_create_collection(mock_yaml_load, mock_file, mock_config, mock_qdrant_client):
    mock_yaml_load.return_value = mock_config
    pipeline = QdrantMetadataFilteringPipeline("config.yaml")

    pipeline._create_collection("new_collection", 128)

    mock_qdrant_client["instance"].create_collection.assert_called_with(
        collection_name="new_collection",
        vectors_config=VectorParams(size=128, distance=Distance.COSINE),
    )


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_create_existing_collection(
    mock_yaml_load, mock_file, mock_config, mock_qdrant_client
):
    mock_yaml_load.return_value = mock_config
    # Mock existing collection
    mock_col = MagicMock()
    mock_col.name = "existing_col"
    mock_qdrant_client["instance"].get_collections.return_value.collections = [mock_col]

    pipeline = QdrantMetadataFilteringPipeline("config.yaml")
    pipeline._create_collection("existing_col", 128)

    mock_qdrant_client["instance"].create_collection.assert_not_called()


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_index_documents(mock_yaml_load, mock_file, mock_config, mock_qdrant_client):
    mock_yaml_load.return_value = mock_config
    pipeline = QdrantMetadataFilteringPipeline("config.yaml")
    pipeline.collection_name = "test_col"

    docs = [Document(content="doc1", meta={"k": "v"}, embedding=[0.1] * 128)]

    count = pipeline._index_documents(docs)

    assert count == 1
    mock_qdrant_client["instance"].upsert.assert_called_once()


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_pre_filter(mock_yaml_load, mock_file, mock_config, mock_qdrant_client):
    mock_yaml_load.return_value = mock_config
    pipeline = QdrantMetadataFilteringPipeline("config.yaml")
    pipeline.collection_name = "test_col"

    filter_obj = MagicMock()
    count = pipeline._pre_filter(filter_obj)

    assert count == 3  # from mock
    mock_qdrant_client["instance"].scroll.assert_called_with(
        collection_name="test_col",
        scroll_filter=filter_obj,
        limit=10000,
        with_payload=False,
        with_vectors=False,
    )


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_vector_search(mock_yaml_load, mock_file, mock_config, mock_qdrant_client):
    mock_yaml_load.return_value = mock_config
    pipeline = QdrantMetadataFilteringPipeline("config.yaml")
    pipeline.collection_name = "test_col"

    results = pipeline._vector_search([0.1] * 128, MagicMock(), top_k=5)

    assert len(results) == 1
    assert results[0]["score"] == 0.92

    mock_qdrant_client["instance"].search.assert_called_once()


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_vector_search_without_client_raises(
    mock_yaml_load, mock_file, mock_config, mock_qdrant_client
):
    mock_yaml_load.return_value = mock_config
    pipeline = QdrantMetadataFilteringPipeline("config.yaml")
    pipeline.client = None

    with pytest.raises(ValueError, match="Client not initialized"):
        pipeline._vector_search([0.1] * 128, MagicMock(), top_k=5)


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_run_pipeline(
    mock_yaml_load, mock_file, mock_config, mock_qdrant_client, mock_embedders
):
    mock_yaml_load.return_value = mock_config
    pipeline = QdrantMetadataFilteringPipeline("config.yaml")

    results = pipeline.run()

    assert len(results) == 1
    assert isinstance(results[0], FilteredQueryResult)
    assert results[0].relevance_score == 0.92

    mock_qdrant_client["instance"].create_collection.assert_called()
    mock_embedders["text_instance"].run.assert_called()
    mock_qdrant_client["instance"].scroll.assert_called()
    mock_qdrant_client["instance"].search.assert_called()
