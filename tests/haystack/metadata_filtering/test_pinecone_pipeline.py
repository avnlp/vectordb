"""Tests for PineconeMetadataFilteringPipeline."""

from unittest.mock import MagicMock, mock_open, patch

import pytest
from haystack import Document

from vectordb.haystack.metadata_filtering.pinecone import (
    PineconeMetadataFilteringPipeline,
)
from vectordb.haystack.metadata_filtering.vectordb_pipeline_type import (
    FilteredQueryResult,
)


@pytest.fixture
def mock_config():
    return {
        "pinecone": {
            "api_key": "test-key",
            "environment": "test-env",
            "namespace": "test-namespace",
            "index_name": "test-index",
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
def mock_pinecone_db():
    with patch(
        "vectordb.haystack.metadata_filtering.pinecone.PineconeVectorDB"
    ) as mock_db_cls:
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        # Setup mock db behavior
        mock_db.upsert.return_value = 10
        mock_db.estimate_match_count.return_value = 5
        mock_db.describe_index_stats.return_value = {"total_vector_count": 100}

        # Mock query result
        mock_doc = Document(content="test content", meta={"category": "news"})
        mock_doc.score = 0.85
        mock_db.query.return_value = [mock_doc]

        yield {"cls": mock_db_cls, "instance": mock_db}


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
def test_initialization(mock_yaml_load, mock_file, mock_config, mock_pinecone_db):
    mock_yaml_load.return_value = mock_config

    pipeline = PineconeMetadataFilteringPipeline("config.yaml")

    assert pipeline.config == mock_config
    mock_pinecone_db["cls"].assert_called_with(config=mock_config)
    assert pipeline.namespace == "test-namespace"


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_create_index(mock_yaml_load, mock_file, mock_config, mock_pinecone_db):
    mock_yaml_load.return_value = mock_config
    pipeline = PineconeMetadataFilteringPipeline("config.yaml")

    pipeline._create_index("test-index", 128)

    mock_pinecone_db["instance"].create_index.assert_called_with(
        index_name="test-index", dimension=128, metric="cosine"
    )


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_index_documents(mock_yaml_load, mock_file, mock_config, mock_pinecone_db):
    mock_yaml_load.return_value = mock_config
    pipeline = PineconeMetadataFilteringPipeline("config.yaml")

    docs = [Document(content="doc1")]
    count = pipeline._index_documents(docs)

    assert count == 10  # from mock return value
    mock_pinecone_db["instance"].upsert.assert_called_with(
        data=docs, namespace="test-namespace"
    )


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_pre_filter(mock_yaml_load, mock_file, mock_config, mock_pinecone_db):
    mock_yaml_load.return_value = mock_config
    pipeline = PineconeMetadataFilteringPipeline("config.yaml")

    filter_dict = {"key": "value"}
    count = pipeline._pre_filter(filter_dict)

    assert count == 5  # from mock
    mock_pinecone_db["instance"].estimate_match_count.assert_called_with(
        filter_dict, namespace="test-namespace"
    )


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_vector_search(mock_yaml_load, mock_file, mock_config, mock_pinecone_db):
    mock_yaml_load.return_value = mock_config
    pipeline = PineconeMetadataFilteringPipeline("config.yaml")

    results = pipeline._vector_search([0.1] * 128, {"key": "val"}, top_k=5)

    assert len(results) == 1
    mock_pinecone_db["instance"].query.assert_called_with(
        vector=[0.1] * 128,
        top_k=5,
        filter={"key": "val"},
        namespace="test-namespace",
        include_metadata=True,
    )


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_run_pipeline(
    mock_yaml_load, mock_file, mock_config, mock_pinecone_db, mock_embedders
):
    mock_yaml_load.return_value = mock_config
    pipeline = PineconeMetadataFilteringPipeline("config.yaml")

    results = pipeline.run()

    assert len(results) == 1
    assert isinstance(results[0], FilteredQueryResult)
    assert results[0].relevance_score == 0.85

    mock_pinecone_db["instance"].create_index.assert_called()
    mock_embedders["text_instance"].run.assert_called()
    mock_pinecone_db["instance"].estimate_match_count.assert_called()
    mock_pinecone_db["instance"].query.assert_called()
