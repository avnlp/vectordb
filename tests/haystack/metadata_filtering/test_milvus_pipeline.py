"""Tests for MilvusMetadataFilteringPipeline."""

from unittest.mock import MagicMock, mock_open, patch

import pytest
from haystack import Document

from vectordb.haystack.metadata_filtering.milvus import MilvusMetadataFilteringPipeline
from vectordb.haystack.metadata_filtering.vectordb_pipeline_type import (
    FilteredQueryResult,
)


@pytest.fixture
def mock_config():
    return {
        "milvus": {
            "host": "localhost",
            "port": "19530",
        },
        "collection": {
            "name": "test_collection",
        },
        "embeddings": {"dimension": 128, "model": "test-model"},
        "dataloader": {"limit": 10},
        "metadata_filtering": {
            "schema": [
                {"field": "category", "type": "string", "operators": ["eq"]},
                {"field": "year", "type": "integer", "operators": ["gt", "lt"]},
            ],
            "test_filters": [
                {
                    "name": "test_filter",
                    "conditions": [
                        {"field": "category", "operator": "eq", "value": "news"}
                    ],
                }
            ],
            "filter_query": "category == 'news' AND year > 2020",
            "test_query": "test query",
        },
        "logging": {"name": "test_logger", "level": "DEBUG"},
    }


@pytest.fixture
def mock_milvus_components():
    with (
        patch("vectordb.haystack.metadata_filtering.milvus.connections") as mock_conn,
        patch("vectordb.haystack.metadata_filtering.milvus.Collection") as mock_col_cls,
        patch(
            "vectordb.haystack.metadata_filtering.milvus.CollectionSchema"
        ) as mock_schema,
        patch("vectordb.haystack.metadata_filtering.milvus.FieldSchema") as mock_field,
    ):
        mock_collection = MagicMock()
        mock_col_cls.return_value = mock_collection

        # Setup mock collection behavior
        mock_collection.num_entities = 100
        mock_collection.insert.return_value = None
        mock_collection.search.return_value = []
        mock_collection.query.return_value = [{"id": 1}]

        yield {
            "connections": mock_conn,
            "Collection": mock_col_cls,
            "collection_instance": mock_collection,
            "CollectionSchema": mock_schema,
            "FieldSchema": mock_field,
        }


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


@patch("builtins.open", new_callable=mock_open, read_data="config: data")
@patch("yaml.safe_load")
def test_initialization(mock_yaml_load, mock_file, mock_config, mock_milvus_components):
    mock_yaml_load.return_value = mock_config

    pipeline = MilvusMetadataFilteringPipeline("config.yaml")

    assert pipeline.config == mock_config
    mock_milvus_components["connections"].connect.assert_called_with(
        alias="default", host="localhost", port="19530"
    )


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_create_collection(
    mock_yaml_load, mock_file, mock_config, mock_milvus_components
):
    mock_yaml_load.return_value = mock_config
    pipeline = MilvusMetadataFilteringPipeline("config.yaml")

    collection = pipeline._create_collection("test_coll", 128)

    assert collection == mock_milvus_components["collection_instance"]
    mock_milvus_components["Collection"].assert_called_with(
        name="test_coll", schema=mock_milvus_components["CollectionSchema"].return_value
    )
    mock_milvus_components["collection_instance"].create_index.assert_called_once()
    mock_milvus_components["collection_instance"].load.assert_called_once()


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_index_documents(
    mock_yaml_load, mock_file, mock_config, mock_milvus_components
):
    mock_yaml_load.return_value = mock_config
    pipeline = MilvusMetadataFilteringPipeline("config.yaml")
    pipeline.collection = mock_milvus_components["collection_instance"]

    docs = [
        Document(content="doc1", meta={"key": "val1"}, embedding=[0.1] * 128),
        Document(content="doc2", meta={"key": "val2"}, embedding=[0.2] * 128),
    ]

    count = pipeline._index_documents(docs)

    assert count == 2
    pipeline.collection.insert.assert_called_once()
    pipeline.collection.flush.assert_called_once()


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_pre_filter(mock_yaml_load, mock_file, mock_config, mock_milvus_components):
    mock_yaml_load.return_value = mock_config
    pipeline = MilvusMetadataFilteringPipeline("config.yaml")
    pipeline.collection = mock_milvus_components["collection_instance"]

    pipeline._pre_filter("expr")

    pipeline.collection.query.assert_called_with(expr="expr", output_fields=["id"])


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_vector_search(mock_yaml_load, mock_file, mock_config, mock_milvus_components):
    mock_yaml_load.return_value = mock_config
    pipeline = MilvusMetadataFilteringPipeline("config.yaml")
    pipeline.collection = mock_milvus_components["collection_instance"]

    # Mock search results
    mock_hit = MagicMock()
    mock_hit.id = 1
    mock_hit.score = 0.9
    mock_hit.entity.get.side_effect = lambda k, d: "content" if k == "content" else {}

    pipeline.collection.search.return_value = [[mock_hit]]

    results = pipeline._vector_search([0.1] * 128, "expr", top_k=5)

    assert len(results) == 1
    assert results[0]["id"] == 1
    assert results[0]["score"] == 0.9

    pipeline.collection.search.assert_called_once()


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_run_pipeline(
    mock_yaml_load, mock_file, mock_config, mock_milvus_components, mock_embedders
):
    mock_yaml_load.return_value = mock_config
    pipeline = MilvusMetadataFilteringPipeline("config.yaml")

    # Mock search results for run
    mock_hit = MagicMock()
    mock_hit.id = 1
    mock_hit.score = 0.95
    mock_hit.entity.get.side_effect = lambda k, d: (
        "content" if k == "content" else {"category": "news"}
    )
    mock_milvus_components["collection_instance"].search.return_value = [[mock_hit]]

    results = pipeline.run()

    assert len(results) == 1
    assert isinstance(results[0], FilteredQueryResult)
    assert results[0].relevance_score == 0.95
    assert results[0].document.content == "content"

    # Verify sequence
    mock_embedders["text_instance"].run.assert_called_once()
    mock_milvus_components["collection_instance"].query.assert_called()
    mock_milvus_components["collection_instance"].search.assert_called()


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_pre_filter_empty_expression(
    mock_yaml_load, mock_file, mock_config, mock_milvus_components
):
    """Test _pre_filter returns num_entities when filter_expr is empty."""
    mock_yaml_load.return_value = mock_config
    pipeline = MilvusMetadataFilteringPipeline("config.yaml")
    pipeline.collection = mock_milvus_components["collection_instance"]
    mock_milvus_components["collection_instance"].num_entities = 50

    result = pipeline._pre_filter("")

    assert result == 50
    pipeline.collection.query.assert_not_called()


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_pre_filter_no_collection(
    mock_yaml_load, mock_file, mock_config, mock_milvus_components
):
    """Test _pre_filter returns 0 when collection is None."""
    mock_yaml_load.return_value = mock_config
    pipeline = MilvusMetadataFilteringPipeline("config.yaml")
    pipeline.collection = None

    result = pipeline._pre_filter("category == 'news'")

    assert result == 0


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_vector_search_no_collection_raises(
    mock_yaml_load, mock_file, mock_config, mock_milvus_components
):
    """Test _vector_search raises ValueError when collection is None."""
    mock_yaml_load.return_value = mock_config
    pipeline = MilvusMetadataFilteringPipeline("config.yaml")
    pipeline.collection = None

    with pytest.raises(ValueError, match="Collection not initialized"):
        pipeline._vector_search([0.1] * 128, "expr", top_k=5)


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_index_documents_no_collection_raises(
    mock_yaml_load, mock_file, mock_config, mock_milvus_components
):
    """Test _index_documents raises ValueError when collection is None."""
    mock_yaml_load.return_value = mock_config
    pipeline = MilvusMetadataFilteringPipeline("config.yaml")
    pipeline.collection = None

    docs = [Document(content="doc1", meta={"key": "val1"}, embedding=[0.1] * 128)]

    with pytest.raises(ValueError, match="Collection not initialized"):
        pipeline._index_documents(docs)


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_vector_search_empty_filter_passes_none(
    mock_yaml_load, mock_file, mock_config, mock_milvus_components
):
    """Test _vector_search passes None when filter_expr is empty."""
    mock_yaml_load.return_value = mock_config
    pipeline = MilvusMetadataFilteringPipeline("config.yaml")
    pipeline.collection = mock_milvus_components["collection_instance"]
    pipeline.collection.search.return_value = []

    pipeline._vector_search([0.1] * 128, "", top_k=5)

    call_kwargs = pipeline.collection.search.call_args[1]
    assert call_kwargs["expr"] is None


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_run_pipeline_without_embedder_uses_zero_vector(
    mock_yaml_load, mock_file, mock_config, mock_milvus_components
):
    """Test run uses zero vector when embedder is None."""
    mock_yaml_load.return_value = mock_config
    pipeline = MilvusMetadataFilteringPipeline("config.yaml")
    pipeline.embedder = None
    mock_milvus_components["collection_instance"].search.return_value = []

    with patch.object(pipeline, "_init_embedder"):
        results = pipeline.run()

    assert results == []
    search_call = mock_milvus_components["collection_instance"].search.call_args
    query_vector = search_call[1]["data"][0]
    assert all(v == 0.0 for v in query_vector)


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_run_pipeline_multiple_results(
    mock_yaml_load, mock_file, mock_config, mock_milvus_components, mock_embedders
):
    """Test run pipeline with multiple search results."""
    mock_yaml_load.return_value = mock_config
    pipeline = MilvusMetadataFilteringPipeline("config.yaml")

    mock_hits = []
    for i in range(3):
        mock_hit = MagicMock()
        mock_hit.id = i + 1
        mock_hit.score = 0.9 - (i * 0.1)
        mock_hit.entity.get.side_effect = lambda k, d, idx=i: (
            f"content_{idx}" if k == "content" else {"rank": idx}
        )
        mock_hits.append(mock_hit)

    mock_milvus_components["collection_instance"].search.return_value = [mock_hits]

    results = pipeline.run()

    assert len(results) == 3
    assert results[0].rank == 1
    assert results[1].rank == 2
    assert results[2].rank == 3
    assert results[0].timing is not None
    assert results[1].timing is None
    assert results[2].timing is None


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_run_pipeline_empty_results(
    mock_yaml_load, mock_file, mock_config, mock_milvus_components, mock_embedders
):
    """Test run pipeline with no search results."""
    mock_yaml_load.return_value = mock_config
    pipeline = MilvusMetadataFilteringPipeline("config.yaml")
    mock_milvus_components["collection_instance"].search.return_value = [[]]

    results = pipeline.run()

    assert results == []


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_connect_with_default_config(mock_yaml_load, mock_file, mock_milvus_components):
    """Test _connect uses default host/port when not in config."""
    config = {"milvus": {}, "logging": {"name": "test", "level": "DEBUG"}}
    mock_yaml_load.return_value = config

    MilvusMetadataFilteringPipeline("config.yaml")

    mock_milvus_components["connections"].connect.assert_called_with(
        alias="default", host="localhost", port="19530"
    )


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_index_documents_with_none_embeddings(
    mock_yaml_load, mock_file, mock_config, mock_milvus_components
):
    """Test _index_documents filters out documents without embeddings."""
    mock_yaml_load.return_value = mock_config
    pipeline = MilvusMetadataFilteringPipeline("config.yaml")
    pipeline.collection = mock_milvus_components["collection_instance"]

    docs = [
        Document(content="doc1", meta={"key": "val1"}, embedding=[0.1] * 128),
        Document(content="doc2", meta={"key": "val2"}, embedding=None),
        Document(content="doc3", meta={"key": "val3"}, embedding=[0.3] * 128),
    ]

    count = pipeline._index_documents(docs)

    assert count == 3
    insert_args = pipeline.collection.insert.call_args[0][0]
    embeddings = insert_args[0]
    assert len(embeddings) == 2


@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load")
def test_index_documents_with_none_content(
    mock_yaml_load, mock_file, mock_config, mock_milvus_components
):
    """Test _index_documents handles documents with None content."""
    mock_yaml_load.return_value = mock_config
    pipeline = MilvusMetadataFilteringPipeline("config.yaml")
    pipeline.collection = mock_milvus_components["collection_instance"]

    docs = [
        Document(content=None, meta={"key": "val1"}, embedding=[0.1] * 128),
    ]

    count = pipeline._index_documents(docs)

    assert count == 1
    insert_args = pipeline.collection.insert.call_args[0][0]
    contents = insert_args[1]
    assert contents == [""]
