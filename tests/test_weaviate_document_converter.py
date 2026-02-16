"""Tests for the Weaviate document converter module."""

import pytest
from haystack import Document as HaystackDocument
from langchain_core.documents import Document as LangchainDocument

from vectordb.utils.weaviate_document_converter import WeaviateDocumentConverter


class MockQueryResultObject:
    """Mock object for Weaviate query results."""

    def __init__(self, properties, score=None):
        self.properties = properties
        self.metadata = MockMetadata(score)


class MockMetadata:
    """Mock metadata for query results."""

    def __init__(self, score):
        self.score = score


class MockQueryReturn:
    """Mock QueryReturn for Weaviate query results."""

    def __init__(self, objects=None):
        self.objects = objects or []


class TestWeaviateDocumentConverter:
    """Test cases for WeaviateDocumentConverter class."""

    def test_prepare_haystack_documents_for_upsert_success(self):
        """Test preparing valid Haystack documents for upsert."""
        documents = [
            HaystackDocument(
                content="Test document 1",
                id="doc1",
                embedding=[0.1, 0.2, 0.3],
                meta={"source": "test"},
            ),
            HaystackDocument(
                content="Test document 2",
                id="doc2",
                embedding=[0.4, 0.5, 0.6],
                meta={"source": "test2"},
            ),
        ]

        result = WeaviateDocumentConverter.prepare_haystack_documents_for_upsert(
            documents
        )

        assert len(result) == 2
        assert result[0]["vector"] == [0.1, 0.2, 0.3]
        assert result[0]["text"] == "Test document 1"
        assert result[0]["metadata"]["source"] == "test"
        assert result[1]["vector"] == [0.4, 0.5, 0.6]
        assert result[1]["text"] == "Test document 2"
        # UUID should be generated
        assert "uuid" in result[0]
        assert "uuid" in result[1]

    def test_prepare_haystack_documents_missing_embedding_raises_error(self):
        """Test that documents without embeddings raise ValueError."""
        documents = [
            HaystackDocument(
                content="Test document",
                id="doc1",
                embedding=None,
                meta={"source": "test"},
            ),
        ]

        with pytest.raises(ValueError, match="must have a dense embedding"):
            WeaviateDocumentConverter.prepare_haystack_documents_for_upsert(documents)

    def test_prepare_haystack_documents_joins_list_metadata(self):
        """Test that list metadata values are joined with separator."""
        documents = [
            HaystackDocument(
                content="Test document",
                id="doc1",
                embedding=[0.1, 0.2, 0.3],
                meta={"tags": ["tag1", "tag2", "tag3"]},
            ),
        ]

        result = WeaviateDocumentConverter.prepare_haystack_documents_for_upsert(
            documents
        )

        assert result[0]["metadata"]["tags"] == "tag1$ $tag2$ $tag3"

    def test_prepare_haystack_documents_renames_id_metadata(self):
        """Test that 'id' metadata key is renamed to 'document_id'."""
        documents = [
            HaystackDocument(
                content="Test document",
                id="doc1",
                embedding=[0.1, 0.2, 0.3],
                meta={"id": "original_id", "source": "test"},
            ),
        ]

        result = WeaviateDocumentConverter.prepare_haystack_documents_for_upsert(
            documents
        )

        assert "id" not in result[0]["metadata"]
        assert result[0]["metadata"]["document_id"] == "original_id"
        assert result[0]["metadata"]["source"] == "test"

    def test_prepare_langchain_documents_for_upsert_success(self):
        """Test preparing valid LangChain documents for upsert."""
        documents = [
            LangchainDocument(page_content="Test doc 1", metadata={"source": "test1"}),
            LangchainDocument(page_content="Test doc 2", metadata={"source": "test2"}),
        ]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        result = WeaviateDocumentConverter.prepare_langchain_documents_for_upsert(
            documents, embeddings
        )

        assert len(result) == 2
        assert result[0]["vector"] == [0.1, 0.2, 0.3]
        assert result[0]["text"] == "Test doc 1"
        assert result[0]["metadata"]["source"] == "test1"
        assert result[1]["vector"] == [0.4, 0.5, 0.6]
        assert result[1]["text"] == "Test doc 2"
        # UUID should be generated
        assert "uuid" in result[0]
        assert "uuid" in result[1]

    def test_prepare_langchain_documents_joins_list_metadata(self):
        """Test that list metadata values in LangChain docs are joined."""
        documents = [
            LangchainDocument(
                page_content="Test doc", metadata={"tags": ["a", "b", "c"]}
            ),
        ]
        embeddings = [[0.1, 0.2, 0.3]]

        result = WeaviateDocumentConverter.prepare_langchain_documents_for_upsert(
            documents, embeddings
        )

        assert result[0]["metadata"]["tags"] == "a$ $b$ $c"

    def test_prepare_langchain_documents_renames_id_metadata(self):
        """Test that 'id' metadata key is renamed to 'document_id' in LangChain docs."""
        documents = [
            LangchainDocument(
                page_content="Test doc",
                metadata={"id": "original_id", "source": "test"},
            ),
        ]
        embeddings = [[0.1, 0.2, 0.3]]

        result = WeaviateDocumentConverter.prepare_langchain_documents_for_upsert(
            documents, embeddings
        )

        assert "id" not in result[0]["metadata"]
        assert result[0]["metadata"]["document_id"] == "original_id"
        assert result[0]["metadata"]["source"] == "test"

    def test_convert_query_results_to_haystack_documents(self):
        """Test converting query results to Haystack documents."""
        query_results = MockQueryReturn(
            objects=[
                MockQueryResultObject(
                    properties={"text": "Content 1", "metadata": {"source": "test1"}},
                    score=0.95,
                ),
                MockQueryResultObject(
                    properties={"text": "Content 2", "metadata": {"source": "test2"}},
                    score=0.85,
                ),
            ]
        )

        result = WeaviateDocumentConverter.convert_query_results_to_haystack_documents(
            query_results
        )

        assert len(result) == 2
        assert result[0].content == "Content 1"
        assert result[0].meta["source"] == "test1"
        assert result[0].score == 0.95
        assert result[1].content == "Content 2"
        assert result[1].meta["source"] == "test2"
        assert result[1].score == 0.85

    def test_convert_query_results_to_haystack_documents_empty(self):
        """Test converting empty query results returns empty list."""
        query_results = MockQueryReturn(objects=[])

        result = WeaviateDocumentConverter.convert_query_results_to_haystack_documents(
            query_results
        )

        assert result == []

    def test_convert_query_results_to_haystack_documents_none(self):
        """Test converting None query results returns empty list."""
        query_results = MockQueryReturn(objects=None)

        result = WeaviateDocumentConverter.convert_query_results_to_haystack_documents(
            query_results
        )

        assert result == []

    def test_convert_query_results_to_langchain_documents(self):
        """Test converting query results to LangChain documents."""
        query_results = MockQueryReturn(
            objects=[
                MockQueryResultObject(
                    properties={"text": "Content 1", "metadata": {"source": "test1"}},
                    score=0.95,
                ),
                MockQueryResultObject(
                    properties={"text": "Content 2", "metadata": {"source": "test2"}},
                    score=0.85,
                ),
            ]
        )

        result = WeaviateDocumentConverter.convert_query_results_to_langchain_documents(
            query_results
        )

        assert len(result) == 2
        assert result[0].page_content == "Content 1"
        assert result[0].metadata["source"] == "test1"
        assert result[1].page_content == "Content 2"
        assert result[1].metadata["source"] == "test2"

    def test_convert_query_results_to_langchain_documents_empty(self):
        """Test converting empty query results to LangChain returns empty list."""
        query_results = MockQueryReturn(objects=[])

        result = WeaviateDocumentConverter.convert_query_results_to_langchain_documents(
            query_results
        )

        assert result == []

    def test_convert_query_results_to_langchain_documents_none(self):
        """Test converting None query results to LangChain returns empty list."""
        query_results = MockQueryReturn(objects=None)

        result = WeaviateDocumentConverter.convert_query_results_to_langchain_documents(
            query_results
        )

        assert result == []
