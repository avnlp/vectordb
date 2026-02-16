"""Tests for the Chroma document converter module."""

import pytest
from haystack import Document as HaystackDocument
from langchain_core.documents import Document as LangchainDocument

from vectordb.utils.chroma_document_converter import ChromaDocumentConverter


class TestChromaDocumentConverter:
    """Test cases for ChromaDocumentConverter class."""

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

        result = ChromaDocumentConverter.prepare_haystack_documents_for_upsert(
            documents
        )

        assert result["texts"] == ["Test document 1", "Test document 2"]
        assert result["embeddings"] == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert result["ids"] == ["doc1", "doc2"]
        assert result["metadatas"] == [{"source": "test"}, {"source": "test2"}]

    def test_prepare_haystack_documents_empty_list_raises_error(self):
        """Test that empty document list raises ValueError."""
        with pytest.raises(ValueError, match="document list is empty"):
            ChromaDocumentConverter.prepare_haystack_documents_for_upsert([])

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

        with pytest.raises(ValueError, match="lacks embeddings"):
            ChromaDocumentConverter.prepare_haystack_documents_for_upsert(documents)

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

        result = ChromaDocumentConverter.prepare_haystack_documents_for_upsert(
            documents
        )

        assert result["metadatas"][0]["tags"] == "tag1$ $tag2$ $tag3"

    def test_prepare_langchain_documents_for_upsert_success(self):
        """Test preparing valid LangChain documents for upsert."""
        documents = [
            LangchainDocument(page_content="Test doc 1", metadata={"source": "test1"}),
            LangchainDocument(page_content="Test doc 2", metadata={"source": "test2"}),
        ]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        result = ChromaDocumentConverter.prepare_langchain_documents_for_upsert(
            documents, embeddings
        )

        assert result["texts"] == ["Test doc 1", "Test doc 2"]
        assert result["embeddings"] == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert result["ids"] == ["0", "1"]
        assert result["metadatas"] == [{"source": "test1"}, {"source": "test2"}]

    def test_prepare_langchain_documents_empty_list_raises_error(self):
        """Test that empty LangChain document list raises ValueError."""
        with pytest.raises(ValueError, match="document list is empty"):
            ChromaDocumentConverter.prepare_langchain_documents_for_upsert([], [])

    def test_prepare_langchain_documents_joins_list_metadata(self):
        """Test that list metadata values in LangChain docs are joined."""
        documents = [
            LangchainDocument(
                page_content="Test doc", metadata={"tags": ["a", "b", "c"]}
            ),
        ]
        embeddings = [[0.1, 0.2, 0.3]]

        result = ChromaDocumentConverter.prepare_langchain_documents_for_upsert(
            documents, embeddings
        )

        assert result["metadatas"][0]["tags"] == "a$ $b$ $c"

    def test_convert_query_results_to_haystack_documents(self):
        """Test converting query results to Haystack documents."""
        query_results = {
            "documents": [["Result 1"], ["Result 2"]],
            "metadatas": [[{"score": 0.9}], [{"score": 0.8}]],
        }

        result = ChromaDocumentConverter.convert_query_results_to_haystack_documents(
            query_results
        )

        assert len(result) == 2
        assert result[0].content == "Result 1"
        assert result[0].meta == {"score": 0.9}
        assert result[1].content == "Result 2"
        assert result[1].meta == {"score": 0.8}

    def test_convert_query_results_to_langchain_documents(self):
        """Test converting query results to LangChain documents."""
        query_results = {
            "documents": [["Content 1"], ["Content 2"]],
            "metadatas": [[{"key": "val1"}], [{"key": "val2"}]],
        }

        result = ChromaDocumentConverter.convert_query_results_to_langchain_documents(
            query_results
        )

        assert len(result) == 2
        assert result[0].page_content == "Content 1"
        assert result[0].metadata == {"key": "val1"}
        assert result[1].page_content == "Content 2"
        assert result[1].metadata == {"key": "val2"}
