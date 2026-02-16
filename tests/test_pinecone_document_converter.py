"""Tests for the Pinecone document converter module."""

import pytest
from haystack import Document as HaystackDocument
from haystack.dataclasses import SparseEmbedding
from langchain_core.documents import Document as LangchainDocument

from vectordb.utils.pinecone_document_converter import PineconeDocumentConverter


class TestPineconeDocumentConverter:
    """Test cases for PineconeDocumentConverter class."""

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

        result = PineconeDocumentConverter.prepare_haystack_documents_for_upsert(
            documents
        )

        assert len(result) == 2
        assert result[0]["id"] == "doc1"
        assert result[0]["values"] == [0.1, 0.2, 0.3]
        assert result[0]["metadata"]["text"] == "Test document 1"
        assert result[0]["metadata"]["source"] == "test"
        assert result[1]["id"] == "doc2"
        assert result[1]["values"] == [0.4, 0.5, 0.6]

    def test_prepare_haystack_documents_with_sparse_embeddings(self):
        """Test preparing documents with sparse embeddings."""
        sparse_embed = SparseEmbedding(indices=[0, 2, 4], values=[0.1, 0.3, 0.5])
        documents = [
            HaystackDocument(
                content="Test document",
                id="doc1",
                embedding=[0.1, 0.2, 0.3],
                meta={"source": "test"},
                sparse_embedding=sparse_embed,
            ),
        ]

        result = PineconeDocumentConverter.prepare_haystack_documents_for_upsert(
            documents
        )

        assert "sparse_values" in result[0]
        assert result[0]["sparse_values"]["indices"] == [0, 2, 4]
        assert result[0]["sparse_values"]["values"] == [0.1, 0.3, 0.5]

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

        with pytest.raises(ValueError, match="must have an ID and a dense embedding"):
            PineconeDocumentConverter.prepare_haystack_documents_for_upsert(documents)

    def test_prepare_langchain_documents_for_upsert_success(self):
        """Test preparing valid LangChain documents for upsert."""
        documents = [
            LangchainDocument(page_content="Test doc 1", metadata={"source": "test1"}),
            LangchainDocument(page_content="Test doc 2", metadata={"source": "test2"}),
        ]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        result = PineconeDocumentConverter.prepare_langchain_documents_for_upsert(
            documents, embeddings
        )

        assert len(result) == 2
        assert result[0]["id"] == "0"
        assert result[0]["values"] == [0.1, 0.2, 0.3]
        assert result[0]["metadata"]["text"] == "Test doc 1"
        assert result[1]["id"] == "1"
        assert result[1]["values"] == [0.4, 0.5, 0.6]

    def test_prepare_langchain_documents_with_sparse_embeddings(self):
        """Test preparing LangChain documents with sparse embeddings."""
        from langchain_qdrant.sparse_embeddings import SparseVector

        documents = [
            LangchainDocument(page_content="Test doc", metadata={"source": "test"}),
        ]
        embeddings = [[0.1, 0.2, 0.3]]
        sparse_embeddings = [SparseVector(indices=[0, 1], values=[0.5, 0.5])]

        result = PineconeDocumentConverter.prepare_langchain_documents_for_upsert(
            documents, embeddings, sparse_embeddings
        )

        assert "sparse_values" in result[0]
        assert result[0]["sparse_values"]["indices"] == [0, 1]
        assert result[0]["sparse_values"]["values"] == [0.5, 0.5]

    def test_prepare_langchain_documents_without_sparse_embeddings(self):
        """Test preparing LangChain documents without sparse embeddings."""
        documents = [
            LangchainDocument(page_content="Test doc", metadata={"source": "test"}),
        ]
        embeddings = [[0.1, 0.2, 0.3]]

        result = PineconeDocumentConverter.prepare_langchain_documents_for_upsert(
            documents, embeddings
        )

        assert "sparse_values" not in result[0]

    def test_convert_query_results_to_haystack_documents(self):
        """Test converting query results to Haystack documents."""
        query_results = {
            "matches": [
                {
                    "id": "doc1",
                    "values": [0.1, 0.2, 0.3],
                    "score": 0.95,
                    "metadata": {"text": "Content 1", "source": "test1"},
                },
                {
                    "id": "doc2",
                    "values": [0.4, 0.5, 0.6],
                    "score": 0.85,
                    "metadata": {"text": "Content 2", "source": "test2"},
                },
            ]
        }

        result = PineconeDocumentConverter.convert_query_results_to_haystack_documents(
            query_results
        )

        assert len(result) == 2
        assert result[0].id == "doc1"
        assert result[0].content == "Content 1"
        assert result[0].score == 0.95
        assert result[0].embedding == [0.1, 0.2, 0.3]
        assert result[1].id == "doc2"
        assert result[1].content == "Content 2"

    def test_convert_query_results_to_haystack_documents_with_sparse(self):
        """Test converting query results with sparse embeddings."""
        query_results = {
            "matches": [
                {
                    "id": "doc1",
                    "values": [0.1, 0.2, 0.3],
                    "score": 0.95,
                    "metadata": {"text": "Content 1"},
                    "sparse_values": {"indices": [0, 2], "values": [0.5, 0.5]},
                },
            ]
        }

        result = PineconeDocumentConverter.convert_query_results_to_haystack_documents(
            query_results
        )

        assert result[0].sparse_embedding is not None
        assert result[0].sparse_embedding.indices == [0, 2]
        assert result[0].sparse_embedding.values == [0.5, 0.5]

    def test_convert_query_results_to_haystack_documents_empty_matches(self):
        """Test converting empty query results."""
        query_results = {"matches": []}

        result = PineconeDocumentConverter.convert_query_results_to_haystack_documents(
            query_results
        )

        assert result == []

    def test_convert_query_results_to_langchain_documents(self):
        """Test converting query results to LangChain documents."""
        query_results = {
            "matches": [
                {
                    "id": "doc1",
                    "metadata": {"text": "Content 1", "source": "test1"},
                },
            ]
        }

        result = PineconeDocumentConverter.convert_query_results_to_langchain_documents(
            query_results
        )

        assert len(result) == 1
        assert result[0].content == "Content 1"
        assert result[0].meta["source"] == "test1"
