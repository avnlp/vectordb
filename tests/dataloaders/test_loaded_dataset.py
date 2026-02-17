"""Unit tests for LoadedDataset."""

from unittest.mock import patch

from haystack import Document as HaystackDocument
from langchain_core.documents import Document as LangChainDocument

from vectordb.dataloaders.dataset import LoadedDataset
from vectordb.dataloaders.types import DatasetRecord


class TestLoadedDatasetConversion:
    """Tests for document conversion helpers."""

    def test_records_roundtrip(self) -> None:
        """Test that records are returned unchanged via the records() method."""
        records = [DatasetRecord(text="text", metadata={"id": 1})]
        dataset = LoadedDataset("arc", records)

        assert dataset.records() == records

    def test_converter_delegation(self) -> None:
        """Test that to_haystack() and to_langchain() delegate to DocumentConverter."""
        records = [DatasetRecord(text="text", metadata={"id": 1})]
        dataset = LoadedDataset("arc", records)

        with patch("vectordb.dataloaders.dataset.DocumentConverter") as converter:
            converter.to_haystack.return_value = [HaystackDocument(content="text")]
            converter.to_langchain.return_value = [
                LangChainDocument(page_content="text")
            ]

            haystack_docs = dataset.to_haystack()
            langchain_docs = dataset.to_langchain()

        assert haystack_docs[0].content == "text"
        assert langchain_docs[0].page_content == "text"


class TestLoadedDatasetEvaluationQueries:
    """Tests for evaluation query extraction."""

    def test_evaluation_dedup_and_limit(self) -> None:
        """Test queries are deduplicated case-insensitively and respect limit."""
        records = [
            DatasetRecord(text="", metadata={"question": " Who? ", "answers": ["A"]}),
            DatasetRecord(text="", metadata={"question": "who?", "answers": ["B"]}),
            DatasetRecord(text="", metadata={"question": "What?", "answer": "C"}),
        ]
        dataset = LoadedDataset("arc", records)

        queries = dataset.evaluation_queries(limit=1)

        assert len(queries) == 1
        assert queries[0].query.strip().casefold() == "who?"
