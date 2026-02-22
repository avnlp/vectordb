"""Tests for LangChain namespace types and sampling utilities."""

from unittest.mock import patch

from langchain_core.documents import Document

from vectordb.langchain.namespaces.types import NamespaceNameGenerator, QuerySampler


class TestNamespaceNameGenerator:
    """Tests for NamespaceNameGenerator helpers."""

    def test_from_split_without_prefix(self) -> None:
        """Build namespace names from dataset and split."""
        assert NamespaceNameGenerator.from_split("arc", "train") == "arc_train"

    def test_from_split_with_prefix(self) -> None:
        """Build namespace names with an explicit prefix."""
        assert NamespaceNameGenerator.from_split("arc", "train", "qa") == "qa_arc_train"

    def test_from_ticker(self) -> None:
        """Build namespace names from ticker symbols."""
        assert NamespaceNameGenerator.from_ticker("AAPL") == "earnings_AAPL"

    def test_from_config_extracts_only_named_definitions(self) -> None:
        """Extract only namespace definitions that contain a name."""
        config = {
            "namespaces": {
                "definitions": [
                    {"name": "train_ns", "split": "train"},
                    {"description": "missing-name"},
                    {"name": "test_ns", "split": "test"},
                ]
            }
        }

        assert NamespaceNameGenerator.from_config(config) == ["train_ns", "test_ns"]


class TestQuerySampler:
    """Tests for QuerySampler.sample_from_documents."""

    def test_sample_from_documents_prefers_metadata_field(self) -> None:
        """Sample query strings from metadata when the field exists."""
        documents = [
            Document(page_content="fallback A", metadata={"question": "What is ML?"}),
            Document(
                page_content="fallback B", metadata={"question": "How does AI work?"}
            ),
            Document(page_content="fallback C", metadata={"question": "Explain NLP"}),
        ]

        sampled = QuerySampler.sample_from_documents(
            documents,
            sample_size=2,
            query_field="question",
            seed=42,
        )

        assert len(sampled) == 2
        assert set(sampled).issubset(
            {"What is ML?", "How does AI work?", "Explain NLP"}
        )

    def test_sample_from_documents_uses_content_fallback(self) -> None:
        """Use page_content slice when metadata query field is absent."""
        long_text = "x" * 140
        documents = [Document(page_content=long_text, metadata={})]

        sampled = QuerySampler.sample_from_documents(documents, sample_size=1, seed=7)

        assert sampled == [long_text[:100]]

    def test_sample_from_documents_returns_empty_for_no_queries(self) -> None:
        """Return an empty list when neither metadata nor content is usable."""
        documents = [Document(page_content="", metadata={})]

        assert QuerySampler.sample_from_documents(documents, sample_size=5) == []

    @patch("vectordb.langchain.namespaces.types.random.seed")
    @patch("vectordb.langchain.namespaces.types.random.sample")
    def test_sample_from_documents_sets_seed_and_caps_sample_size(
        self,
        mock_sample,
        mock_seed,
    ) -> None:
        """Set deterministic seed and cap requested sample size to available data."""
        documents = [
            Document(page_content="d1", metadata={"question": "q1"}),
            Document(page_content="d2", metadata={"question": "q2"}),
        ]
        mock_sample.return_value = ["q1", "q2"]

        sampled = QuerySampler.sample_from_documents(
            documents, sample_size=10, seed=123
        )

        assert sampled == ["q1", "q2"]
        mock_seed.assert_called_once_with(123)
        mock_sample.assert_called_once()
        sampled_queries, requested_size = mock_sample.call_args.args
        assert sampled_queries == ["q1", "q2"]
        assert requested_size == 2
