"""Unit tests for evaluation extractor."""

from vectordb.dataloaders.evaluation import EvaluationExtractor
from vectordb.dataloaders.types import DatasetRecord


class TestEvaluationExtractor:
    """Tests for evaluation query extraction."""

    def test_query_extraction_with_fallback(self) -> None:
        """Test query extraction falls back from 'question' to 'entity' field.

        Verifies that the extractor correctly retrieves queries from different
        metadata fields, preferring 'question' and falling back to 'entity'.
        """
        records = [
            DatasetRecord(text="", metadata={"question": "Who?", "answers": ["A"]}),
            DatasetRecord(text="", metadata={"entity": "Entity", "answer": "B"}),
        ]

        queries = EvaluationExtractor.extract(records)

        assert [query.query for query in queries] == ["Who?", "Entity"]

    def test_answers_normalization(self) -> None:
        """Test answers are normalized from both 'answer' and 'answers' fields.

        Verifies that single 'answer' strings and 'answers' lists are both
        processed correctly, with whitespace-only answers filtered out.
        """
        records = [
            DatasetRecord(text="", metadata={"question": "Q", "answer": "A"}),
            DatasetRecord(text="", metadata={"question": "R", "answers": ["B", " "]}),
        ]

        queries = EvaluationExtractor.extract(records)

        assert queries[0].answers == ["A"]
        assert queries[1].answers == ["B"]

    def test_dedup_normalized_query(self) -> None:
        """Test duplicate queries are deduplicated after normalization.

        Verifies that queries with different casing or surrounding whitespace
        are normalized and deduplicated, keeping only the first occurrence.
        """
        records = [
            DatasetRecord(text="", metadata={"question": " Who? ", "answer": "A"}),
            DatasetRecord(text="", metadata={"question": "who?", "answer": "B"}),
        ]

        queries = EvaluationExtractor.extract(records)

        assert len(queries) == 1
        assert queries[0].answers == ["A"]

    def test_limit_after_dedup(self) -> None:
        """Test limit parameter restricts results after deduplication.

        Verifies that the limit parameter correctly caps the number of
        returned queries after any deduplication has been applied.
        """
        records = [
            DatasetRecord(text="", metadata={"question": "Q1", "answer": "A"}),
            DatasetRecord(text="", metadata={"question": "Q2", "answer": "B"}),
        ]

        queries = EvaluationExtractor.extract(records, limit=1)

        assert len(queries) == 1
