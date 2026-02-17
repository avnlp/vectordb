"""Shared fixtures for dataloader tests."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def arc_sample_rows() -> list[dict[str, object]]:
    """Provide sample ARC dataset rows."""
    return [
        {
            "id": "arc_1",
            "question": "What is the capital of France?",
            "choices": {
                "label": ["A", "B", "C", "D"],
                "text": ["Paris", "Lyon", "Marseille", "Nice"],
            },
            "answerKey": "A",
        },
        {
            "id": "arc_2",
            "question": "What is 2+2?",
            "choices": {
                "label": ["A", "B", "C", "D"],
                "text": ["3", "4", "5", "6"],
            },
            "answerKey": "B",
        },
    ]


@pytest.fixture
def triviaqa_sample_rows() -> list[dict[str, object]]:
    """Provide sample TriviaQA dataset rows."""
    return [
        {
            "question": "What is the capital of France?",
            "answer": ["Paris", "City of Light"],
            "search_results": {
                "rank": [1, 2],
                "title": ["Paris (France)", "Paris (Texas)"],
                "search_context": [
                    "Paris is the capital of France.",
                    "Paris, Texas is a city in Texas.",
                ],
                "description": ["A city in France", "A city in Texas"],
            },
        },
        {
            "question": "Who was the first president?",
            "answer": "George Washington",
            "search_results": {
                "rank": [1],
                "title": ["George Washington"],
                "search_context": ["George Washington was the first US president."],
                "description": ["The first president"],
            },
        },
    ]


@pytest.fixture
def popqa_sample_rows() -> list[dict[str, object]]:
    """Provide sample PopQA dataset rows."""
    return [
        {
            "question": "What is the capital of France?",
            "possible_answers": ["Paris"],
            "subj": "France",
            "prop": "capital",
            "obj": "Paris",
            "content": "Paris is the capital and largest city of France.",
        },
        {
            "question": "What is the largest planet?",
            "possible_answers": ["Jupiter"],
            "subj": "Solar System",
            "prop": "largest_planet",
            "obj": "Jupiter",
            "content": "Jupiter is the largest planet in the solar system.",
        },
    ]


@pytest.fixture
def factscore_sample_rows() -> list[dict[str, object]]:
    """Provide sample FactScore dataset rows."""
    return [
        {
            "id": "fact_1",
            "topic": "Albert Einstein",
            "entity": "Albert Einstein",
            "wikipedia_text": "Albert Einstein was a German-born physicist.",
            "one_fact_prompt": "Tell me one fact about Albert Einstein.",
            "factscore_prompt": "Evaluate the facts about Albert Einstein.",
            "facts": ["Einstein was born in Germany", "He won the Nobel Prize"],
            "decomposed_facts": [
                {"sent_id": 0, "fact": "Einstein was born in Germany"},
                {"sent_id": 1, "fact": "He won the Nobel Prize"},
            ],
        }
    ]


@pytest.fixture
def earnings_calls_sample_rows() -> list[dict[str, object]]:
    """Provide sample earnings calls dataset rows."""
    return [
        {
            "question": "What was the revenue?",
            "answer": "10 billion",
            "date": "2024-01-15",
            "transcript": "Q4 earnings call transcript for Acme Corp.",
            "q": "2023-Q4",
            "ticker": "ACME",
            "company": "Acme Corp",
            "id": "call_1",
        },
        {
            "question": "What was the profit?",
            "answer": "2 billion",
            "date": "2024-02-20",
            "transcript": "Q1 earnings call transcript for TechCorp.",
            "q": "2024-Q1",
            "ticker": "TECH",
            "company": "TechCorp",
            "id": "call_2",
        },
    ]


@pytest.fixture
def make_streaming_dataset() -> MagicMock:
    """Return a helper that produces a streaming dataset mock."""

    def _factory(rows: list[dict[str, object]]) -> MagicMock:
        dataset = MagicMock()
        dataset.__iter__ = MagicMock(return_value=iter(rows))
        return dataset

    return _factory


@pytest.fixture
def triviaqa_edge_missing_context() -> list[dict[str, object]]:
    """Provide TriviaQA rows missing search context."""
    return [
        {
            "question": "Fallback question",
            "answer": "Fallback answer",
            "search_results": {
                "rank": [1],
                "title": ["Fallback"],
                "search_context": [],
                "description": ["Fallback description"],
            },
        }
    ]


@pytest.fixture
def popqa_edge_missing_content() -> list[dict[str, object]]:
    """Provide PopQA rows without content field."""
    return [
        {
            "question": "What is the capital of France?",
            "possible_answers": ["Paris"],
            "subj": "France",
            "prop": "capital",
            "obj": "Paris",
        }
    ]


@pytest.fixture
def factscore_edge_missing_optional() -> list[dict[str, object]]:
    """Provide FactScore rows missing optional fields."""
    return [
        {
            "entity": "Ada Lovelace",
            "wikipedia_text": "Ada Lovelace was a mathematician.",
        }
    ]


@pytest.fixture
def earnings_calls_edge_bad_q() -> list[dict[str, object]]:
    """Provide earnings calls rows with malformed quarter."""
    return [
        {
            "question": "Malformed quarter?",
            "answer": "Unknown",
            "date": "2024-01-15",
            "transcript": "Transcript",
            "q": "not-a-quarter",
            "ticker": "ACME",
        }
    ]
