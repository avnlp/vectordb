"""Shared fixtures for LangChain dataloader tests.

This module provides pytest fixtures specifically designed for testing
LangChain-based dataloader integrations. These fixtures create mock data
structures and components that simulate the LangChain framework's expected
interfaces and data formats.

Fixtures:
    mock_langchain_document: Sample LangChain Document object.
    langchain_arc_sample_rows: ARC rows adapted for LangChain processing.
    langchain_triviaqa_sample_rows: TriviaQA rows with LangChain context format.
    langchain_popqa_sample_rows: PopQA rows for LangChain pipeline testing.
    langchain_factscore_sample_rows: FactScore rows with LangChain-compatible fields.
    langchain_earnings_calls_sample_rows: Earnings calls data for LangChain tests.
    mock_text_splitter: Mock LangChain text splitter component.
    mock_groq_generator: Mock Groq LLM generator for LangChain chains.

Note:
    LangChain fixtures use 'page_content' and 'metadata' structure
    which are the standard fields for LangChain Document objects.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document


@pytest.fixture
def mock_langchain_document():
    """Create a mock LangChain Document."""
    return Document(
        page_content="Test content",
        metadata={"source": "test", "page": 1},
    )


@pytest.fixture
def langchain_arc_sample_rows():
    """Create sample ARC rows for LangChain tests."""
    return [
        {
            "id": "arc_1",
            "question": "What is the capital of France?",
            "choices": {
                "label": ["A", "B", "C", "D"],
                "text": ["Paris", "Lyon", "Marseille", "Nice"],
            },
            "answerKey": "A",
            "ctxs": [
                {
                    "id": "ctx_1",
                    "title": "France",
                    "text": "France is a country in Europe.",
                },
                {
                    "id": "ctx_2",
                    "title": "Paris",
                    "text": "Paris is the capital of France.",
                },
            ],
        },
        {
            "id": "arc_2",
            "question": "What is 2+2?",
            "choices": {
                "label": ["A", "B", "C", "D"],
                "text": ["3", "4", "5", "6"],
            },
            "answerKey": "B",
            "ctxs": [
                {
                    "id": "ctx_3",
                    "title": "Arithmetic",
                    "text": "Two plus two equals four.",
                }
            ],
        },
    ]


@pytest.fixture
def langchain_triviaqa_sample_rows():
    """Create sample TriviaQA rows for LangChain tests."""
    return [
        {
            "question": "What is the capital of France?",
            "answers": ["Paris"],
            "ctxs": [
                {
                    "id": "ctx_1",
                    "title": "Paris (France)",
                    "text": "Paris is the capital of France.",
                    "rank": 1,
                },
                {
                    "id": "ctx_2",
                    "title": "Paris (Texas)",
                    "text": "Paris, Texas is a city in Texas.",
                    "rank": 2,
                },
            ],
        },
        {
            "question": "Who was the first president?",
            "answers": ["George Washington"],
            "ctxs": [
                {
                    "id": "ctx_3",
                    "title": "George Washington",
                    "text": "George Washington was the first US president.",
                    "rank": 1,
                },
            ],
        },
    ]


@pytest.fixture
def langchain_popqa_sample_rows():
    """Create sample PopQA rows for LangChain tests."""
    return [
        {
            "question": "What is the capital of France?",
            "answers": ["Paris"],
            "subj": "France",
            "prop": "capital",
            "obj": "Paris",
            "ctxs": [
                {
                    "id": "ctx_1",
                    "title": "France",
                    "text": "Paris is the capital and largest city of France.",
                },
                {
                    "id": "ctx_2",
                    "title": "Geography",
                    "text": "France is a country in Western Europe.",
                },
            ],
        },
        {
            "question": "What is the largest planet?",
            "answers": ["Jupiter"],
            "subj": "Solar System",
            "prop": "largest_planet",
            "obj": "Jupiter",
            "ctxs": [
                {
                    "id": "ctx_3",
                    "title": "Solar System",
                    "text": "Jupiter is the largest planet in the solar system.",
                },
            ],
        },
    ]


@pytest.fixture
def langchain_factscore_sample_rows():
    """Create sample FactScore rows for LangChain tests."""
    return [
        {
            "question": "Tell me about Albert Einstein",
            "answers": ["Einstein was a physicist"],
            "entity": "Albert Einstein",
            "topic": "Albert Einstein",
            "id": "fact_1",
            "facts": ["Einstein was born in Germany", "He won the Nobel Prize"],
            "decomposed_facts": [
                {"sent_id": 0, "fact": "Einstein was born in Germany"},
                {"sent_id": 1, "fact": "He won the Nobel Prize"},
            ],
            "ctxs": [
                {
                    "id": "ctx_1",
                    "title": "Albert Einstein",
                    "text": "Albert Einstein was a German-born physicist.",
                },
                {
                    "id": "ctx_2",
                    "title": "Einstein's Theory",
                    "text": "He developed the theory of relativity.",
                },
            ],
        },
    ]


@pytest.fixture
def langchain_earnings_calls_sample_rows():
    """Create sample earnings calls rows for LangChain tests."""
    return [
        {
            "question": "What was the revenue?",
            "answer": "10 billion",
            "date": "2024-01-15",
            "transcript": "Q4 earnings call transcript for Acme Corp.",
            "q": "2023-Q4",
            "ticker": "ACME",
        },
        {
            "question": "What was the profit?",
            "answer": "2 billion",
            "date": "2024-02-20",
            "transcript": "Q1 earnings call transcript for TechCorp.",
            "q": "2024-Q1",
            "ticker": "TECH",
        },
    ]


@pytest.fixture
def mock_text_splitter():
    """Create a mock text splitter for LangChain."""
    splitter = MagicMock()
    splitter.transform_documents = MagicMock(
        side_effect=lambda docs: docs  # Return docs unchanged
    )
    return splitter


@pytest.fixture
def mock_groq_generator():
    """Create a mock Groq generator."""
    generator = MagicMock()
    generator.generate = MagicMock(return_value="Generated summary")
    return generator
