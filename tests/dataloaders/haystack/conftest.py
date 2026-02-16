"""Shared fixtures for Haystack dataloader tests.

This module provides pytest fixtures specifically designed for testing
Haystack-based dataloader integrations. These fixtures create mock data
structures and components that simulate the Haystack framework's expected
interfaces and data formats.

Fixtures:
    haystack_arc_sample_rows: ARC rows with Haystack-specific context structure.
    haystack_triviaqa_sample_rows: TriviaQA rows with ctxs for Haystack Documents.
    haystack_popqa_sample_rows: PopQA rows adapted for Haystack pipeline testing.
    haystack_factscore_sample_rows: FactScore rows with entity and context info.
    haystack_earnings_calls_sample_rows: Earnings calls data for Haystack tests.
    mock_openai_chat_generator: Mock Haystack OpenAIChatGenerator component.
    mock_recursive_document_splitter: Mock Haystack document splitter component.
    mock_haystack_document: Sample Haystack Document object.

Note:
    Haystack fixtures include 'ctxs' (contexts) field which maps to
    Document objects when processed through Haystack pipelines.
"""

from unittest.mock import MagicMock

import pytest
from haystack import Document
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.preprocessors import RecursiveDocumentSplitter


@pytest.fixture
def haystack_arc_sample_rows():
    """Create sample ARC rows for Haystack tests."""
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
def haystack_triviaqa_sample_rows():
    """Create sample TriviaQA rows for Haystack tests."""
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
def haystack_popqa_sample_rows():
    """Create sample PopQA rows for Haystack tests."""
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
def haystack_factscore_sample_rows():
    """Create sample FactScore rows for Haystack tests."""
    return [
        {
            "question": "Tell me about Albert Einstein",
            "answers": ["Einstein was a physicist"],
            "entity": "Albert Einstein",
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
def haystack_earnings_calls_sample_rows():
    """Create sample earnings calls rows for Haystack tests."""
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
def mock_openai_chat_generator():
    """Create a mock OpenAIChatGenerator for Haystack tests."""
    generator = MagicMock(spec=OpenAIChatGenerator)
    mock_reply = MagicMock()
    mock_reply.text = "Mocked summary response"
    generator.run = MagicMock(return_value={"replies": [mock_reply]})
    return generator


@pytest.fixture
def mock_recursive_document_splitter():
    """Create a mock RecursiveDocumentSplitter for Haystack tests."""
    splitter = MagicMock(spec=RecursiveDocumentSplitter)
    # Mock the warm_up method to avoid NLTK import issues
    splitter.warm_up = MagicMock(return_value=None)
    splitter.run = MagicMock(side_effect=lambda docs: {"documents": docs})
    return splitter


@pytest.fixture
def mock_haystack_document():
    """Create a mock Haystack Document."""
    return Document(content="Test content", meta={"source": "test"})
