"""Shared fixtures for dataloader tests.

This module provides pytest fixtures specifically designed for testing
dataloader functionality. These fixtures create mock data structures
that simulate the expected output format from various dataset loaders.

Fixtures:
    mock_hf_dataset: Mock HuggingFace dataset iterator.
    arc_sample_rows: Sample ARC dataset rows with questions, choices, and answers.
    triviaqa_sample_rows: Sample TriviaQA rows with questions, answers.
    popqa_sample_rows: Sample PopQA rows with entity-relation questions and answers.
    factscore_sample_rows: Sample FactScore rows with topics, facts, and decomposition.
    earnings_calls_sample_rows: Sample earnings calls Q&A pairs with transcripts.
    mock_groq_generator: Mock LLM generator for testing with summarization.
    dataloader_config: Base dataloader configuration dictionary.
    triviaqa_config_with_generator: TriviaQA config with LLM generator settings.

These fixtures are used by test modules to verify dataloader behavior
without requiring actual dataset downloads or API calls.
"""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_hf_dataset():
    """Create a mock HuggingFace dataset for testing.

    This fixture provides a MagicMock object that simulates a HuggingFace
    dataset's iterator interface. The mock is configured to return an
    empty iterator by default.

    Returns:
        MagicMock: A mock dataset object with __iter__ method configured.

    Example:
        with patch("vectordb.dataloaders.arc.hf_load_dataset") as mock_load:
            mock_load.return_value = mock_hf_dataset
    """


@pytest.fixture
def arc_sample_rows():
    """Create sample ARC (AI2 Reasoning Challenge) rows for testing.

    The ARC dataset contains multiple-choice questions with four answer choices.
    This fixture provides realistic sample data matching the expected HuggingFace
    dataset format.

    Returns:
        list[dict]: List of dictionaries containing:
            - id: Unique question identifier
            - question: The question text
            - choices: Dict with 'label' (A-D) and 'text' (answer options)
            - answerKey: The correct answer label

    Example:
        >>> rows = arc_sample_rows
        >>> rows[0]["question"]
        'What is the capital of France?'
    """
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
def triviaqa_sample_rows():
    """Create sample TriviaQA rows for testing.

    TriviaQA is a question answering dataset that includes the question,
    answer, and web search results. This fixture simulates the format returned
    by the HuggingFace datasets library.

    Returns:
        list[dict]: List of dictionaries containing:
            - question: The trivia question
            - answer: The correct answer
            - search_results: Dict with rank, title, search_context, description

    Note:
        TriviaQA differs from other datasets by including search context
        which is useful for open-book QA evaluation.
    """
    return [
        {
            "question": "What is the capital of France?",
            "answer": "Paris",
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
def popqa_sample_rows():
    """Create sample PopQA (Popular Question Answering) rows for testing.

    PopQA contains entity-centric questions with popularity scores. Each question
    is about a specific entity and relation, making it ideal for testing
    knowledge-intensive retrieval tasks.

    Returns:
        list[dict]: List of dictionaries containing:
            - question: Entity-relation question
            - possible_answers: List of valid answer strings
            - subj: The subject entity
            - prop: The relation/property being asked about
            - obj: The object/value answer
            - content: Contextual information about the entity

    Note:
        PopQA's entity-relation structure makes it valuable for testing
        attribute-based retrieval and entity linking.
    """
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
def factscore_sample_rows():
    """Create sample FactScore rows for testing.

    FactScore evaluates faithfulness of generated text against source documents.
    This fixture provides sample data with decomposed facts for testing
    fact-checking and faithfulness evaluation pipelines.

    Returns:
        list[dict]: List of dictionaries containing:
            - id: Unique identifier
            - topic: The entity/topic being evaluated
            - entity: Entity name
            - wikipedia_text: Source Wikipedia article text
            - one_fact_prompt: Prompt for single fact extraction
            - factscore_prompt: Prompt for fact evaluation
            - facts: List of atomic facts to verify
            - decomposed_facts: List of dicts with sent_id and fact text

    Note:
        FactScore is particularly useful for testing RAG systems where
        factual accuracy is critical.
    """
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
        },
    ]


@pytest.fixture
def earnings_calls_sample_rows():
    """Create sample earnings calls Q&A rows for testing.

    Earnings calls datasets contain financial Q&A pairs extracted from
    quarterly earnings call transcripts. This fixture provides sample data
    for testing financial domain RAG applications.

    Returns:
        list[dict]: List of dictionaries containing:
            - question: Financial question about the earnings call
            - answer: Answer extracted from transcript
            - date: Date of the earnings call
            - transcript: Full or partial transcript text
            - q: Quarter identifier (e.g., "2023-Q4")
            - ticker: Stock ticker symbol
            - company: Company name

    Note:
        Earnings calls data is domain-specific and requires understanding
        of financial terminology and quarter reporting cycles.
    """
    return [
        {
            "question": "What was the revenue?",
            "answer": "10 billion",
            "date": "2024-01-15",
            "transcript": "Q4 earnings call transcript for Acme Corp.",
            "q": "2023-Q4",
            "ticker": "ACME",
            "company": "Acme Corp",
        },
        {
            "question": "What was the profit?",
            "answer": "2 billion",
            "date": "2024-02-20",
            "transcript": "Q1 earnings call transcript for TechCorp.",
            "q": "2024-Q1",
            "ticker": "TECH",
            "company": "TechCorp",
        },
    ]


@pytest.fixture
def mock_groq_generator():
    """Create a mock Groq LLM generator for testing.

    This fixture provides a MagicMock that simulates a Groq API generator
    for testing dataloader components without making actual API calls.

    Returns:
        MagicMock: A mock generator with configured generate method.

    Note:
        The mock's generate method returns "Generated summary" by default,
        which is suitable for testing summarization and extraction flows.
    """
    generator = MagicMock()
    generator.generate = MagicMock(return_value="Generated summary")
    return generator


@pytest.fixture
def dataloader_config():
    """Create a base dataloader configuration for testing.

    This fixture provides a minimal configuration dictionary that can be
    used as a template for creating dataloaders with various settings.

    Returns:
        dict: Configuration dictionary with:
            - dataloader.type: Dataset type identifier
            - dataloader.dataset_name: HuggingFace dataset name
            - dataloader.split: Dataset split to load (train/test/validation)

    Note:
        This is a base configuration meant to be extended or modified
        for specific test scenarios.
    """
    return {
        "dataloader": {
            "type": "arc",
            "dataset_name": "ai2_arc",
            "split": "test",
        }
    }


@pytest.fixture
def triviaqa_config_with_generator():
    """Create TriviaQA configuration with LLM generator settings.

    This fixture provides a complete configuration for testing TriviaQA
    dataloaders that require an LLM generator for answer generation.

    Returns:
        dict: Configuration dictionary with:
            - dataloader.type: "triviaqa"
            - dataloader.dataset_name: HuggingFace dataset name
            - dataloader.split: Dataset split
            - generator.model: Groq model identifier
            - generator.api_key: API key for authentication

    Note:
        TriviaQA is often used with LLM generators for generating
        answers from retrieved context, making this config useful
        for end-to-end RAG pipeline testing.
    """
    return {
        "dataloader": {
            "type": "triviaqa",
            "dataset_name": "trivia_qa",
            "split": "test",
        },
        "generator": {
            "model": "llama-3.3-70b-versatile",
            "api_key": "test-key",
        },
    }
