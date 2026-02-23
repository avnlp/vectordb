"""Shared fixtures for cost-optimized RAG pipeline tests.

This module provides pytest fixtures for testing cost-optimized RAG
implementations in Haystack. Cost optimization reduces API costs through
query routing, tiered retrieval, response caching, and model selection.

Fixtures:
    base_config_dict: Complete configuration dictionary with collection,
        embeddings, dataloader, indexing, search, reranker, generator,
        and logging settings for cost-optimized pipeline testing.
    sample_documents: Haystack Documents with geographic content and
        metadata for testing cost optimization strategies.

Note:
    Cost-optimized RAG tests validate that query routing and tiered
    retrieval strategies maintain answer quality while reducing
    LLM token usage and API call frequency.
"""

from typing import Any

import pytest
from haystack import Document


@pytest.fixture
def base_config_dict() -> dict[str, Any]:
    """Create base configuration for cost-optimized RAG testing.

    Returns:
        Configuration dictionary with collection, embeddings, dataloader,
        indexing, search, reranker, generator, and logging settings.
    """
    return {
        "collection": {"name": "test_collection", "description": "Test"},
        "embeddings": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 32,
        },
        "dataloader": {
            "type": "triviaqa",
            "dataset_name": "trivia_qa",
            "config": "rc",
            "split": "test",
            "limit": 10,
        },
        "indexing": {
            "vector_config": {"size": 384, "distance": "Cosine"},
        },
        "search": {"top_k": 5, "reranking_enabled": False},
        "reranker": {
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "top_k": 5,
        },
        "generator": {
            "enabled": False,
            "model": "llama-3.3-70b-versatile",
            "api_key": "",
            "api_base_url": "https://api.groq.com/openai/v1",
            "temperature": 0.7,
            "max_tokens": 2048,
        },
        "logging": {"name": "test", "level": "DEBUG"},
    }


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for cost-optimized RAG testing.

    Returns:
        List of Haystack Documents with geographic content for
        testing retrieval and answer generation pipelines.
    """
    return [
        Document(
            id="doc1",
            content="Paris is the capital of France.",
            meta={"source": "geography"},
        ),
        Document(
            id="doc2",
            content="Berlin is the capital of Germany.",
            meta={"source": "geography"},
        ),
        Document(
            id="doc3",
            content="Rome is the capital of Italy.",
            meta={"source": "geography"},
        ),
    ]
