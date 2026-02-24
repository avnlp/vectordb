"""LLM generator utilities for query enhancement.

Uses Haystack's OpenAIChatGenerator with Groq API for query enhancement.
"""

import os
from typing import Any

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret


def create_groq_generator(config: dict[str, Any]) -> OpenAIChatGenerator:
    """Create a Groq chat generator using OpenAIChatGenerator.

    Args:
        config: Configuration dictionary with 'query_enhancement.llm' section.

    Returns:
        OpenAIChatGenerator configured for Groq API.

    Raises:
        ValueError: If GROQ_API_KEY is not available.
    """
    llm_config = config.get("query_enhancement", {}).get("llm", {})
    model = llm_config.get("model", "llama-3.3-70b-versatile")
    api_key = llm_config.get("api_key") or os.environ.get("GROQ_API_KEY")

    if not api_key:
        msg = "GROQ_API_KEY required. Set it as environment variable."
        raise ValueError(msg)

    generation_kwargs = llm_config.get("kwargs", {})
    if "temperature" not in generation_kwargs:
        generation_kwargs["temperature"] = 0.7
    if "max_tokens" not in generation_kwargs:
        generation_kwargs["max_tokens"] = 1024

    return OpenAIChatGenerator(
        api_key=Secret.from_token(api_key),
        model=model,
        api_base_url="https://api.groq.com/openai/v1",
        generation_kwargs=generation_kwargs,
    )


# Prompt templates for query enhancement
MULTI_QUERY_PROMPT = """Generate {num_queries} different search queries that would help find information to answer this question: "{query}"

Return ONLY the queries, one per line, without numbering or extra text."""

HYDE_PROMPT = """Generate {num_docs} hypothetical document excerpts that would directly answer this question: "{query}"

Return ONLY the document excerpts, separated by "---", without numbering or extra text."""

STEP_BACK_PROMPT = """Given the question: "{query}"

Generate a more abstract, higher-level question that captures the core concept without specific details.

Return ONLY the abstracted question, without explanation."""
