"""RAG generator creation using Groq via Haystack's OpenAI-compatible API."""

from __future__ import annotations

import os
from typing import Any

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator


DEFAULT_RAG_PROMPT = """Based on the retrieved documents, answer the question.

Context:
{% for doc in documents %}
- {{ doc.content }}
{% endfor %}

Question: {{ query }}

Answer:"""


def create_rag_generator(config: dict[str, Any]) -> OpenAIGenerator:
    """Create Groq-compatible RAG generator.

    Uses Haystack's OpenAIGenerator with Groq's OpenAI-compatible API.

    Args:
        config: Configuration with 'generator' section.

    Returns:
        Configured OpenAIGenerator for Groq.
    """
    generator_config = config.get("generator", {})

    # Groq uses OpenAI-compatible API
    api_key = generator_config.get("api_key") or os.environ.get("GROQ_API_KEY")
    model = generator_config.get("model", "llama-3.3-70b-versatile")
    api_base_url = generator_config.get(
        "api_base_url", "https://api.groq.com/openai/v1"
    )

    generation_kwargs = generator_config.get("kwargs", {})
    if "temperature" not in generation_kwargs:
        generation_kwargs["temperature"] = 0.5
    if "max_tokens" not in generation_kwargs:
        generation_kwargs["max_tokens"] = 2048

    return OpenAIGenerator(
        api_key=api_key,
        model=model,
        api_base_url=api_base_url,
        generation_kwargs=generation_kwargs,
    )


def create_rag_pipeline(config: dict[str, Any]) -> Pipeline:
    """Create RAG pipeline with prompt builder and generator.

    Args:
        config: Configuration with 'rag' and 'generator' sections.

    Returns:
        Haystack Pipeline for RAG.
    """
    rag_config = config.get("rag", {})
    prompt_template = rag_config.get("prompt_template", DEFAULT_RAG_PROMPT)

    pipeline = Pipeline()
    pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
    pipeline.add_component("generator", create_rag_generator(config))
    pipeline.connect("prompt_builder.prompt", "generator.prompt")

    return pipeline
