"""RAG (Retrieval-Augmented Generation) utilities for Haystack pipelines.

This module provides helper methods for LLM-based answer generation using
retrieved documents as context. It uses Haystack's OpenAIGenerator configured
for Groq's fast inference API.

Key Features:
    - Groq Integration: Uses Groq's OpenAI-compatible API for fast inference
    - Prompt Formatting: Constructs prompts from query and retrieved documents
    - Configurable: Temperature, max tokens, and custom prompt templates
    - Graceful Handling: Returns empty string on generation failures

LLM Configuration:
    The RAGHelper reads from a standardized config structure:

    rag:
      enabled: true
      model: "llama-3.3-70b-versatile"
      api_key: "${GROQ_API_KEY}"
      api_base_url: "https://api.groq.com/openai/v1"
      temperature: 0.7
      max_tokens: 2048

Prompt Template:
    Default template includes context documents and query. Custom templates
    can use {context} and {query} placeholders.

Usage:
    >>> from vectordb.haystack.utils import RAGHelper
    >>> config = {"rag": {"enabled": True, "model": "llama-3.3-70b-versatile"}}
    >>> generator = RAGHelper.create_generator(config)
    >>> answer = RAGHelper.generate(generator, "What is ML?", retrieved_docs)
"""

import os
from typing import Any

from haystack import Document
from haystack.components.generators import OpenAIGenerator


class RAGHelper:
    """Helper class for RAG (Retrieval-Augmented Generation) operations.

    Uses Haystack's OpenAIGenerator with Groq endpoint.

    Example config:
        rag:
          enabled: true
          model: "llama-3.3-70b-versatile"
          api_key: "${GROQ_API_KEY}"
          api_base_url: "https://api.groq.com/openai/v1"
          temperature: 0.7
          max_tokens: 2048
    """

    DEFAULT_PROMPT_TEMPLATE = """Answer the following question based on the provided context.
If the answer cannot be found in the context, say so.

Context:
{context}

Question: {query}

Answer:"""

    @classmethod
    def create_generator(cls, config: dict[str, Any]) -> OpenAIGenerator | None:
        """Create a Groq-compatible RAG generator.

        Args:
            config: Configuration with 'rag' section.

        Returns:
            Configured OpenAIGenerator or None if RAG is disabled.
        """
        rag_config = config.get("rag", {})

        if not rag_config.get("enabled", False):
            return None

        model = rag_config["model"]
        api_key = rag_config.get("api_key") or os.environ.get("GROQ_API_KEY")
        api_base = rag_config.get("api_base_url", "https://api.groq.com/openai/v1")

        if not api_key:
            msg = (
                "RAG enabled but no API key provided. "
                "Set rag.api_key in config or GROQ_API_KEY environment variable."
            )
            raise ValueError(msg)

        return OpenAIGenerator(
            api_key=api_key,
            model=model,
            api_base_url=api_base,
            generation_kwargs={
                "temperature": rag_config.get("temperature", 0.7),
                "max_tokens": rag_config.get("max_tokens", 2048),
            },
        )

    @classmethod
    def format_prompt(
        cls,
        query: str,
        documents: list[Document],
        template: str | None = None,
    ) -> str:
        """Format a RAG prompt with query and retrieved context.

        Args:
            query: The user's question.
            documents: Retrieved documents for context.
            template: Optional custom template. Use {context} and {query} placeholders.

        Returns:
            Formatted prompt string.
        """
        context = "\n\n".join(
            f"Document {i + 1}:\n{doc.content}" for i, doc in enumerate(documents)
        )

        if template:
            return template.format(context=context, query=query)

        return cls.DEFAULT_PROMPT_TEMPLATE.format(context=context, query=query)

    @classmethod
    def generate(
        cls,
        generator: OpenAIGenerator,
        query: str,
        documents: list[Document],
        template: str | None = None,
    ) -> str:
        """Generate a RAG response.

        Args:
            generator: Initialized OpenAIGenerator.
            query: The user's question.
            documents: Retrieved documents for context.
            template: Optional custom prompt template.

        Returns:
            Generated answer string.
        """
        prompt = cls.format_prompt(query, documents, template)
        result = generator.run(prompt=prompt)
        replies = result.get("replies", [])
        return replies[0] if replies else ""
