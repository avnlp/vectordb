"""Context compression and summarization for RAG.

Reduces token usage and noise by compressing retrieved context before LLM generation.
This module implements three compression strategies:
- Abstractive: LLM-generated summaries that capture key information
- Extractive: Selection of most relevant sentences from original text
- Relevance Filtering: Removal of low-relevance content chunks

The compressor uses Groq API (Llama models) for fast inference, making it
suitable for real-time RAG pipelines where latency matters.

Design Considerations:
    - Temperature=0 for consistent, deterministic compression
    - Max tokens limited to prevent excessive output length
    - Graceful fallback to original context on compression failures
    - Compression ratio logging for performance monitoring

Usage:
    >>> from vectordb.haystack.components import ContextCompressor
    >>> compressor = ContextCompressor(model="llama-3.3-70b-versatile")
    >>> compressed = compressor.compress(context, query, compression_type="abstractive")
"""

import logging
import os
from typing import Any

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret


logger = logging.getLogger(__name__)


class ContextCompressor:
    """Compress and summarize retrieved context.

    Supports:
    - Extractive summarization: Select key sentences
    - Abstractive summarization: LLM-based summaries
    - Relevance filtering: Keep only relevant chunks

    This component reduces token consumption in the generation phase by
    condensing retrieved documents while preserving query-relevant information.
    It's particularly valuable when:
    - Retrieved context exceeds LLM context window limits
    - Many retrieved documents contain redundant information
    - Token costs need to be minimized

    Attributes:
        generator: Haystack OpenAIChatGenerator for LLM-based compression.

    Note:
        All compression methods return the original context on failure,
        ensuring the RAG pipeline never fails due to compression errors.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
    ) -> None:
        """Initialize context compressor.

        Args:
            model: LLM model name (Groq API).
            api_key: Groq API key (or set GROQ_API_KEY env var).
        """
        resolved_api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_api_key:
            msg = "GROQ_API_KEY required. Set it as environment variable."
            raise ValueError(msg)

        try:
            self.generator = OpenAIChatGenerator(
                api_key=Secret.from_token(resolved_api_key),
                model=model,
                api_base_url="https://api.groq.com/openai/v1",
                generation_kwargs={"temperature": 0, "max_tokens": 2048},
            )
            logger.info("Initialized ContextCompressor with model: %s", model)
        except Exception as e:
            logger.error("Failed to initialize ContextCompressor: %s", str(e))
            raise

    def compress_abstractive(
        self,
        context: str,
        query: str,
        max_tokens: int = 2048,
    ) -> str:
        """Abstractive compression using LLM summarization.

        Args:
            context: Retrieved context to compress.
            query: Original query (for relevance).
            max_tokens: Maximum tokens in compressed output.

        Returns:
            Compressed context summary.
        """
        prompt = f"""Summarize the following context to answer this question: "{query}"

Keep only the most relevant information. Be concise.

Context:
{context}

Summary (max {max_tokens} tokens):"""

        try:
            messages = [ChatMessage.from_user(prompt)]
            response = self.generator.run(messages=messages)
            summary = (
                response["replies"][0].text if response.get("replies") else context
            )

            # Calculate and log compression ratio for monitoring
            # Higher ratios indicate more aggressive compression
            compression_ratio = len(context) / (len(summary) + 1)
            logger.info("Abstractive compression: %.2fx", compression_ratio)
            return summary
        except Exception as e:
            # Return original context on failure to ensure pipeline continuity
            logger.error("Abstractive compression failed: %s", str(e))
            return context

    def compress_extractive(
        self,
        context: str,
        query: str,
        num_sentences: int = 5,
    ) -> str:
        """Extractive compression: select key sentences.

        Args:
            context: Retrieved context to compress.
            query: Original query (for relevance).
            num_sentences: Number of sentences to extract.

        Returns:
            Compressed context (selected sentences).
        """
        prompt = f"""Extract the {num_sentences} most relevant sentences from the following context to answer: "{query}"

Return ONLY the selected sentences in order, without numbering."""

        try:
            messages = [ChatMessage.from_user(prompt)]
            response = self.generator.run(messages=messages)
            summary = (
                response["replies"][0].text if response.get("replies") else context
            )
            logger.info("Extractive compression: selected key sentences")
            return summary
        except Exception as e:
            logger.error("Extractive compression failed: %s", str(e))
            return context

    def filter_by_relevance(
        self,
        context: str,
        query: str,
        relevance_threshold: float = 0.5,
    ) -> str:
        """Filter context chunks by relevance to query.

        Args:
            context: Retrieved context to filter.
            query: Original query.
            relevance_threshold: Minimum relevance score (0-1).

        Returns:
            Filtered context containing only relevant chunks.
        """
        prompt = f"""Given this query: "{query}"

Evaluate each paragraph in the context below for relevance (0-100 scale).
Keep only paragraphs with relevance > {int(relevance_threshold * 100)}.

Context:
{context}

Output only the relevant paragraphs:"""

        try:
            messages = [ChatMessage.from_user(prompt)]
            response = self.generator.run(messages=messages)
            filtered = (
                response["replies"][0].text if response.get("replies") else context
            )
            logger.info("Relevance filtering: kept relevant chunks")
            return filtered
        except Exception as e:
            logger.error("Relevance filtering failed: %s", str(e))
            return context

    def compress(
        self,
        context: str,
        query: str,
        compression_type: str = "abstractive",
        **kwargs: Any,
    ) -> str:
        """Compress context using specified technique.

        Args:
            context: Context to compress.
            query: Original query.
            compression_type: "abstractive", "extractive", or "relevance_filter".
            **kwargs: Additional parameters (max_tokens, num_sentences, threshold).

        Returns:
            Compressed context.

        Raises:
            ValueError: If compression_type is unsupported.
        """
        if compression_type == "abstractive":
            max_tokens = kwargs.get("max_tokens", 2048)
            return self.compress_abstractive(context, query, max_tokens)

        if compression_type == "extractive":
            num_sentences = kwargs.get("num_sentences", 5)
            return self.compress_extractive(context, query, num_sentences)

        if compression_type == "relevance_filter":
            threshold = kwargs.get("relevance_threshold", 0.5)
            return self.filter_by_relevance(context, query, threshold)

        raise ValueError(
            f"Unsupported compression type: {compression_type}. "
            "Must be 'abstractive', 'extractive', or 'relevance_filter'"
        )

    def validate_config(
        self,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate compression configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            Validated configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        if not isinstance(config, dict):
            raise ValueError(f"Config must be dict, got {type(config)}")

        compression_type = config.get("type", "abstractive").lower()
        if compression_type not in [
            "abstractive",
            "extractive",
            "relevance_filter",
        ]:
            raise ValueError(
                f"Unsupported compression type: {compression_type}. "
                "Must be 'abstractive', 'extractive', or 'relevance_filter'"
            )

        return config
