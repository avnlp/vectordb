"""Query enhancement and expansion strategies.

Implements:
- Multi-Query: Generate N variations of the query
- HyDE (Hypothetical Document Embeddings): Generate hypothetical relevant documents
- Step-Back: Generate broader, abstracted version of the query

Query enhancement improves retrieval coverage by expanding a single query into
multiple related queries. This is particularly effective when:
- The original query is ambiguous or underspecified
- Important information might be expressed using different terminology
- The user wants comprehensive coverage of a topic

HyDE (Hypothetical Document Embeddings):
    Instead of embedding the query directly, generate hypothetical documents
    that would answer the query, then embed those. This bridges the lexical
    gap between queries and documents since hypothetical docs are more likely
    to match the vocabulary of actual documents.

Step-Back Prompting:
    Generate a more abstract version of the query to retrieve broader context,
    then use the specific query for final answer generation. This helps when
    the specific query might miss relevant background information.

Architecture:
    Uses Groq API (Llama models) for fast LLM inference. Temperature is set
    higher (0.7) than routing components to encourage query diversity.

Usage:
    >>> from vectordb.haystack.components import QueryEnhancer
    >>> enhancer = QueryEnhancer(model="llama-3.3-70b-versatile")
    >>> queries = enhancer.enhance_query("neural networks", "multi_query")
    >>> # queries = ["neural networks", "deep learning basics", "ANN architecture"]
"""

import logging
import os
from typing import Any

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret


logger = logging.getLogger(__name__)


class QueryEnhancer:
    """Enhance and expand queries using LLM-based techniques.

    Supports:
    - Multi-Query: Generate diverse query variations
    - HyDE: Generate hypothetical documents
    - Step-Back: Generate abstracted/broader query

    The QueryEnhancer improves retrieval recall by generating multiple
    query representations. Each technique serves a different purpose:

    - Multi-query: Captures different aspects of the information need
    - HyDE: Bridges lexical gap between queries and documents
    - Step-back: Retrieves broader context for better grounding

    Attributes:
        generator: Haystack OpenAIChatGenerator for LLM interactions.

    Note:
        Temperature is set to 0.7 (higher than routing) to encourage
        diversity in generated queries and hypothetical documents.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
    ) -> None:
        """Initialize query enhancer.

        Args:
            model: LLM model name (Groq API).
            api_key: Groq API key (or set GROQ_API_KEY env var).
        """
        resolved_api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_api_key:
            msg = "GROQ_API_KEY required. Set it as environment variable."
            raise ValueError(msg)

        self.generator = OpenAIChatGenerator(
            api_key=Secret.from_token(resolved_api_key),
            model=model,
            api_base_url="https://api.groq.com/openai/v1",
            generation_kwargs={"temperature": 0.7, "max_tokens": 1024},
        )
        logger.info("Initialized QueryEnhancer with model: %s", model)

    def generate_multi_queries(
        self,
        query: str,
        num_queries: int = 3,
    ) -> list[str]:
        """Generate multiple query variations.

        Args:
            query: Original query.
            num_queries: Number of variations to generate.

        Returns:
            List of query variations (including original).
        """
        if num_queries < 1:
            raise ValueError(f"num_queries must be >= 1, got {num_queries}")

        prompt = f"""Generate {num_queries} different queries that would help retrieve relevant information for: "{query}"

Return ONLY the queries, one per line, without numbering or extra text."""

        try:
            messages = [ChatMessage.from_user(prompt)]
            response = self.generator.run(messages=messages)
            content = response["replies"][0].text if response.get("replies") else ""

            # Parse generated queries, filtering empty lines
            queries = [q.strip() for q in content.split("\n") if q.strip()]

            # Always include original query first, then add generated variations
            # Limit to requested number to avoid excessive retrieval calls
            queries = [query] + queries[: num_queries - 1]
            logger.info(
                "Generated %d multi-queries from: %s",
                len(queries),
                query,
            )
            return queries
        except Exception as e:
            # Return original query on failure to maintain pipeline flow
            logger.error("Multi-query generation failed: %s", str(e))
            return [query]

    def generate_hypothetical_documents(
        self,
        query: str,
        num_docs: int = 3,
    ) -> list[str]:
        """Generate hypothetical relevant documents (HyDE).

        Args:
            query: Query to generate hypothetical documents for.
            num_docs: Number of hypothetical documents.

        Returns:
            List of hypothetical document texts.
        """
        if num_docs < 1:
            raise ValueError(f"num_docs must be >= 1, got {num_docs}")

        prompt = f"""Generate {num_docs} hypothetical document excerpts that would directly answer this question: "{query}"

Return ONLY the document excerpts, separated by "---", without numbering or extra text."""

        try:
            messages = [ChatMessage.from_user(prompt)]
            response = self.generator.run(messages=messages)
            content = response["replies"][0].text if response.get("replies") else ""
            docs = [d.strip() for d in content.split("---") if d.strip()]
            docs = docs[:num_docs]
            logger.info(
                "Generated %d hypothetical documents for: %s",
                len(docs),
                query,
            )
            return docs
        except Exception as e:
            logger.error("HyDE generation failed: %s", str(e))
            return [query]

    def generate_step_back_query(self, query: str) -> str:
        """Generate a broader, abstracted version of the query (Step-Back).

        Args:
            query: Original query.

        Returns:
            Abstracted/step-back query.
        """
        prompt = f"""Given the question: "{query}"

Generate a more abstract, higher-level question that captures the core concept without specific details.

Return ONLY the abstracted question, without explanation."""

        try:
            messages = [ChatMessage.from_user(prompt)]
            response = self.generator.run(messages=messages)
            step_back = response["replies"][0].text if response.get("replies") else ""
            logger.info("Generated step-back query for: %s → %s", query, step_back)
            return step_back
        except Exception as e:
            logger.error("Step-back generation failed: %s", str(e))
            return query

    def enhance_query(
        self,
        query: str,
        enhancement_type: str = "multi_query",
        **kwargs: Any,
    ) -> list[str]:
        """Enhance query using specified technique.

        Args:
            query: Original query.
            enhancement_type: "multi_query", "hyde", or "step_back".
            **kwargs: Additional parameters for the enhancement technique.

        Returns:
            List of enhanced/expanded queries.

        Raises:
            ValueError: If enhancement_type is unsupported.
        """
        if enhancement_type == "multi_query":
            num_queries = kwargs.get("num_queries", 3)
            return self.generate_multi_queries(query, num_queries)

        if enhancement_type == "hyde":
            num_docs = kwargs.get("num_docs", 3)
            return self.generate_hypothetical_documents(query, num_docs)

        if enhancement_type == "step_back":
            step_back = self.generate_step_back_query(query)
            return [query, step_back]

        raise ValueError(
            f"Unsupported enhancement type: {enhancement_type}. "
            "Must be 'multi_query', 'hyde', or 'step_back'"
        )
