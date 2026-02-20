"""RAG generation utilities for metadata filtering pipelines.

Provides optional RAG capabilities using OpenAI-compatible APIs (e.g., Groq)
for generating answers from retrieved documents.
"""

import logging
from typing import Any

from haystack import Document
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret


__all__ = ["create_rag_generator", "generate_answer"]

logger = logging.getLogger(__name__)


def create_rag_generator(config: dict[str, Any]) -> OpenAIGenerator | None:
    """Create RAG generator from configuration if enabled.

    Configuration should include a 'rag' section with:
    - enabled: Boolean flag (default: False)
    - model: Model name (e.g., llama-3.3-70b-versatile for Groq)
    - api_key: API key (uses environment variable if not set)
    - api_base_url: Optional API base URL (default: OpenAI endpoint)

    Args:
        config: Configuration dictionary.

    Returns:
        OpenAIGenerator if RAG is enabled, None otherwise.

    Raises:
        ValueError: If RAG is enabled but required config is missing.
    """
    rag_config = config.get("rag", {})

    if not rag_config.get("enabled", False):
        return None

    model = rag_config.get("model")
    if not model:
        raise ValueError("'rag.model' is required when RAG is enabled")

    api_key = rag_config.get("api_key")
    api_base_url = rag_config.get("api_base_url")

    try:
        generator = OpenAIGenerator(
            model=model,
            api_key=Secret.from_token(api_key) if api_key else None,
            api_base_url=api_base_url,
        )
        logger.info("Created RAG generator with model: %s", model)
        return generator
    except Exception as e:
        logger.error("Failed to create RAG generator: %s", e)
        raise


def generate_answer(
    query: str,
    documents: list[Document],
    generator: OpenAIGenerator,
    max_docs: int = 5,
) -> str | None:
    """Generate answer using RAG from retrieved documents.

    Args:
        query: Original user query.
        documents: Retrieved documents to use as context.
        generator: OpenAIGenerator instance.
        max_docs: Maximum documents to include in context (default: 5).

    Returns:
        Generated answer string, or None if generation fails.
    """
    if not documents:
        logger.warning("No documents provided for RAG generation")
        return None

    # Use top-k documents
    top_docs = documents[:max_docs]

    context_parts = []
    for i, doc in enumerate(top_docs, 1):
        content = doc.content or ""
        context_parts.append(f"Document {i}:\n{content}")

    context = "\n\n".join(context_parts)

    prompt = f"""Answer the following question based on the provided context.
If the answer is not found in the context, say so clearly.

Context:
{context}

Question: {query}

Answer:"""

    try:
        result = generator.run(prompt=prompt)
        if result and "replies" in result and result["replies"]:
            answer = result["replies"][0]
            logger.info("Generated answer: %s", answer[:100])
            return answer
        return None
    except Exception as e:
        logger.error("Failed to generate answer: %s", e)
        return None
