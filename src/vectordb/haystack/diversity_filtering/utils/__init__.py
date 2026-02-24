"""Utility modules for diversity filtering pipelines.

Supporting utilities for configuring and executing diversity-aware retrieval
pipelines across all supported vector databases.

Modules:
- config_loader: Pydantic-based configuration validation with support for
  environment variable substitution (${VAR_NAME} syntax). Validates configs
  for Qdrant, Pinecone, Weaviate, Chroma, and Milvus backends.
- prompts: Dataset-specific RAG prompt templates (TriviaQA, ARC, PopQA,
  FactScore, Earnings Calls) with document formatting utilities.

Configuration includes diversity filtering parameters such as algorithm selection
(MMR vs clustering), top_k settings, and lambda trade-off values.
"""

from vectordb.haystack.diversity_filtering.utils.config_loader import (
    ConfigLoader,
    DiversityFilteringConfig,
)
from vectordb.haystack.diversity_filtering.utils.prompts import (
    format_documents,
    get_prompt_template,
)


__all__ = [
    "ConfigLoader",
    "DiversityFilteringConfig",
    "get_prompt_template",
    "format_documents",
]
