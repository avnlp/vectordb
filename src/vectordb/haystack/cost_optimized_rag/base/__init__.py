"""Base modules for cost-optimized RAG.

This package provides foundational components for building cost-efficient
RAG pipelines. Each module addresses specific cost optimization concerns:

    - config: Centralized configuration with environment variable resolution
      for secure credential management without hardcoding
    - metrics: Retrieval quality metrics (MRR, NDCG, Recall@K) to validate
      that cost reductions don't compromise search accuracy

Design Philosophy:
    Each component exposes a trade-off between cost and quality. The configuration
    system allows fine-tuning these trade-offs per use case without code changes.
"""

from vectordb.haystack.cost_optimized_rag.base.config import (
    RAGConfig,
    load_config,
)


__all__ = [
    "RAGConfig",
    "load_config",
]
