"""Shared utilities for compression pipelines.

Provides factory patterns for creating compressors, token counting utilities,
and result formatting. Supports both reranking (embedding-based) and LLM-based
compression strategies.

Compression Algorithms:
    1. Reranking: Uses cross-encoders or API services (Cohere, Voyage, BGE) to
       compute relevance scores for query-document pairs. Higher scores indicate
       greater relevance. Documents are ranked by score and top-k selected.
    2. LLM Extraction: Uses generative models (GPT-4o-mini via OpenAI-compatible
       APIs) to extract only relevant text passages. Reduces tokens by removing
       irrelevant content within documents.

Token Estimation:
    Approximates tokens using a 4:1 character-to-token ratio for English text.
    Tracks compression ratios and token savings for cost optimization.
"""

from dataclasses import dataclass
from typing import Any

from haystack import Document

from vectordb.haystack.components.rerankers import UnifiedReranker


@dataclass
class RankerResult:
    """Result of ranking/compression for a single document.

    Attributes:
        document: The Haystack Document object.
        score: Relevance score (0-1 for rerankers, variable for others).
        rank: Rank position in final results.
        token_count: Estimated token count in document.
    """

    document: Document
    score: float
    rank: int | None = None
    token_count: int = 0


class CompressorFactory:
    """Factory for instantiating compressor instances from configuration."""

    @staticmethod
    def create_compressor(config: dict[str, Any]) -> Any:
        """Create a compressor instance (Reranker or LLM Extractor).

        Args:
            config: Configuration dictionary.

        Returns:
            Initialized compressor instance.
        """
        compression_config = config.get("compression", {})
        # Support legacy/flat config where 'reranker' or 'llm_compression'
        # is at top level
        if not compression_config:
            if "reranker" in config:
                return CompressorFactory.create_reranker(config)
            if "llm_compression" in config:
                return CompressorFactory.create_llm_extractor(config)
            # Default to reranking if nothing specified but likely to fail if empty
            compression_config = {"type": "reranking"}

        compression_type = compression_config.get("type", "reranking")

        if compression_type == "reranking":
            return CompressorFactory.create_reranker(config)
        if compression_type == "llm_extraction":
            return CompressorFactory.create_llm_extractor(config)
        raise ValueError(f"Unsupported compression type: {compression_type}")

    @staticmethod
    def create_reranker(config: dict[str, Any]) -> Any:
        """Create a reranker instance from configuration.

        Instantiates the appropriate reranker (Cohere, Voyage, BGE) based on
        config settings. Delegates to UnifiedReranker for supported types.

        Args:
            config: Configuration dictionary containing reranker settings.

        Returns:
            An initialized reranker instance (Haystack Ranker).

        Raises:
            ValueError: If reranker type is not supported.
        """
        # Handle nested config (compression.reranker) or flat (reranker)
        compression_config = config.get("compression", {})
        reranker_config = compression_config.get("reranker")
        if not reranker_config:
            reranker_config = config.get("reranker", {})

        reranker_type = reranker_config.get("type", "").lower()

        # CROSS-ENCODER RERANKING ALGORITHM
        # Architecture: Joint encoding of query+document pairs
        # How it works:
        #   1. Concatenate query and document with [SEP] token
        #   2. Pass through transformer encoder (BERT-like)
        #   3. Output layer predicts relevance score (0-1)
        # Benefits: Captures query-document interactions directly
        # Trade-offs: Slower than bi-encoders (O(n) forward passes for n docs)
        if reranker_type in [
            "cross_encoder",
            "cross_encoder_light",
            "cross_encoder_qwen",
        ]:
            model = reranker_config.get(
                "model", UnifiedReranker.get_default_model(reranker_type)
            )
            top_k = reranker_config.get("top_k", 5)
            return UnifiedReranker.create_cross_encoder_reranker(
                model=model, top_k=top_k
            )

        # COHERE API RERANKING ALGORITHM
        # Architecture: Cloud-based neural reranking service
        # How it works:
        #   1. Send query + batch of documents to Cohere API
        #   2. Cohere's model computes relevance scores server-side
        #   3. Returns ranked list with relevance scores (0-1)
        # Benefits: No local GPU needed; constantly updated models
        # Trade-offs: API latency, rate limits, cost per request
        if reranker_type == "cohere":
            api_key = reranker_config.get("api_key")
            # API key resolution: config loader handles env vars; if missing,
            # UnifiedReranker will fall back to COHERE_API_KEY env var.
            model = reranker_config.get("model", "rerank-english-v3.0")
            # Note: 'top_n' in some configs, 'top_k' in others
            top_k = reranker_config.get("top_n", 5)
            if "top_k" in reranker_config:
                top_k = reranker_config["top_k"]

            return UnifiedReranker.create_cohere_reranker(
                api_key=api_key, model=model, top_k=top_k
            )

        # VOYAGE AI RERANKING ALGORITHM
        # Architecture: Specialized embedding-based reranking API
        # How it works:
        #   1. Voyage embeds query and documents into semantic space
        #   2. Computes similarity scores between query and each document
        #   3. Returns ranked results with relevance scores
        # Benefits: Optimized for long-context documents
        # Trade-offs: Requires voyage-embedders-haystack package, API costs
        if reranker_type == "voyage":
            try:
                from haystack.utils import Secret
                from haystack_integrations.components.rankers.voyage import (
                    VoyageRanker,
                )

                api_key = reranker_config.get("api_key")
                model = reranker_config.get("model", "rerank-2")

                secret = (
                    Secret.from_token(api_key)
                    if api_key
                    else Secret.from_env_var("VOYAGE_API_KEY")
                )

                return VoyageRanker(model=model, api_key=secret)
            except ImportError as e:
                raise ValueError(
                    "Voyage reranker requires voyage-embedders-haystack: "
                    "pip install voyage-embedders-haystack"
                ) from e

        # BGE (BAAI General Embedding) RERANKING ALGORITHM
        # Architecture: Cross-encoder with multilingual support
        # How it works:
        #   1. BGE models encode query-document pairs jointly
        #   2. Classification head outputs relevance probability
        #   3. Supports 100+ languages with strong zero-shot performance
        # Benefits: Open-source, multilingual, strong benchmark results
        # Trade-offs: Requires local GPU for speed; memory intensive
        if reranker_type == "bge":
            # BGE is often treated as a cross-encoder in UnifiedReranker
            # But if specified explicitly as 'bge', we can map it
            model_name = reranker_config.get("model", "BAAI/bge-reranker-v2-m3")
            top_k = reranker_config.get("top_n", 5)
            return UnifiedReranker.create_cross_encoder_reranker(
                model=model_name, top_k=top_k
            )

        raise ValueError(
            f"Unsupported reranker type: {reranker_type}. "
            f"Supported: cross_encoder, cohere, voyage, bge"
        )

    @staticmethod
    def create_llm_extractor(config: dict[str, Any]) -> Any:
        """Create an LLM-based extractor from configuration.

        Uses Haystack's OpenAIGenerator (compatible with Groq/custom endpoints).

        Args:
            config: Configuration dictionary containing LLM settings.

        Returns:
            An initialized LLM generator instance.

        Raises:
            ValueError: If LLM model type is not supported.
        """
        try:
            from haystack.components.generators import OpenAIGenerator
            from haystack.utils import Secret
        except ImportError as e:
            raise ValueError(
                "LLM extraction requires openai integration: pip install openai"
            ) from e

        # LLM EXTRACTION COMPRESSION ALGORITHM
        # Architecture: Generative LLM with extraction prompts
        # How it works:
        #   1. Construct prompt with query + document content
        #   2. LLM identifies relevant passages for the query
        #   3. Returns extracted text (subset of original document)
        # Benefits: Context-aware extraction; removes irrelevant sections
        # Trade-offs: Higher latency than reranking; LLM API costs
        #
        # Prompt Template (simplified):
        #   "Given query: {query}
        #    Extract only the relevant information from:
        #    {document_content}
        #    Relevant excerpt:"
        llm_config = config.get("compression", {}).get("llm")
        if not llm_config:
            llm_config = config.get("llm_compression", {})

        if not llm_config:
            raise ValueError("No LLM compression config found")

        model = llm_config.get("model", "gpt-4o-mini")
        api_key = llm_config.get("api_key")
        api_base_url = llm_config.get("api_base_url")  # For Groq/custom endpoints

        secret = (
            Secret.from_token(api_key)
            if api_key
            else Secret.from_env_var("OPENAI_API_KEY")
        )

        kwargs = {"model": model, "api_key": secret}
        if api_base_url:
            kwargs["api_base_url"] = api_base_url

        return OpenAIGenerator(**kwargs)


class TokenCounter:
    """Estimate token usage and compression savings."""

    # Rough approximation: 1 token ≈ 4 characters (average for English)
    CHARS_PER_TOKEN = 4

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count for text using simple heuristic.

        Uses character-to-token ratio approximation.

        Args:
            text: Text to estimate token count for.

        Returns:
            Estimated token count.
        """
        return max(1, len(text) // TokenCounter.CHARS_PER_TOKEN)

    @staticmethod
    def calculate_tokens_saved(
        original_docs: list[Document],
        compressed_docs: list[Document],
    ) -> int:
        """Calculate total tokens saved by compression.

        Args:
            original_docs: Original retrieved documents.
            compressed_docs: Compressed/extracted documents.

        Returns:
            Total tokens saved (positive value).
        """
        original_tokens = sum(
            TokenCounter.estimate_tokens(doc.content) for doc in original_docs
        )
        compressed_tokens = sum(
            TokenCounter.estimate_tokens(doc.content) for doc in compressed_docs
        )
        return max(0, original_tokens - compressed_tokens)

    @staticmethod
    def calculate_compression_ratio(
        original_tokens: int,
        compressed_tokens: int,
    ) -> float:
        """Calculate compression ratio.

        Args:
            original_tokens: Total tokens in original documents.
            compressed_tokens: Total tokens after compression.

        Returns:
            Compression ratio (compressed / original). Lower is better.
        """
        if original_tokens == 0:
            return 1.0
        return compressed_tokens / original_tokens


def prepare_retrieval_batch(
    documents: list[Document], batch_size: int
) -> list[list[Document]]:
    """Prepare documents for batch retrieval operations.

    Args:
        documents: List of documents to batch.
        batch_size: Number of documents per batch.

    Returns:
        List of document batches.
    """
    batches: list[list[Document]] = []
    for i in range(0, len(documents), batch_size):
        batches.append(documents[i : i + batch_size])
    return batches


def format_compression_results(
    results: list[RankerResult], include_metadata: bool = True
) -> dict[str, Any]:
    """Format compression results for output and logging.

    Args:
        results: List of RankerResult objects from compression.
        include_metadata: Whether to include document metadata.

    Returns:
        Dictionary with formatted results.
    """
    return {
        "total_results": len(results),
        "results": [
            {
                "rank": result.rank,
                "score": result.score,
                "content": result.document.content[:200] + "..."
                if len(result.document.content) > 200
                else result.document.content,
                "token_count": result.token_count,
                "metadata": result.document.meta if include_metadata else None,
            }
            for result in results
        ],
    }
