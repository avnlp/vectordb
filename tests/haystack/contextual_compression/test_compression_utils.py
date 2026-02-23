"""Comprehensive tests for compression utilities module.

Tests cover:
- RankerResult dataclass
- CompressorFactory with all reranker types and LLM extractors
- TokenCounter utility functions
- prepare_retrieval_batch function
- format_compression_results function
"""

import sys
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.contextual_compression.compression_utils import (
    CompressorFactory,
    RankerResult,
    TokenCounter,
    format_compression_results,
    prepare_retrieval_batch,
)


class TestRankerResult:
    """Tests for RankerResult dataclass."""

    def test_ranker_result_creation(self) -> None:
        """Test creating a RankerResult with all fields."""
        doc = Document(content="Test content", meta={"id": "1"})
        result = RankerResult(
            document=doc,
            score=0.95,
            rank=1,
            token_count=100,
        )

        assert result.document == doc
        assert result.score == 0.95
        assert result.rank == 1
        assert result.token_count == 100

    def test_ranker_result_defaults(self) -> None:
        """Test RankerResult with default values."""
        doc = Document(content="Test content")
        result = RankerResult(document=doc, score=0.8)

        assert result.rank is None
        assert result.token_count == 0

    def test_ranker_result_immutable(self) -> None:
        """Test that RankerResult fields can be modified (dataclass is not frozen)."""
        doc = Document(content="Test content")
        result = RankerResult(document=doc, score=0.8)

        # Should be able to modify
        result.rank = 2
        result.token_count = 50

        assert result.rank == 2
        assert result.token_count == 50


class TestCompressorFactoryCreateCompressor:
    """Tests for CompressorFactory.create_compressor method."""

    @patch.object(CompressorFactory, "create_reranker")
    def test_create_compressor_with_compression_type_reranking(
        self, mock_create_reranker: MagicMock
    ) -> None:
        """Test creating compressor with explicit reranking type."""
        mock_reranker = MagicMock()
        mock_create_reranker.return_value = mock_reranker

        config = {
            "compression": {
                "type": "reranking",
                "reranker": {"type": "cross_encoder"},
            }
        }

        result = CompressorFactory.create_compressor(config)

        assert result == mock_reranker
        mock_create_reranker.assert_called_once_with(config)

    @patch.object(CompressorFactory, "create_llm_extractor")
    def test_create_compressor_with_compression_type_llm_extraction(
        self, mock_create_llm_extractor: MagicMock
    ) -> None:
        """Test creating compressor with explicit llm_extraction type."""
        mock_extractor = MagicMock()
        mock_create_llm_extractor.return_value = mock_extractor

        config = {
            "compression": {
                "type": "llm_extraction",
                "llm": {"model": "gpt-4o-mini"},
            }
        }

        result = CompressorFactory.create_compressor(config)

        assert result == mock_extractor
        mock_create_llm_extractor.assert_called_once_with(config)

    @patch.object(CompressorFactory, "create_reranker")
    def test_create_compressor_legacy_flat_reranker_config(
        self, mock_create_reranker: MagicMock
    ) -> None:
        """Test creating compressor with legacy flat reranker config."""
        mock_reranker = MagicMock()
        mock_create_reranker.return_value = mock_reranker

        config = {"reranker": {"type": "cross_encoder"}}

        result = CompressorFactory.create_compressor(config)

        assert result == mock_reranker
        mock_create_reranker.assert_called_once_with(config)

    @patch.object(CompressorFactory, "create_llm_extractor")
    def test_create_compressor_legacy_flat_llm_config(
        self, mock_create_llm_extractor: MagicMock
    ) -> None:
        """Test creating compressor with legacy flat llm_compression config."""
        mock_extractor = MagicMock()
        mock_create_llm_extractor.return_value = mock_extractor

        config = {"llm_compression": {"model": "gpt-4o-mini"}}

        result = CompressorFactory.create_compressor(config)

        assert result == mock_extractor
        mock_create_llm_extractor.assert_called_once_with(config)

    @patch.object(CompressorFactory, "create_reranker")
    def test_create_compressor_default_to_reranking(
        self, mock_create_reranker: MagicMock
    ) -> None:
        """Test that empty config defaults to reranking."""
        mock_reranker = MagicMock()
        mock_create_reranker.return_value = mock_reranker

        config = {}

        result = CompressorFactory.create_compressor(config)

        assert result == mock_reranker
        mock_create_reranker.assert_called_once_with(config)

    def test_create_compressor_unsupported_type(self) -> None:
        """Test that unsupported compression type raises ValueError."""
        config = {"compression": {"type": "unsupported_type"}}

        with pytest.raises(ValueError, match="Unsupported compression type"):
            CompressorFactory.create_compressor(config)


class TestCompressorFactoryCreateReranker:
    """Tests for CompressorFactory.create_reranker method."""

    @patch("vectordb.haystack.contextual_compression.compression_utils.UnifiedReranker")
    def test_create_cross_encoder_reranker(
        self, mock_unified_reranker: MagicMock
    ) -> None:
        """Test creating cross encoder reranker."""
        mock_reranker = MagicMock()
        mock_unified_reranker.create_cross_encoder_reranker.return_value = mock_reranker
        mock_unified_reranker.get_default_model.return_value = (
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        config = {
            "compression": {
                "reranker": {
                    "type": "cross_encoder",
                    "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    "top_k": 10,
                }
            }
        }

        result = CompressorFactory.create_reranker(config)

        assert result == mock_reranker
        mock_unified_reranker.create_cross_encoder_reranker.assert_called_once_with(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_k=10,
        )

    @patch("vectordb.haystack.contextual_compression.compression_utils.UnifiedReranker")
    def test_create_cross_encoder_light(self, mock_unified_reranker: MagicMock) -> None:
        """Test creating cross_encoder_light reranker."""
        mock_reranker = MagicMock()
        mock_unified_reranker.create_cross_encoder_reranker.return_value = mock_reranker
        mock_unified_reranker.get_default_model.return_value = (
            "cross-encoder/ms-marco-MiniLM-L-2-v2"
        )

        config = {
            "compression": {
                "reranker": {
                    "type": "cross_encoder_light",
                    "top_k": 5,
                }
            }
        }

        result = CompressorFactory.create_reranker(config)

        assert result == mock_reranker
        mock_unified_reranker.create_cross_encoder_reranker.assert_called_once()

    @patch("vectordb.haystack.contextual_compression.compression_utils.UnifiedReranker")
    def test_create_cross_encoder_qwen(self, mock_unified_reranker: MagicMock) -> None:
        """Test creating cross_encoder_qwen reranker."""
        mock_reranker = MagicMock()
        mock_unified_reranker.create_cross_encoder_reranker.return_value = mock_reranker

        config = {
            "compression": {
                "reranker": {
                    "type": "cross_encoder_qwen",
                    "top_k": 5,
                }
            }
        }

        result = CompressorFactory.create_reranker(config)

        assert result == mock_reranker
        mock_unified_reranker.create_cross_encoder_reranker.assert_called_once()

    @patch("vectordb.haystack.contextual_compression.compression_utils.UnifiedReranker")
    def test_create_cohere_reranker(self, mock_unified_reranker: MagicMock) -> None:
        """Test creating Cohere reranker."""
        mock_reranker = MagicMock()
        mock_unified_reranker.create_cohere_reranker.return_value = mock_reranker

        config = {
            "compression": {
                "reranker": {
                    "type": "cohere",
                    "api_key": "test-api-key",
                    "model": "rerank-english-v3.0",
                    "top_n": 10,
                }
            }
        }

        result = CompressorFactory.create_reranker(config)

        assert result == mock_reranker
        mock_unified_reranker.create_cohere_reranker.assert_called_once_with(
            api_key="test-api-key",
            model="rerank-english-v3.0",
            top_k=10,
        )

    @patch("vectordb.haystack.contextual_compression.compression_utils.UnifiedReranker")
    def test_create_cohere_reranker_with_top_k(
        self, mock_unified_reranker: MagicMock
    ) -> None:
        """Test creating Cohere reranker using top_k instead of top_n."""
        mock_reranker = MagicMock()
        mock_unified_reranker.create_cohere_reranker.return_value = mock_reranker

        config = {
            "compression": {
                "reranker": {
                    "type": "cohere",
                    "top_k": 8,
                }
            }
        }

        result = CompressorFactory.create_reranker(config)

        assert result == mock_reranker
        mock_unified_reranker.create_cohere_reranker.assert_called_once_with(
            api_key=None,
            model="rerank-english-v3.0",
            top_k=8,
        )

    @patch("vectordb.haystack.contextual_compression.compression_utils.UnifiedReranker")
    def test_create_cohere_reranker_default_model(
        self, mock_unified_reranker: MagicMock
    ) -> None:
        """Test creating Cohere reranker with default model."""
        mock_reranker = MagicMock()
        mock_unified_reranker.create_cohere_reranker.return_value = mock_reranker

        config = {"compression": {"reranker": {"type": "cohere"}}}

        CompressorFactory.create_reranker(config)

        mock_unified_reranker.create_cohere_reranker.assert_called_once_with(
            api_key=None,
            model="rerank-english-v3.0",
            top_k=5,
        )

    @patch("vectordb.haystack.contextual_compression.compression_utils.UnifiedReranker")
    def test_create_voyage_reranker(self, mock_unified_reranker: MagicMock) -> None:
        """Test creating Voyage reranker.

        Note: This test is skipped if voyage-embedders-haystack is not installed.
        """
        pytest.importorskip(
            "haystack_integrations.components.rankers.voyage",
            reason="voyage-embedders-haystack not installed",
        )

        mock_reranker = MagicMock()

        # Mock VoyageRanker at the source module level
        with patch(
            "haystack_integrations.components.rankers.voyage.VoyageRanker"
        ) as mock_voyage:
            mock_voyage.return_value = mock_reranker

            config = {
                "compression": {
                    "reranker": {
                        "type": "voyage",
                        "api_key": "test-voyage-key",
                        "model": "rerank-2",
                    }
                }
            }

            result = CompressorFactory.create_reranker(config)

            assert result == mock_reranker
            mock_voyage.assert_called_once()

    @patch("vectordb.haystack.contextual_compression.compression_utils.UnifiedReranker")
    def test_create_voyage_reranker_import_error(
        self, mock_unified_reranker: MagicMock
    ) -> None:
        """Test Voyage reranker raises ValueError when import fails."""
        # Mock the import to fail when trying to import VoyageRanker
        config = {
            "compression": {
                "reranker": {
                    "type": "voyage",
                }
            }
        }

        # The import happens inside the function - we need to make it fail
        # by temporarily removing the module from sys.modules
        voyage_module = sys.modules.pop(
            "haystack_integrations.components.rankers.voyage", None
        )
        try:
            # Create a mock that raises ImportError when trying to import
            import builtins

            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if "voyage" in name:
                    raise ImportError("No module named 'voyage'")
                return original_import(name, *args, **kwargs)

            builtins.__import__ = mock_import
            try:
                with pytest.raises(ValueError, match="Voyage reranker requires"):
                    CompressorFactory.create_reranker(config)
            finally:
                builtins.__import__ = original_import
        finally:
            # Restore the module if it was present
            if voyage_module is not None:
                sys.modules["haystack_integrations.components.rankers.voyage"] = (
                    voyage_module
                )

    @patch("vectordb.haystack.contextual_compression.compression_utils.UnifiedReranker")
    def test_create_bge_reranker(self, mock_unified_reranker: MagicMock) -> None:
        """Test creating BGE reranker."""
        mock_reranker = MagicMock()
        mock_unified_reranker.create_cross_encoder_reranker.return_value = mock_reranker

        config = {
            "compression": {
                "reranker": {
                    "type": "bge",
                    "model": "BAAI/bge-reranker-v2-m3",
                    "top_n": 5,
                }
            }
        }

        result = CompressorFactory.create_reranker(config)

        assert result == mock_reranker
        mock_unified_reranker.create_cross_encoder_reranker.assert_called_once_with(
            model="BAAI/bge-reranker-v2-m3",
            top_k=5,
        )

    @patch("vectordb.haystack.contextual_compression.compression_utils.UnifiedReranker")
    def test_create_bge_reranker_default_model(
        self, mock_unified_reranker: MagicMock
    ) -> None:
        """Test creating BGE reranker with default model."""
        mock_reranker = MagicMock()
        mock_unified_reranker.create_cross_encoder_reranker.return_value = mock_reranker

        config = {"compression": {"reranker": {"type": "bge"}}}

        CompressorFactory.create_reranker(config)

        mock_unified_reranker.create_cross_encoder_reranker.assert_called_once_with(
            model="BAAI/bge-reranker-v2-m3",
            top_k=5,
        )

    @patch("vectordb.haystack.contextual_compression.compression_utils.UnifiedReranker")
    def test_create_reranker_flat_config(
        self, mock_unified_reranker: MagicMock
    ) -> None:
        """Test creating reranker with flat config (no compression wrapper)."""
        mock_reranker = MagicMock()
        mock_unified_reranker.create_cross_encoder_reranker.return_value = mock_reranker

        config = {"reranker": {"type": "cross_encoder", "top_k": 5}}

        result = CompressorFactory.create_reranker(config)

        assert result == mock_reranker

    def test_create_reranker_unsupported_type(self) -> None:
        """Test that unsupported reranker type raises ValueError."""
        config = {"compression": {"reranker": {"type": "unknown_reranker"}}}

        with pytest.raises(ValueError, match="Unsupported reranker type"):
            CompressorFactory.create_reranker(config)


class TestCompressorFactoryCreateLLMExtractor:
    """Tests for CompressorFactory.create_llm_extractor method."""

    @patch("haystack.components.generators.OpenAIGenerator")
    @patch("haystack.utils.Secret")
    def test_create_llm_extractor_with_api_key(
        self, mock_secret: MagicMock, mock_generator: MagicMock
    ) -> None:
        """Test creating LLM extractor with explicit API key."""
        mock_extractor = MagicMock()
        mock_generator.return_value = mock_extractor
        mock_secret.from_token.return_value = MagicMock()

        config = {
            "compression": {
                "llm": {
                    "model": "gpt-4o-mini",
                    "api_key": "test-openai-key",
                }
            }
        }

        result = CompressorFactory.create_llm_extractor(config)

        assert result == mock_extractor
        mock_secret.from_token.assert_called_once_with("test-openai-key")
        mock_generator.assert_called_once()

    @patch("haystack.components.generators.OpenAIGenerator")
    @patch("haystack.utils.Secret")
    def test_create_llm_extractor_with_env_var(
        self, mock_secret: MagicMock, mock_generator: MagicMock
    ) -> None:
        """Test creating LLM extractor using environment variable."""
        mock_extractor = MagicMock()
        mock_generator.return_value = mock_extractor
        mock_secret.from_env_var.return_value = MagicMock()

        config = {"compression": {"llm": {"model": "gpt-4o"}}}

        result = CompressorFactory.create_llm_extractor(config)

        assert result == mock_extractor
        mock_secret.from_env_var.assert_called_once_with("OPENAI_API_KEY")

    @patch("haystack.components.generators.OpenAIGenerator")
    @patch("haystack.utils.Secret")
    def test_create_llm_extractor_with_base_url(
        self, mock_secret: MagicMock, mock_generator: MagicMock
    ) -> None:
        """Test creating LLM extractor with custom base URL (e.g., Groq)."""
        mock_extractor = MagicMock()
        mock_generator.return_value = mock_extractor
        mock_secret.from_token.return_value = MagicMock()

        config = {
            "compression": {
                "llm": {
                    "model": "llama-3.1-8b-instant",
                    "api_key": "test-groq-key",
                    "api_base_url": "https://api.groq.com/openai/v1",
                }
            }
        }

        result = CompressorFactory.create_llm_extractor(config)

        assert result == mock_extractor
        call_kwargs = mock_generator.call_args.kwargs
        assert call_kwargs["model"] == "llama-3.1-8b-instant"
        assert call_kwargs["api_base_url"] == "https://api.groq.com/openai/v1"

    @patch("haystack.components.generators.OpenAIGenerator")
    @patch("haystack.utils.Secret")
    def test_create_llm_extractor_flat_config(
        self, mock_secret: MagicMock, mock_generator: MagicMock
    ) -> None:
        """Test creating LLM extractor with flat config."""
        mock_extractor = MagicMock()
        mock_generator.return_value = mock_extractor
        mock_secret.from_token.return_value = MagicMock()

        config = {"llm_compression": {"model": "gpt-4o-mini", "api_key": "test-key"}}

        result = CompressorFactory.create_llm_extractor(config)

        assert result == mock_extractor

    def test_create_llm_extractor_no_config(self) -> None:
        """Test that missing LLM config raises ValueError."""
        config = {"compression": {}}

        with pytest.raises(ValueError, match="No LLM compression config found"):
            CompressorFactory.create_llm_extractor(config)


class TestTokenCounter:
    """Tests for TokenCounter utility class."""

    def test_estimate_tokens_short_text(self) -> None:
        """Test token estimation for short text."""
        text = "Hello world"
        # 11 chars / 4 = 2.75 -> max(1, 2) = 2
        assert TokenCounter.estimate_tokens(text) == 2

    def test_estimate_tokens_long_text(self) -> None:
        """Test token estimation for longer text."""
        text = "This is a longer text that should be estimated correctly."
        # 57 chars / 4 = 14.25 -> 14
        assert TokenCounter.estimate_tokens(text) == 14

    def test_estimate_tokens_empty_text(self) -> None:
        """Test token estimation for empty text returns at least 1."""
        text = ""
        assert TokenCounter.estimate_tokens(text) == 1

    def test_estimate_tokens_exact_multiple(self) -> None:
        """Test token estimation when text length is exact multiple of 4."""
        text = "abcd"  # 4 chars / 4 = 1
        assert TokenCounter.estimate_tokens(text) == 1

    def test_calculate_tokens_saved(self) -> None:
        """Test calculating tokens saved."""
        original_docs = [
            Document(content="a" * 40),  # 10 tokens
            Document(content="b" * 40),  # 10 tokens
        ]
        compressed_docs = [
            Document(content="c" * 20),  # 5 tokens
        ]

        saved = TokenCounter.calculate_tokens_saved(original_docs, compressed_docs)
        # (10 + 10) - 5 = 15
        assert saved == 15

    def test_calculate_tokens_saved_no_savings(self) -> None:
        """Test when compressed docs are larger or equal."""
        original_docs = [Document(content="a" * 40)]  # 10 tokens
        compressed_docs = [Document(content="b" * 80)]  # 20 tokens

        saved = TokenCounter.calculate_tokens_saved(original_docs, compressed_docs)
        # max(0, 10 - 20) = 0
        assert saved == 0

    def test_calculate_tokens_saved_empty_lists(self) -> None:
        """Test with empty document lists."""
        assert TokenCounter.calculate_tokens_saved([], []) == 0
        # "test" has 4 chars, 4/4 = 1 token
        assert TokenCounter.calculate_tokens_saved([Document(content="test")], []) == 1

    def test_calculate_compression_ratio(self) -> None:
        """Test compression ratio calculation."""
        ratio = TokenCounter.calculate_compression_ratio(100, 50)
        assert ratio == 0.5

    def test_calculate_compression_ratio_zero_original(self) -> None:
        """Test compression ratio with zero original tokens."""
        ratio = TokenCounter.calculate_compression_ratio(0, 50)
        assert ratio == 1.0

    def test_calculate_compression_ratio_equal(self) -> None:
        """Test compression ratio when equal."""
        ratio = TokenCounter.calculate_compression_ratio(100, 100)
        assert ratio == 1.0

    def test_calculate_compression_ratio_larger_compressed(self) -> None:
        """Test compression ratio when compressed is larger."""
        ratio = TokenCounter.calculate_compression_ratio(100, 150)
        assert ratio == 1.5


class TestPrepareRetrievalBatch:
    """Tests for prepare_retrieval_batch function."""

    def test_prepare_retrieval_batch_exact_multiple(self) -> None:
        """Test batching when document count is exact multiple of batch size."""
        docs = [Document(content=f"doc{i}") for i in range(6)]
        batches = prepare_retrieval_batch(docs, 3)

        assert len(batches) == 2
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3

    def test_prepare_retrieval_batch_remainder(self) -> None:
        """Test batching with remainder."""
        docs = [Document(content=f"doc{i}") for i in range(5)]
        batches = prepare_retrieval_batch(docs, 2)

        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1

    def test_prepare_retrieval_batch_single_batch(self) -> None:
        """Test when all docs fit in single batch."""
        docs = [Document(content=f"doc{i}") for i in range(3)]
        batches = prepare_retrieval_batch(docs, 10)

        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_prepare_retrieval_batch_empty(self) -> None:
        """Test with empty document list."""
        batches = prepare_retrieval_batch([], 5)

        assert batches == []

    def test_prepare_retrieval_batch_batch_size_one(self) -> None:
        """Test with batch size of 1."""
        docs = [Document(content=f"doc{i}") for i in range(3)]
        batches = prepare_retrieval_batch(docs, 1)

        assert len(batches) == 3
        assert all(len(batch) == 1 for batch in batches)


class TestFormatCompressionResults:
    """Tests for format_compression_results function."""

    def test_format_compression_results_basic(self) -> None:
        """Test basic result formatting."""
        doc1 = Document(content="Short content", meta={"source": "test"})
        doc2 = Document(content="Another content", meta={"source": "test2"})

        results = [
            RankerResult(document=doc1, score=0.9, rank=1, token_count=10),
            RankerResult(document=doc2, score=0.8, rank=2, token_count=15),
        ]

        formatted = format_compression_results(results)

        assert formatted["total_results"] == 2
        assert len(formatted["results"]) == 2

        first = formatted["results"][0]
        assert first["rank"] == 1
        assert first["score"] == 0.9
        assert first["token_count"] == 10
        assert first["metadata"] == {"source": "test"}

    def test_format_compression_results_long_content_truncation(self) -> None:
        """Test that long content is truncated."""
        long_content = "a" * 300
        doc = Document(content=long_content, meta={})

        results = [RankerResult(document=doc, score=0.9, rank=1, token_count=100)]

        formatted = format_compression_results(results)

        assert formatted["results"][0]["content"].endswith("...")
        assert len(formatted["results"][0]["content"]) == 203  # 200 + "..."

    def test_format_compression_results_short_content_no_truncation(self) -> None:
        """Test that short content is not truncated."""
        short_content = "Short text"
        doc = Document(content=short_content, meta={})

        results = [RankerResult(document=doc, score=0.9, rank=1, token_count=10)]

        formatted = format_compression_results(results)

        assert formatted["results"][0]["content"] == short_content
        assert not formatted["results"][0]["content"].endswith("...")

    def test_format_compression_results_no_metadata(self) -> None:
        """Test formatting without metadata."""
        doc = Document(content="Content", meta={"source": "test"})
        results = [RankerResult(document=doc, score=0.9, rank=1, token_count=10)]

        formatted = format_compression_results(results, include_metadata=False)

        assert formatted["results"][0]["metadata"] is None

    def test_format_compression_results_empty(self) -> None:
        """Test formatting empty results."""
        formatted = format_compression_results([])

        assert formatted["total_results"] == 0
        assert formatted["results"] == []

    def test_format_compression_results_none_rank(self) -> None:
        """Test formatting with None rank."""
        doc = Document(content="Content", meta={})
        results = [RankerResult(document=doc, score=0.9, rank=None, token_count=10)]

        formatted = format_compression_results(results)

        assert formatted["results"][0]["rank"] is None
