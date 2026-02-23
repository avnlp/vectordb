"""Tests for ContextCompressor component (LangChain).

This module tests the ContextCompressor class which implements contextual compression
techniques for pruning retrieved documents. Compression reduces noise and improves
RAG answer quality by retaining only the most relevant content.

Compression Modes:
    - Reranking: Uses cross-encoder to score and filter documents
    - LLM Extraction: Uses LLM to extract relevant passages from documents

Reranking Mode:
    - Cross-encoder scores query-document relevance
    - Documents sorted by score
    - Top-k documents returned
    - Fast, no LLM calls during compression

LLM Extraction Mode:
    - LLM extracts relevant passages from all documents
    - Returns single synthesized document
    - More expensive but higher quality
    - Useful for long documents with mixed relevance

Test Coverage:
    - Initialization with both compression modes
    - Mode validation and error handling
    - Reranking with various document counts
    - LLM extraction with different content types
    - Edge cases: empty docs, long docs, special characters
    - Metadata preservation through compression
    - Integration patterns for RAG workflows

All tests mock LLM and reranker to avoid external API calls.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage


class TestContextCompressor:
    """Unit tests for ContextCompressor class.

    Base test class providing fixtures for ContextCompressor tests.
    Sets up mock LLM, reranker, and compressor instances for use
    across test subclasses.

    Fixtures:
        mock_llm: MagicMock simulating LangChain LLM interface
        mock_reranker: MagicMock simulating cross-encoder reranker
        compressor_reranking: ContextCompressor in reranking mode
        compressor_llm_extraction: ContextCompressor in LLM extraction mode
    """

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create mock LLM for testing."""
        return MagicMock()

    @pytest.fixture
    def mock_reranker(self) -> MagicMock:
        """Create mock cross-encoder reranker for testing."""
        return MagicMock()

    @pytest.fixture
    def compressor_reranking(self, mock_reranker):
        """Create ContextCompressor instance with reranking mode."""
        from vectordb.langchain.components.context_compressor import ContextCompressor

        return ContextCompressor(mode="reranking", reranker=mock_reranker)

    @pytest.fixture
    def compressor_llm_extraction(self, mock_llm):
        """Create ContextCompressor instance with LLM extraction mode."""
        from vectordb.langchain.components.context_compressor import ContextCompressor

        return ContextCompressor(mode="llm_extraction", llm=mock_llm)


class TestContextCompressorInitialization(TestContextCompressor):
    """Tests for ContextCompressor initialization and validation.

    Validates that ContextCompressor properly initializes with:
    - Valid reranking mode with reranker component
    - Valid LLM extraction mode with LLM component
    - Proper error handling for invalid configurations

    Configuration Requirements:
        - Reranking mode: requires reranker parameter
        - LLM extraction mode: requires llm parameter
        - Mode must be one of: "reranking", "llm_extraction"
    """

    def test_initialization_with_valid_reranking_mode(self, mock_reranker):
        """Test initialization with valid reranking mode and reranker."""
        from vectordb.langchain.components.context_compressor import ContextCompressor

        compressor = ContextCompressor(mode="reranking", reranker=mock_reranker)
        assert compressor.mode == "reranking"
        assert compressor.reranker is mock_reranker
        assert compressor.llm is None

    def test_initialization_with_valid_llm_extraction_mode(self, mock_llm):
        """Test initialization with valid LLM extraction mode and LLM."""
        from vectordb.langchain.components.context_compressor import ContextCompressor

        compressor = ContextCompressor(mode="llm_extraction", llm=mock_llm)
        assert compressor.mode == "llm_extraction"
        assert compressor.llm is mock_llm
        assert compressor.reranker is None

    def test_initialization_with_invalid_mode_raises_value_error(self):
        """Test initialization raises ValueError for invalid mode."""
        from vectordb.langchain.components.context_compressor import ContextCompressor

        with pytest.raises(ValueError) as exc_info:
            ContextCompressor(mode="invalid_mode", reranker=MagicMock())

        assert "Invalid mode" in str(exc_info.value)
        assert "invalid_mode" in str(exc_info.value)

    def test_initialization_reranking_without_reranker_raises_value_error(self):
        """Test reranking mode requires reranker component."""
        from vectordb.langchain.components.context_compressor import ContextCompressor

        with pytest.raises(ValueError) as exc_info:
            ContextCompressor(mode="reranking", reranker=None)

        assert "Reranker required" in str(exc_info.value)
        assert "reranking" in str(exc_info.value)

    def test_initialization_llm_extraction_without_llm_raises_value_error(self):
        """Test LLM extraction mode requires LLM component."""
        from vectordb.langchain.components.context_compressor import ContextCompressor

        with pytest.raises(ValueError) as exc_info:
            ContextCompressor(mode="llm_extraction", llm=None)

        assert "LLM required" in str(exc_info.value)
        assert "llm_extraction" in str(exc_info.value)

    def test_extraction_template_is_class_attribute(self):
        """Test that extraction template is defined as class attribute."""
        from vectordb.langchain.components.context_compressor import ContextCompressor

        assert hasattr(ContextCompressor, "EXTRACTION_TEMPLATE")
        assert "Relevant passages" in ContextCompressor.EXTRACTION_TEMPLATE
        assert "{query}" in ContextCompressor.EXTRACTION_TEMPLATE
        assert "{documents}" in ContextCompressor.EXTRACTION_TEMPLATE

    def test_initialization_stores_mode_correctly(self, mock_reranker, mock_llm):
        """Test that mode is stored correctly for both modes."""
        from vectordb.langchain.components.context_compressor import ContextCompressor

        reranking_compressor = ContextCompressor(
            mode="reranking", reranker=mock_reranker
        )
        llm_compressor = ContextCompressor(mode="llm_extraction", llm=mock_llm)

        assert reranking_compressor.mode == "reranking"
        assert llm_compressor.mode == "llm_extraction"


class TestContextCompressorReranking(TestContextCompressor):
    """Tests for compress_reranking method.

    Validates the reranking-based compression which uses a cross-encoder
    to score document relevance and returns the top-k highest scoring docs.

    Reranking Process:
        1. Score each document with cross-encoder
        2. Sort documents by relevance score
        3. Return top-k documents

    Benefits:
        - Fast (no LLM calls)
        - Preserves original document structure
        - Configurable compression ratio via top_k

    Edge Cases:
        - Empty document list
        - Fewer documents than top_k
        - Equal scores (stable sort)
        - Negative scores
    """

    def test_compress_reranking_returns_top_k_documents(
        self, compressor_reranking, mock_reranker
    ):
        """Test reranking returns top-k highest scoring documents."""
        docs = [
            Document(page_content="Doc 1 content"),
            Document(page_content="Doc 2 content"),
            Document(page_content="Doc 3 content"),
        ]
        # Scores: doc 3 is highest, then doc 1, then doc 2
        mock_reranker.rank.return_value = [0.3, 0.1, 0.8]

        result = compressor_reranking.compress_reranking("test query", docs, top_k=2)

        assert len(result) == 2
        assert result[0].page_content == "Doc 3 content"  # Highest score
        assert result[1].page_content == "Doc 1 content"  # Second highest

    def test_compress_reranking_returns_all_docs_when_fewer_than_top_k(
        self, compressor_reranking, mock_reranker
    ):
        """Test returns all documents when fewer than top_k."""
        docs = [
            Document(page_content="Doc 1"),
            Document(page_content="Doc 2"),
        ]
        mock_reranker.rank.return_value = [0.5, 0.3]

        result = compressor_reranking.compress_reranking("test", docs, top_k=5)

        assert len(result) == 2

    def test_compress_reranking_empty_documents_returns_empty(
        self, compressor_reranking, mock_reranker
    ):
        """Test empty document list returns empty list."""
        result = compressor_reranking.compress_reranking("test", [], top_k=5)

        assert result == []
        mock_reranker.rank.assert_not_called()

    def test_compress_reranking_preserves_document_metadata(
        self, compressor_reranking, mock_reranker
    ):
        """Test that document metadata is preserved after reranking."""
        docs = [
            Document(page_content="Doc 1", metadata={"source": "test1", "page": 1}),
            Document(page_content="Doc 2", metadata={"source": "test2", "page": 2}),
        ]
        mock_reranker.rank.return_value = [0.5, 0.8]

        result = compressor_reranking.compress_reranking("test", docs, top_k=2)

        assert result[0].metadata["source"] == "test2"
        assert result[0].metadata["page"] == 2
        assert result[1].metadata["source"] == "test1"
        assert result[1].metadata["page"] == 1

    def test_compress_reranking_with_equal_scores(
        self, compressor_reranking, mock_reranker
    ):
        """Test reranking with equal scores maintains original order."""
        docs = [
            Document(page_content="Doc 1"),
            Document(page_content="Doc 2"),
            Document(page_content="Doc 3"),
        ]
        # All scores equal - should maintain relative order due to stable sort
        mock_reranker.rank.return_value = [0.5, 0.5, 0.5]

        result = compressor_reranking.compress_reranking("test", docs, top_k=3)

        assert len(result) == 3
        # Original order should be preserved with stable sort
        assert result[0].page_content == "Doc 1"
        assert result[1].page_content == "Doc 2"
        assert result[2].page_content == "Doc 3"

    def test_compress_reranking_single_document(
        self, compressor_reranking, mock_reranker
    ):
        """Test reranking with single document."""
        docs = [Document(page_content="Single doc")]
        mock_reranker.rank.return_value = [0.9]

        result = compressor_reranking.compress_reranking("test", docs, top_k=5)

        assert len(result) == 1
        assert result[0].page_content == "Single doc"

    def test_compress_reranking_creates_query_document_pairs(
        self, compressor_reranking, mock_reranker
    ):
        """Test that reranker is called with correct query-document pairs."""
        docs = [
            Document(page_content="Content A"),
            Document(page_content="Content B"),
        ]
        mock_reranker.rank.return_value = [0.5, 0.5]

        compressor_reranking.compress_reranking("my query", docs, top_k=2)

        # Verify rank was called with query-document pairs
        mock_reranker.rank.assert_called_once()
        call_args = mock_reranker.rank.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0] == ["my query", "Content A"]
        assert call_args[1] == ["my query", "Content B"]

    def test_compress_reranking_default_top_k(
        self, compressor_reranking, mock_reranker
    ):
        """Test that default top_k is 5."""
        docs = [Document(page_content=f"Doc {i}") for i in range(10)]
        mock_reranker.rank.return_value = list(range(10))

        result = compressor_reranking.compress_reranking("test", docs)

        assert len(result) == 5


class TestContextCompressorLLMExtraction(TestContextCompressor):
    """Tests for compress_llm_extraction method.

    Validates the LLM-based extraction which uses an LLM to extract
    relevant passages from documents and synthesize a single document.

    Extraction Process:
        1. Format documents with query into prompt
        2. LLM extracts relevant passages
        3. Return single synthesized document

    Benefits:
        - Higher quality than reranking
        - Synthesizes information across documents
        - Removes irrelevant content completely

    Trade-offs:
        - Requires LLM call (more expensive)
        - Slower than reranking
        - Loses original document boundaries
    """

    def test_compress_llm_extraction_returns_single_document(
        self, compressor_llm_extraction, mock_llm
    ):
        """Test LLM extraction returns a single compressed document."""
        docs = [
            Document(page_content="Relevant information about Python."),
            Document(page_content="Python is a programming language."),
        ]
        mock_response = AIMessage(
            content="Python is a high-level programming language used for various applications."
        )
        mock_llm.invoke.return_value = mock_response

        result = compressor_llm_extraction.compress_llm_extraction(
            "What is Python?", docs
        )

        assert len(result) == 1
        assert isinstance(result[0], Document)

    def test_compress_llm_extraction_content_from_llm(
        self, compressor_llm_extraction, mock_llm
    ):
        """Test that compressed document content comes from LLM response."""
        docs = [Document(page_content="Some source content.")]
        expected_extraction = "Extracted relevant passage about the topic."
        mock_response = AIMessage(content=expected_extraction)
        mock_llm.invoke.return_value = mock_response

        result = compressor_llm_extraction.compress_llm_extraction("test query", docs)

        assert result[0].page_content == expected_extraction

    def test_compress_llm_extraction_empty_documents_returns_empty(
        self, compressor_llm_extraction, mock_llm
    ):
        """Test empty document list returns empty list."""
        result = compressor_llm_extraction.compress_llm_extraction("test", [])

        assert result == []
        mock_llm.invoke.assert_not_called()

    def test_compress_llm_extraction_sets_metadata(
        self, compressor_llm_extraction, mock_llm
    ):
        """Test that compressed document has correct metadata."""
        docs = [
            Document(page_content="Doc 1"),
            Document(page_content="Doc 2"),
            Document(page_content="Doc 3"),
        ]
        mock_response = AIMessage(content="Extracted content")
        mock_llm.invoke.return_value = mock_response

        result = compressor_llm_extraction.compress_llm_extraction("test", docs)

        assert result[0].metadata["source"] == "compressed"
        assert result[0].metadata["original_doc_count"] == 3

    def test_compress_llm_extraction_single_document(
        self, compressor_llm_extraction, mock_llm
    ):
        """Test LLM extraction with single document."""
        docs = [Document(page_content="Single document content.")]
        mock_response = AIMessage(content="Extracted single doc")
        mock_llm.invoke.return_value = mock_response

        result = compressor_llm_extraction.compress_llm_extraction("test", docs)

        assert len(result) == 1
        assert result[0].metadata["original_doc_count"] == 1

    def test_compress_llm_extraction_strips_whitespace(
        self, compressor_llm_extraction, mock_llm
    ):
        """Test that LLM response is stripped of whitespace."""
        docs = [Document(page_content="Content.")]
        mock_response = AIMessage(content="  Extracted content with whitespace  \n\n")
        mock_llm.invoke.return_value = mock_response

        result = compressor_llm_extraction.compress_llm_extraction("test", docs)

        assert result[0].page_content == "Extracted content with whitespace"

    def test_compress_llm_extraction_invokes_llm_with_formatted_prompt(
        self, compressor_llm_extraction, mock_llm
    ):
        """Test that LLM is invoked with correctly formatted prompt."""
        docs = [
            Document(page_content="First document content."),
            Document(page_content="Second document content."),
        ]
        mock_response = AIMessage(content="Extracted content")
        mock_llm.invoke.return_value = mock_response

        compressor_llm_extraction.compress_llm_extraction("test query", docs)

        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args[0][0]
        assert "test query" in call_args
        assert "First document content" in call_args
        assert "Second document content" in call_args

    def test_compress_llm_extraction_document_formatting(
        self, compressor_llm_extraction, mock_llm
    ):
        """Test that documents are formatted with Document X prefix."""
        docs = [
            Document(page_content="Content A"),
            Document(page_content="Content B"),
        ]
        mock_response = AIMessage(content="Extracted")
        mock_llm.invoke.return_value = mock_response

        compressor_llm_extraction.compress_llm_extraction("test", docs)

        call_args = mock_llm.invoke.call_args[0][0]
        assert "Document 1:" in call_args
        assert "Document 2:" in call_args
        assert "Content A" in call_args
        assert "Content B" in call_args


class TestContextCompressorCompress(TestContextCompressor):
    """Tests for compress method (main interface).

    Validates the unified compress() interface that dispatches to
    specific compression implementations based on configured mode.

    Interface Design:
        - Single method for both compression modes
        - Mode determined at initialization
        - Same signature regardless of mode

    Usage Pattern:
        compressor = ContextCompressor(mode="reranking", reranker=r)
        compressed = compressor.compress(query, docs, top_k=5)
    """

    def test_compress_reranking_mode(self, compressor_reranking, mock_reranker):
        """Test compress uses reranking mode when configured."""
        docs = [
            Document(page_content="Doc 1"),
            Document(page_content="Doc 2"),
        ]
        mock_reranker.rank.return_value = [0.5, 0.8]

        result = compressor_reranking.compress("test query", docs, top_k=2)

        assert len(result) == 2
        assert result[0].page_content == "Doc 2"

    def test_compress_llm_extraction_mode(self, compressor_llm_extraction, mock_llm):
        """Test compress uses LLM extraction mode when configured."""
        docs = [Document(page_content="Content")]
        mock_response = AIMessage(content="Extracted")
        mock_llm.invoke.return_value = mock_response

        result = compressor_llm_extraction.compress("test query", docs)

        assert len(result) == 1
        assert result[0].page_content == "Extracted"

    def test_compress_top_k_passed_to_reranking(
        self, compressor_reranking, mock_reranker
    ):
        """Test that top_k parameter is passed to reranking."""
        docs = [Document(page_content=f"Doc {i}") for i in range(10)]
        mock_reranker.rank.return_value = list(range(10))

        result = compressor_reranking.compress("test", docs, top_k=3)

        assert len(result) == 3


class TestContextCompressorEdgeCases(TestContextCompressor):
    """Tests for edge cases and error handling.

    Validates robust handling of boundary conditions and unexpected inputs.
    ContextCompressor should degrade gracefully with unusual content.

    Edge Cases Covered:
        - Very long documents (10k+ words)
        - Empty document content
        - Special characters and Unicode
        - Negative reranker scores
        - Zero top_k value
        - Metadata-only documents

    Design Philosophy:
        Compression should never crash; it should return best-effort results.
    """

    def test_compress_reranking_with_very_long_documents(
        self, compressor_reranking, mock_reranker
    ):
        """Test reranking with very long document content."""
        long_content = "word " * 10000
        docs = [
            Document(page_content=long_content),
            Document(page_content="Short content"),
        ]
        mock_reranker.rank.return_value = [0.9, 0.5]

        result = compressor_reranking.compress_reranking("test", docs, top_k=1)

        assert len(result) == 1
        assert long_content in result[0].page_content

    def test_compress_llm_extraction_with_very_long_documents(
        self, compressor_llm_extraction, mock_llm
    ):
        """Test LLM extraction with many long documents."""
        long_content = "content " * 5000
        docs = [
            Document(page_content=long_content),
            Document(page_content=long_content),
            Document(page_content=long_content),
        ]
        mock_response = AIMessage(content="Extracted summary")
        mock_llm.invoke.return_value = mock_response

        result = compressor_llm_extraction.compress_llm_extraction("test", docs)

        assert len(result) == 1
        assert result[0].metadata["original_doc_count"] == 3

    def test_compress_with_empty_query(self, compressor_reranking, mock_reranker):
        """Test compression works with empty query string."""
        docs = [Document(page_content="Doc content")]
        mock_reranker.rank.return_value = [0.5]

        result = compressor_reranking.compress_reranking("", docs, top_k=1)

        assert len(result) == 1

    def test_compress_reranking_with_special_characters(
        self, compressor_reranking, mock_reranker
    ):
        """Test reranking with documents containing special characters."""
        docs = [
            Document(page_content="Document with $pecial ch@racters & symbols!"),
            Document(page_content="Unicode: 你好 مرحبا 🚀"),
        ]
        mock_reranker.rank.return_value = [0.5, 0.8]

        result = compressor_reranking.compress_reranking("test", docs, top_k=2)

        assert len(result) == 2
        assert "🚀" in result[0].page_content

    def test_compress_llm_extraction_strips_response_content(
        self, compressor_llm_extraction, mock_llm
    ):
        """Test that response.content attribute is accessed correctly."""
        docs = [Document(page_content="Source content.")]
        # Mock response with content attribute
        mock_response = MagicMock()
        mock_response.content = "Extracted from content attribute"

        mock_llm.invoke.return_value = mock_response

        result = compressor_llm_extraction.compress_llm_extraction("test", docs)

        assert result[0].page_content == "Extracted from content attribute"

    def test_compress_reranking_zero_top_k(self, compressor_reranking, mock_reranker):
        """Test reranking with top_k=0 returns empty list."""
        docs = [
            Document(page_content="Doc 1"),
            Document(page_content="Doc 2"),
        ]
        mock_reranker.rank.return_value = [0.5, 0.8]

        result = compressor_reranking.compress_reranking("test", docs, top_k=0)

        assert result == []

    def test_compress_with_documents_with_metadata_only(
        self, compressor_reranking, mock_reranker
    ):
        """Test reranking preserves metadata-only documents."""
        docs = [
            Document(page_content="", metadata={"source": "metadata-only"}),
        ]
        mock_reranker.rank.return_value = [0.5]

        result = compressor_reranking.compress_reranking("test", docs, top_k=1)

        assert len(result) == 1
        assert result[0].metadata["source"] == "metadata-only"

    def test_compress_reranking_negative_scores(
        self, compressor_reranking, mock_reranker
    ):
        """Test reranking handles negative scores correctly."""
        docs = [
            Document(page_content="Doc 1"),
            Document(page_content="Doc 2"),
            Document(page_content="Doc 3"),
        ]
        # Negative scores should still be sorted correctly
        mock_reranker.rank.return_value = [-0.1, 0.5, -0.5]

        result = compressor_reranking.compress_reranking("test", docs, top_k=3)

        assert len(result) == 3
        assert result[0].page_content == "Doc 2"  # Highest score (0.5)


class TestContextCompressorIntegrationPatterns(TestContextCompressor):
    """Tests for integration patterns and usage scenarios.

    Validates typical usage patterns in production RAG pipelines.
    Demonstrates how ContextCompressor integrates with retrieval
    and generation components.

    Patterns Tested:
        - Reranking to filter irrelevant documents
        - LLM extraction to synthesize relevant passages
        - Metadata preservation for citation tracking
        - Empty result handling

    Performance Considerations:
        - Reranking: O(n) cross-encoder calls
        - LLM extraction: 1 LLM call regardless of doc count
        - Trade-off: speed vs. quality
    """

    def test_reranking_preserves_relevant_info_by_score(
        self, compressor_reranking, mock_reranker
    ):
        """Test that reranking preserves documents by relevance score."""
        docs = [
            Document(page_content="Irrelevant content about weather"),
            Document(page_content="Python programming language info"),
            Document(page_content="Irrelevant sports news"),
            Document(page_content="Python syntax and features"),
        ]
        # Python-related docs should have higher scores
        mock_reranker.rank.return_value = [0.1, 0.9, 0.2, 0.85]

        result = compressor_reranking.compress_reranking(
            "What is Python?", docs, top_k=2
        )

        assert len(result) == 2
        # Should return the two highest-scoring (Python-related) docs
        assert "Python programming language info" in result[0].page_content
        assert "Python syntax and features" in result[1].page_content

    def test_llm_extraction_compresses_to_relevant_passages(
        self, compressor_llm_extraction, mock_llm
    ):
        """Test that LLM extraction compresses to relevant passages."""
        docs = [
            Document(page_content="Python was created by Guido van Rossum in 1991."),
            Document(page_content="Python is named after Monty Python, not the snake."),
            Document(page_content="Python supports multiple programming paradigms."),
        ]
        mock_response = AIMessage(
            content="Python was created by Guido van Rossum in 1991. Python is named after Monty Python, not the snake."
        )
        mock_llm.invoke.return_value = mock_response

        result = compressor_llm_extraction.compress_llm_extraction(
            "Who created Python and what is its name?", docs
        )

        assert len(result) == 1
        # Content should be from LLM extraction
        assert "Guido van Rossum" in result[0].page_content
        assert "Monty Python" in result[0].page_content

    def test_compress_empty_list_returns_empty(
        self, compressor_reranking, compressor_llm_extraction
    ):
        """Test that compress returns empty list for empty input."""
        result_rerank = compressor_reranking.compress("test", [])
        result_llm = compressor_llm_extraction.compress("test", [])

        assert result_rerank == []
        assert result_llm == []

    def test_compress_preserves_important_metadata(
        self, compressor_reranking, mock_reranker
    ):
        """Test that important document metadata is preserved."""
        docs = [
            Document(
                page_content="Content 1", metadata={"page": 1, "source": "doc1.pdf"}
            ),
            Document(
                page_content="Content 2", metadata={"page": 2, "source": "doc2.pdf"}
            ),
        ]
        mock_reranker.rank.return_value = [0.3, 0.7]

        result = compressor_reranking.compress_reranking("test", docs, top_k=2)

        assert result[0].metadata["page"] == 2
        assert result[0].metadata["source"] == "doc2.pdf"
        assert result[1].metadata["page"] == 1
        assert result[1].metadata["source"] == "doc1.pdf"

    def test_both_modes_return_list_type(
        self, compressor_reranking, compressor_llm_extraction, mock_reranker, mock_llm
    ):
        """Test that both compression modes return a list."""
        doc = [Document(page_content="Test content")]

        # Reranking mode
        mock_reranker.rank.return_value = [0.5]
        result_rerank = compressor_reranking.compress("test", doc)

        # LLM extraction mode
        mock_llm.invoke.return_value = AIMessage(content="Extracted")
        result_llm = compressor_llm_extraction.compress("test", doc)

        assert isinstance(result_rerank, list)
        assert isinstance(result_llm, list)
