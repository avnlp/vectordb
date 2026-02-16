"""Tests for document format converters.

This module tests the DocumentConverter class which provides bidirectional
conversion between standardized dictionary format and framework-specific
document types (Haystack and LangChain).

The standardized format uses:
    - 'text': The document content as a string
    - 'metadata': A dictionary containing document metadata

Conversion targets:
    - Haystack: Uses 'content' attribute and '.meta' property
    - LangChain: Uses 'page_content' attribute and '.metadata' property

Test categories:
    TestDocumentConverterToHaystack: Conversion from dict to Haystack Documents.
    TestDocumentConverterToLangchain: Conversion from dict to LangChain Documents.
    TestDocumentConverterCrossFramework: Consistency checks between frameworks.
"""

from haystack import Document as HaystackDocument

from vectordb.dataloaders.converters import DocumentConverter


class TestDocumentConverterToHaystack:
    """Test suite for converting to Haystack documents.

    Tests cover:
    - Converting standardized format to Haystack documents
    - Metadata preservation
    - Content mapping
    - Empty list handling
    """

    def test_convert_to_haystack_basic(self) -> None:
        """Test basic conversion to Haystack documents."""
        items = [
            {
                "text": "Sample content",
                "metadata": {"id": "1", "source": "test"},
            }
        ]

        result = DocumentConverter.to_haystack(items)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], HaystackDocument)

    def test_convert_to_haystack_preserves_content(self) -> None:
        """Test that content is preserved in conversion."""
        items = [
            {
                "text": "Test content",
                "metadata": {"id": "1"},
            }
        ]

        result = DocumentConverter.to_haystack(items)

        assert result[0].content == "Test content"

    def test_convert_to_haystack_preserves_metadata(self) -> None:
        """Test that metadata is preserved."""
        items = [
            {
                "text": "Content",
                "metadata": {"id": "doc1", "source": "wiki", "score": 0.95},
            }
        ]

        result = DocumentConverter.to_haystack(items)

        assert result[0].meta["id"] == "doc1"
        assert result[0].meta["source"] == "wiki"
        assert result[0].meta["score"] == 0.95

    def test_convert_to_haystack_multiple_items(self) -> None:
        """Test conversion of multiple items."""
        items = [
            {"text": "Content 1", "metadata": {"id": "1"}},
            {"text": "Content 2", "metadata": {"id": "2"}},
            {"text": "Content 3", "metadata": {"id": "3"}},
        ]

        result = DocumentConverter.to_haystack(items)

        assert len(result) == 3
        assert result[0].content == "Content 1"
        assert result[2].content == "Content 3"

    def test_convert_to_haystack_empty_list(self) -> None:
        """Test conversion of empty list."""
        result = DocumentConverter.to_haystack([])

        assert result == []

    def test_convert_to_haystack_empty_metadata(self) -> None:
        """Test conversion with empty metadata."""
        items = [{"text": "Content", "metadata": {}}]

        result = DocumentConverter.to_haystack(items)

        assert result[0].meta == {}

    def test_convert_to_haystack_nested_metadata(self) -> None:
        """Test conversion with nested metadata."""
        items = [
            {
                "text": "Content",
                "metadata": {
                    "id": "1",
                    "nested": {
                        "key": "value",
                        "score": 0.9,
                    },
                },
            }
        ]

        result = DocumentConverter.to_haystack(items)

        assert "nested" in result[0].meta
        assert result[0].meta["nested"]["key"] == "value"


class TestDocumentConverterToLangchain:
    """Test suite for converting to LangChain documents.

    Tests cover:
    - Converting standardized format to LangChain documents
    - Metadata preservation
    - Content mapping to page_content
    - Empty list handling
    """

    def test_convert_to_langchain_basic(self) -> None:
        """Test basic conversion to LangChain documents."""
        items = [
            {
                "text": "Sample content",
                "metadata": {"id": "1", "source": "test"},
            }
        ]

        result = DocumentConverter.to_langchain(items)

        assert isinstance(result, list)
        assert len(result) == 1

    def test_convert_to_langchain_preserves_content(self) -> None:
        """Test that content is preserved as page_content."""
        items = [
            {
                "text": "Test content",
                "metadata": {"id": "1"},
            }
        ]

        result = DocumentConverter.to_langchain(items)

        assert result[0].page_content == "Test content"

    def test_convert_to_langchain_preserves_metadata(self) -> None:
        """Test that metadata is preserved."""
        items = [
            {
                "text": "Content",
                "metadata": {"id": "doc1", "source": "wiki", "score": 0.95},
            }
        ]

        result = DocumentConverter.to_langchain(items)

        assert result[0].metadata["id"] == "doc1"
        assert result[0].metadata["source"] == "wiki"
        assert result[0].metadata["score"] == 0.95

    def test_convert_to_langchain_multiple_items(self) -> None:
        """Test conversion of multiple items."""
        items = [
            {"text": "Content 1", "metadata": {"id": "1"}},
            {"text": "Content 2", "metadata": {"id": "2"}},
            {"text": "Content 3", "metadata": {"id": "3"}},
        ]

        result = DocumentConverter.to_langchain(items)

        assert len(result) == 3
        assert result[0].page_content == "Content 1"
        assert result[2].page_content == "Content 3"

    def test_convert_to_langchain_empty_list(self) -> None:
        """Test conversion of empty list."""
        result = DocumentConverter.to_langchain([])

        assert result == []

    def test_convert_to_langchain_empty_metadata(self) -> None:
        """Test conversion with empty metadata."""
        items = [{"text": "Content", "metadata": {}}]

        result = DocumentConverter.to_langchain(items)

        assert result[0].metadata == {}

    def test_convert_to_langchain_nested_metadata(self) -> None:
        """Test conversion with nested metadata."""
        items = [
            {
                "text": "Content",
                "metadata": {
                    "id": "1",
                    "nested": {
                        "key": "value",
                        "score": 0.9,
                    },
                },
            }
        ]

        result = DocumentConverter.to_langchain(items)

        assert "nested" in result[0].metadata
        assert result[0].metadata["nested"]["key"] == "value"


class TestDocumentConverterCrossFramework:
    """Test suite for cross-framework conversion consistency.

    Tests cover:
    - Consistency between Haystack and LangChain conversions
    - Metadata consistency across frameworks
    - Content preservation across conversions
    """

    def test_haystack_and_langchain_convert_same_content(self) -> None:
        """Test that both frameworks convert content identically."""
        items = [{"text": "Test content", "metadata": {"id": "1"}}]

        haystack_result = DocumentConverter.to_haystack(items)
        langchain_result = DocumentConverter.to_langchain(items)

        # Content should be the same (haystack uses 'content', LC uses 'page_content')
        assert haystack_result[0].content == langchain_result[0].page_content

    def test_haystack_and_langchain_preserve_metadata(self) -> None:
        """Test that metadata is preserved in both frameworks."""
        items = [{"text": "Content", "metadata": {"id": "1", "source": "test"}}]

        haystack_result = DocumentConverter.to_haystack(items)
        langchain_result = DocumentConverter.to_langchain(items)

        # Metadata should be identical (haystack uses .meta, LC uses .metadata)
        assert haystack_result[0].meta["id"] == langchain_result[0].metadata["id"]
        assert (
            haystack_result[0].meta["source"] == langchain_result[0].metadata["source"]
        )

    def test_batch_conversion_consistency(self) -> None:
        """Test that batch conversions are consistent."""
        items = [
            {"text": "Doc 1", "metadata": {"id": "1"}},
            {"text": "Doc 2", "metadata": {"id": "2"}},
        ]

        haystack_docs = DocumentConverter.to_haystack(items)
        langchain_docs = DocumentConverter.to_langchain(items)

        assert len(haystack_docs) == len(langchain_docs)
        for hs, lc in zip(haystack_docs, langchain_docs):
            assert hs.content == lc.page_content
