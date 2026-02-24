"""Tests for ID generation utilities in parent document retrieval."""

import pytest

from vectordb.haystack.parent_document_retrieval.utils.ids import (
    generate_chunk_id,
    generate_document_id,
    generate_parent_id,
)


class TestGenerateDocumentId:
    """Test suite for generate_document_id function."""

    def test_generate_document_id_content_only(self, sample_content: str) -> None:
        """Test generating document ID with content only."""
        result = generate_document_id(sample_content)

        assert isinstance(result, str)
        assert len(result) == 20

    def test_generate_document_id_with_source_id(
        self, sample_content: str, sample_source_id: str
    ) -> None:
        """Test generating document ID with content and source_id."""
        result = generate_document_id(sample_content, sample_source_id)

        assert isinstance(result, str)
        assert len(result) == 20

    def test_generate_document_id_empty_content(self) -> None:
        """Test generating document ID with empty content."""
        result = generate_document_id("")

        assert isinstance(result, str)
        assert len(result) == 20

    def test_generate_document_id_deterministic(self, sample_content: str) -> None:
        """Test that document ID generation is deterministic."""
        result1 = generate_document_id(sample_content)
        result2 = generate_document_id(sample_content)

        assert result1 == result2

    def test_generate_document_id_deterministic_with_source_id(
        self, sample_content: str, sample_source_id: str
    ) -> None:
        """Test deterministic behavior with source_id."""
        result1 = generate_document_id(sample_content, sample_source_id)
        result2 = generate_document_id(sample_content, sample_source_id)

        assert result1 == result2

    def test_generate_document_id_different_content(self) -> None:
        """Test that different content produces different IDs."""
        content1 = "This is content one."
        content2 = "This is content two."

        result1 = generate_document_id(content1)
        result2 = generate_document_id(content2)

        assert result1 != result2

    def test_generate_document_id_different_source_id(
        self, sample_content: str
    ) -> None:
        """Test that different source_id produces different IDs."""
        source_id1 = "source_1"
        source_id2 = "source_2"

        result1 = generate_document_id(sample_content, source_id1)
        result2 = generate_document_id(sample_content, source_id2)

        assert result1 != result2

    def test_generate_document_id_none_source_id(self, sample_content: str) -> None:
        """Test generating document ID with None source_id."""
        result = generate_document_id(sample_content, None)

        assert isinstance(result, str)
        assert len(result) == 20

    def test_generate_document_id_empty_string_source_id(
        self, sample_content: str
    ) -> None:
        """Test generating document ID with empty string source_id."""
        result = generate_document_id(sample_content, "")

        assert isinstance(result, str)
        assert len(result) == 20

    def test_generate_document_id_special_characters(self) -> None:
        """Test generating document ID with special characters."""
        content = "Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"
        source_id = "special_!@#$%^&*()"

        result = generate_document_id(content, source_id)

        assert isinstance(result, str)
        assert len(result) == 20

    def test_generate_document_id_unicode_characters(self) -> None:
        """Test generating document ID with Unicode characters."""
        content = "Unicode: ñáéíóú 中文 العربية русский"
        source_id = "unicode_ñáéíóú"

        result = generate_document_id(content, source_id)

        assert isinstance(result, str)
        assert len(result) == 20

    def test_generate_document_id_very_long_content(self) -> None:
        """Test generating document ID with very long content."""
        long_content = "A" * 10000  # 10KB of content

        result = generate_document_id(long_content)

        assert isinstance(result, str)
        assert len(result) == 20

    def test_generate_document_id_whitespace_only(self) -> None:
        """Test generating document ID with whitespace-only content."""
        whitespace_content = "   \t\n\r   "

        result = generate_document_id(whitespace_content)

        assert isinstance(result, str)
        assert len(result) == 20

    def test_generate_document_id_source_id_affects_hash(
        self, sample_content: str
    ) -> None:
        """Test that source_id affects the hash calculation."""
        result_no_source = generate_document_id(sample_content)
        result_with_source = generate_document_id(sample_content, "test_source")

        assert result_no_source != result_with_source

    @pytest.mark.parametrize(
        "content",
        [
            "short",
            "medium length content with some words",
            "This is a longer piece of content that contains multiple sentences and various punctuation marks.",
            "A" * 100,  # 100 characters
            "A" * 1000,  # 1000 characters
        ],
    )
    def test_generate_document_id_various_content_lengths(self, content: str) -> None:
        """Test generate_document_id with various content lengths."""
        result = generate_document_id(content)

        assert isinstance(result, str)
        assert len(result) == 20


class TestGenerateChunkId:
    """Test suite for generate_chunk_id function."""

    def test_generate_chunk_id_basic(self) -> None:
        """Test basic chunk ID generation."""
        result = generate_chunk_id("parent_123", 0, 1)

        assert isinstance(result, str)
        assert len(result) == 16

    def test_generate_chunk_id_deterministic(self) -> None:
        """Test that chunk ID generation is deterministic."""
        result1 = generate_chunk_id("parent_123", 0, 1)
        result2 = generate_chunk_id("parent_123", 0, 1)

        assert result1 == result2

    def test_generate_chunk_id_different_parent_id(self) -> None:
        """Test that different parent_id produces different chunk IDs."""
        result1 = generate_chunk_id("parent_1", 0, 1)
        result2 = generate_chunk_id("parent_2", 0, 1)

        assert result1 != result2

    def test_generate_chunk_id_different_chunk_idx(self) -> None:
        """Test that different chunk_idx produces different chunk IDs."""
        result1 = generate_chunk_id("parent_123", 0, 1)
        result2 = generate_chunk_id("parent_123", 1, 1)

        assert result1 != result2

    def test_generate_chunk_id_different_level(self) -> None:
        """Test that different level produces different chunk IDs."""
        result1 = generate_chunk_id("parent_123", 0, 1)
        result2 = generate_chunk_id("parent_123", 0, 2)

        assert result1 != result2

    def test_generate_chunk_id_empty_parent_id(self) -> None:
        """Test generating chunk ID with empty parent_id."""
        result = generate_chunk_id("", 0, 1)

        assert isinstance(result, str)
        assert len(result) == 16

    def test_generate_chunk_id_special_characters_parent_id(self) -> None:
        """Test generating chunk ID with special characters in parent_id."""
        special_parent_id = "parent_123!@#$%^&*()_+-=[]{}|;':\",./<>?"

        result = generate_chunk_id(special_parent_id, 0, 1)

        assert isinstance(result, str)
        assert len(result) == 16

    def test_generate_chunk_id_negative_chunk_idx(self) -> None:
        """Test generating chunk ID with negative chunk_idx."""
        result = generate_chunk_id("parent_123", -1, 1)

        assert isinstance(result, str)
        assert len(result) == 16

    def test_generate_chunk_id_large_chunk_idx(self) -> None:
        """Test generating chunk ID with large chunk_idx."""
        result = generate_chunk_id("parent_123", 999999, 1)

        assert isinstance(result, str)
        assert len(result) == 16

    def test_generate_chunk_id_negative_level(self) -> None:
        """Test generating chunk ID with negative level."""
        result = generate_chunk_id("parent_123", 0, -1)

        assert isinstance(result, str)
        assert len(result) == 16

    def test_generate_chunk_id_zero_values(self) -> None:
        """Test generating chunk ID with zero values."""
        result = generate_chunk_id("parent_123", 0, 0)

        assert isinstance(result, str)
        assert len(result) == 16

    @pytest.mark.parametrize(
        "parent_id",
        [
            "parent_1",
            "parent_abc",
            "123",
            "parent-with-dash",
            "parent_with_underscore",
            "parent.with.dots",
            "",
        ],
    )
    def test_generate_chunk_id_various_parent_ids(self, parent_id: str) -> None:
        """Test generate_chunk_id with various parent_id formats."""
        result = generate_chunk_id(parent_id, 0, 1)

        assert isinstance(result, str)
        assert len(result) == 16

    @pytest.mark.parametrize("chunk_idx", [0, 1, 5, 10, 100, 999])
    def test_generate_chunk_id_various_chunk_indices(self, chunk_idx: int) -> None:
        """Test generate_chunk_id with various chunk indices."""
        result = generate_chunk_id("parent_123", chunk_idx, 1)

        assert isinstance(result, str)
        assert len(result) == 16

    @pytest.mark.parametrize("level", [0, 1, 2, 3, 5, 10])
    def test_generate_chunk_id_various_levels(self, level: int) -> None:
        """Test generate_chunk_id with various level values."""
        result = generate_chunk_id("parent_123", 0, level)

        assert isinstance(result, str)
        assert len(result) == 16


class TestGenerateParentId:
    """Test suite for generate_parent_id function."""

    def test_generate_parent_id_basic(self, sample_content: str) -> None:
        """Test basic parent ID generation."""
        result = generate_parent_id(sample_content, 0)

        assert isinstance(result, str)
        assert result.startswith("parent_0_")
        assert len(result) > len("parent_0_")  # Should have hash part

    def test_generate_parent_id_deterministic(self, sample_content: str) -> None:
        """Test that parent ID generation is deterministic."""
        result1 = generate_parent_id(sample_content, 0)
        result2 = generate_parent_id(sample_content, 0)

        assert result1 == result2

    def test_generate_parent_id_different_content(self) -> None:
        """Test that different content produces different parent IDs."""
        content1 = "Content one"
        content2 = "Content two"

        result1 = generate_parent_id(content1, 0)
        result2 = generate_parent_id(content2, 0)

        assert result1 != result2

    def test_generate_parent_id_different_doc_idx(self, sample_content: str) -> None:
        """Test that different doc_idx produces different parent IDs."""
        result1 = generate_parent_id(sample_content, 0)
        result2 = generate_parent_id(sample_content, 1)

        assert result1 != result2

    def test_generate_parent_id_format_structure(self, sample_content: str) -> None:
        """Test that parent ID follows expected format."""
        doc_idx = 5
        result = generate_parent_id(sample_content, doc_idx)

        assert result.startswith(f"parent_{doc_idx}_")

        # Extract hash part
        hash_part = result[len(f"parent_{doc_idx}_") :]
        assert len(hash_part) == 16
        assert all(c in "0123456789abcdef" for c in hash_part)

    def test_generate_parent_id_empty_content(self) -> None:
        """Test generating parent ID with empty content."""
        result = generate_parent_id("", 0)

        assert result.startswith("parent_0_")
        hash_part = result[9:]  # Remove "parent_0_"
        assert len(hash_part) == 16

    def test_generate_parent_id_negative_doc_idx(self, sample_content: str) -> None:
        """Test generating parent ID with negative doc_idx."""
        doc_idx = -1
        result = generate_parent_id(sample_content, doc_idx)

        assert result.startswith(f"parent_{doc_idx}_")
        hash_part = result[len(f"parent_{doc_idx}_") :]
        assert len(hash_part) == 16

    def test_generate_parent_id_large_doc_idx(self, sample_content: str) -> None:
        """Test generating parent ID with large doc_idx."""
        doc_idx = 999999
        result = generate_parent_id(sample_content, doc_idx)

        assert result.startswith(f"parent_{doc_idx}_")
        hash_part = result[len(f"parent_{doc_idx}_") :]
        assert len(hash_part) == 16

    def test_generate_parent_id_special_characters(self) -> None:
        """Test generating parent ID with special characters in content."""
        content = "Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"
        result = generate_parent_id(content, 0)

        assert result.startswith("parent_0_")
        hash_part = result[9:]
        assert len(hash_part) == 16

    def test_generate_parent_id_unicode_characters(self) -> None:
        """Test generating parent ID with Unicode characters."""
        content = "Unicode: ñáéíóú 中文 العربية русский"
        result = generate_parent_id(content, 0)

        assert result.startswith("parent_0_")
        hash_part = result[9:]
        assert len(hash_part) == 16

    def test_generate_parent_id_very_long_content(self) -> None:
        """Test generating parent ID with very long content."""
        long_content = "A" * 10000  # 10KB of content
        result = generate_parent_id(long_content, 0)

        assert result.startswith("parent_0_")
        hash_part = result[9:]
        assert len(hash_part) == 16

    def test_generate_parent_id_whitespace_only(self) -> None:
        """Test generating parent ID with whitespace-only content."""
        whitespace_content = "   \t\n\r   "
        result = generate_parent_id(whitespace_content, 0)

        assert result.startswith("parent_0_")
        hash_part = result[9:]
        assert len(hash_part) == 16

    @pytest.mark.parametrize("doc_idx", [0, 1, 5, 10, 100, 999])
    def test_generate_parent_id_various_doc_indices(
        self, doc_idx: int, sample_content: str
    ) -> None:
        """Test generate_parent_id with various doc_idx values."""
        result = generate_parent_id(sample_content, doc_idx)

        assert result.startswith(f"parent_{doc_idx}_")
        hash_part = result[len(f"parent_{doc_idx}_") :]
        assert len(hash_part) == 16

    @pytest.mark.parametrize(
        "content",
        [
            "short",
            "medium length content",
            "This is a longer piece of content that contains multiple sentences.",
            "A" * 100,  # 100 characters
            "A" * 1000,  # 1000 characters
        ],
    )
    def test_generate_parent_id_various_content_lengths(self, content: str) -> None:
        """Test generate_parent_id with various content lengths."""
        result = generate_parent_id(content, 0)

        assert result.startswith("parent_0_")
        hash_part = result[9:]
        assert len(hash_part) == 16

    def test_generate_parent_id_hash_consistency(self, sample_content: str) -> None:
        """Test that hash part is consistent for same content regardless of doc_idx."""
        result1 = generate_parent_id(sample_content, 0)
        result2 = generate_parent_id(sample_content, 1)

        # Extract hash parts - find the underscore after doc_idx
        hash1 = result1.split("_", 2)[-1]  # Split by _, get last part
        hash2 = result2.split("_", 2)[-1]  # Split by _, get last part

        # Hash should be same for same content
        assert hash1 == hash2

    def test_generate_parent_id_different_content_different_hash(self) -> None:
        """Test that different content produces different hash."""
        content1 = "Content one"
        content2 = "Content two"

        result1 = generate_parent_id(content1, 0)
        result2 = generate_parent_id(content2, 0)

        hash1 = result1[9:]
        hash2 = result2[9:]

        assert hash1 != hash2
