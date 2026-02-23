"""Tests for sparse embedding utilities.

This module tests sparse embedding normalization and format conversion utilities
used for hybrid search across multiple vector database backends.

Tested functions:
    normalize_sparse: Convert various sparse formats to SparseEmbedding.
    to_milvus_sparse: Convert SparseEmbedding to Milvus dict format.
    to_pinecone_sparse: Convert SparseEmbedding to Pinecone dict format.
    to_qdrant_sparse: Convert SparseEmbedding to Qdrant SparseVector.
    get_doc_sparse_embedding: Extract sparse embedding from document.

Test coverage includes:
    - None input handling and passthrough behavior
    - Format conversion between Milvus ({int: float}) and Pinecone dict formats
    - SparseEmbedding object creation and property verification
    - Unsupported type error handling
    - Document sparse embedding extraction from standard and meta locations
    - Custom fallback metadata key support
    - Round-trip conversion consistency across all formats
"""

from dataclasses import dataclass, field
from typing import Any

import pytest
from haystack import Document
from haystack.dataclasses import SparseEmbedding

from vectordb.utils.sparse import (
    get_doc_sparse_embedding,
    normalize_sparse,
    to_milvus_sparse,
    to_pinecone_sparse,
    to_qdrant_sparse,
)


class TestNormalizeSparse:
    """Test suite for normalize_sparse function.

    Tests cover:
    - None input handling
    - SparseEmbedding passthrough
    - Milvus format conversion
    - Pinecone format conversion
    - Unsupported type handling
    """

    def test_none_input_returns_none(self) -> None:
        """Test that None input returns None."""
        result = normalize_sparse(None)
        assert result is None

    def test_sparse_embedding_passthrough(self) -> None:
        """Test that SparseEmbedding is passed through unchanged."""
        sparse = SparseEmbedding(indices=[1, 5, 10], values=[0.5, 0.8, 0.3])
        result = normalize_sparse(sparse)

        assert result is sparse
        assert result.indices == [1, 5, 10]
        assert result.values == [0.5, 0.8, 0.3]

    def test_milvus_format_conversion(self) -> None:
        """Test conversion from Milvus format {int: float}."""
        milvus_sparse = {1: 0.5, 5: 0.8, 10: 0.3}
        result = normalize_sparse(milvus_sparse)

        assert isinstance(result, SparseEmbedding)
        assert result.indices == [1, 5, 10]
        assert result.values == [0.5, 0.8, 0.3]

    def test_milvus_format_empty_dict(self) -> None:
        """Test conversion from empty Milvus format dict."""
        result = normalize_sparse({})

        assert isinstance(result, SparseEmbedding)
        assert result.indices == []
        assert result.values == []

    def test_milvus_format_single_entry(self) -> None:
        """Test conversion from single-entry Milvus format."""
        result = normalize_sparse({42: 0.99})

        assert isinstance(result, SparseEmbedding)
        assert result.indices == [42]
        assert result.values == [0.99]

    def test_pinecone_format_conversion(self) -> None:
        """Test conversion from Pinecone format {"indices": [...], "values": [...]}."""
        pinecone_sparse = {"indices": [1, 5, 10], "values": [0.5, 0.8, 0.3]}
        result = normalize_sparse(pinecone_sparse)

        assert isinstance(result, SparseEmbedding)
        assert result.indices == [1, 5, 10]
        assert result.values == [0.5, 0.8, 0.3]

    def test_pinecone_format_empty_lists(self) -> None:
        """Test conversion from Pinecone format with empty lists."""
        pinecone_sparse = {"indices": [], "values": []}
        result = normalize_sparse(pinecone_sparse)

        assert isinstance(result, SparseEmbedding)
        assert result.indices == []
        assert result.values == []

    def test_pinecone_format_with_tuples(self) -> None:
        """Test conversion from Pinecone format with tuple values."""
        pinecone_sparse = {"indices": (1, 2, 3), "values": (0.1, 0.2, 0.3)}
        result = normalize_sparse(pinecone_sparse)

        assert isinstance(result, SparseEmbedding)
        assert result.indices == [1, 2, 3]
        assert result.values == [0.1, 0.2, 0.3]

    def test_unsupported_type_raises_typeerror(self) -> None:
        """Test that unsupported types raise TypeError."""
        with pytest.raises(TypeError, match="Unsupported sparse embedding format"):
            normalize_sparse("invalid")  # type: ignore[arg-type]

    def test_unsupported_list_raises_typeerror(self) -> None:
        """Test that list input raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported sparse embedding format"):
            normalize_sparse([1, 2, 3])  # type: ignore[arg-type]

    def test_unsupported_int_raises_typeerror(self) -> None:
        """Test that int input raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported sparse embedding format"):
            normalize_sparse(42)  # type: ignore[arg-type]


class TestToMilvusSparse:
    """Test suite for to_milvus_sparse function.

    Tests cover:
    - Conversion from SparseEmbedding to {index: value} dict
    - Various indices and values
    """

    def test_basic_conversion(self) -> None:
        """Test basic conversion to Milvus format."""
        sparse = SparseEmbedding(indices=[1, 5, 10], values=[0.5, 0.8, 0.3])
        result = to_milvus_sparse(sparse)

        assert result == {1: 0.5, 5: 0.8, 10: 0.3}

    def test_empty_sparse_embedding(self) -> None:
        """Test conversion of empty SparseEmbedding."""
        sparse = SparseEmbedding(indices=[], values=[])
        result = to_milvus_sparse(sparse)

        assert result == {}

    def test_single_entry(self) -> None:
        """Test conversion with single index-value pair."""
        sparse = SparseEmbedding(indices=[42], values=[0.99])
        result = to_milvus_sparse(sparse)

        assert result == {42: 0.99}

    def test_large_indices(self) -> None:
        """Test conversion with large index values."""
        sparse = SparseEmbedding(indices=[10000, 50000, 100000], values=[0.1, 0.2, 0.3])
        result = to_milvus_sparse(sparse)

        assert result == {10000: 0.1, 50000: 0.2, 100000: 0.3}

    def test_negative_values(self) -> None:
        """Test conversion with negative values."""
        sparse = SparseEmbedding(indices=[1, 2, 3], values=[-0.5, 0.0, 0.5])
        result = to_milvus_sparse(sparse)

        assert result == {1: -0.5, 2: 0.0, 3: 0.5}


class TestToPineconeSparse:
    """Test suite for to_pinecone_sparse function.

    Tests cover:
    - Conversion to {"indices": [...], "values": [...]} format
    - Various values
    """

    def test_basic_conversion(self) -> None:
        """Test basic conversion to Pinecone format."""
        sparse = SparseEmbedding(indices=[1, 5, 10], values=[0.5, 0.8, 0.3])
        result = to_pinecone_sparse(sparse)

        assert result == {"indices": [1, 5, 10], "values": [0.5, 0.8, 0.3]}

    def test_empty_sparse_embedding(self) -> None:
        """Test conversion of empty SparseEmbedding."""
        sparse = SparseEmbedding(indices=[], values=[])
        result = to_pinecone_sparse(sparse)

        assert result == {"indices": [], "values": []}

    def test_single_entry(self) -> None:
        """Test conversion with single index-value pair."""
        sparse = SparseEmbedding(indices=[42], values=[0.99])
        result = to_pinecone_sparse(sparse)

        assert result == {"indices": [42], "values": [0.99]}

    def test_returns_lists(self) -> None:
        """Test that result contains proper lists."""
        sparse = SparseEmbedding(indices=[1, 2], values=[0.1, 0.2])
        result = to_pinecone_sparse(sparse)

        assert isinstance(result["indices"], list)
        assert isinstance(result["values"], list)

    def test_preserves_order(self) -> None:
        """Test that index-value order is preserved."""
        sparse = SparseEmbedding(indices=[100, 1, 50], values=[0.9, 0.1, 0.5])
        result = to_pinecone_sparse(sparse)

        assert result["indices"] == [100, 1, 50]
        assert result["values"] == [0.9, 0.1, 0.5]


class TestToQdrantSparse:
    """Test suite for to_qdrant_sparse function.

    Tests cover:
    - Conversion to Qdrant SparseVector object
    - Verification of indices and values
    """

    def test_basic_conversion(self) -> None:
        """Test basic conversion to Qdrant SparseVector."""
        sparse = SparseEmbedding(indices=[1, 5, 10], values=[0.5, 0.8, 0.3])
        result = to_qdrant_sparse(sparse)

        # Import to verify type
        from qdrant_client.http.models import SparseVector

        assert isinstance(result, SparseVector)
        assert result.indices == [1, 5, 10]
        assert result.values == [0.5, 0.8, 0.3]

    def test_empty_sparse_embedding(self) -> None:
        """Test conversion of empty SparseEmbedding."""
        from qdrant_client.http.models import SparseVector

        sparse = SparseEmbedding(indices=[], values=[])
        result = to_qdrant_sparse(sparse)

        assert isinstance(result, SparseVector)
        assert result.indices == []
        assert result.values == []

    def test_single_entry(self) -> None:
        """Test conversion with single index-value pair."""
        from qdrant_client.http.models import SparseVector

        sparse = SparseEmbedding(indices=[42], values=[0.99])
        result = to_qdrant_sparse(sparse)

        assert isinstance(result, SparseVector)
        assert result.indices == [42]
        assert result.values == [0.99]

    def test_large_sparse_vector(self) -> None:
        """Test conversion with many entries."""
        from qdrant_client.http.models import SparseVector

        indices = list(range(0, 1000, 10))
        values = [i * 0.001 for i in indices]
        sparse = SparseEmbedding(indices=indices, values=values)
        result = to_qdrant_sparse(sparse)

        assert isinstance(result, SparseVector)
        assert len(result.indices) == 100
        assert len(result.values) == 100


class TestGetDocSparseEmbedding:
    """Test suite for get_doc_sparse_embedding function.

    Tests cover:
    - Extracting from doc.sparse_embedding (standard location)
    - Fallback to doc.meta[fallback_meta_key]
    - None when neither location has sparse embedding
    - Custom fallback_meta_key
    - Document without meta attribute
    """

    def test_extract_from_standard_location(self) -> None:
        """Test extracting sparse embedding from doc.sparse_embedding."""
        sparse = SparseEmbedding(indices=[1, 2, 3], values=[0.1, 0.2, 0.3])
        doc = Document(content="Test", sparse_embedding=sparse)

        result = get_doc_sparse_embedding(doc)

        assert result is sparse
        assert result.indices == [1, 2, 3]
        assert result.values == [0.1, 0.2, 0.3]

    def test_fallback_to_meta(self) -> None:
        """Test fallback to doc.meta[fallback_meta_key] for backward compatibility."""
        sparse = SparseEmbedding(indices=[4, 5, 6], values=[0.4, 0.5, 0.6])
        doc = Document(content="Test", meta={"sparse_embedding": sparse})

        result = get_doc_sparse_embedding(doc)

        assert result is sparse

    def test_fallback_meta_with_milvus_format(self) -> None:
        """Test fallback meta with Milvus format dict gets normalized."""
        doc = Document(content="Test", meta={"sparse_embedding": {1: 0.5, 2: 0.8}})

        result = get_doc_sparse_embedding(doc)

        assert isinstance(result, SparseEmbedding)
        assert result.indices == [1, 2]
        assert result.values == [0.5, 0.8]

    def test_fallback_meta_with_pinecone_format(self) -> None:
        """Test fallback meta with Pinecone format dict gets normalized."""
        doc = Document(
            content="Test",
            meta={"sparse_embedding": {"indices": [1, 2], "values": [0.5, 0.8]}},
        )

        result = get_doc_sparse_embedding(doc)

        assert isinstance(result, SparseEmbedding)
        assert result.indices == [1, 2]
        assert result.values == [0.5, 0.8]

    def test_returns_none_when_no_sparse_embedding(self) -> None:
        """Test returns None when neither location has sparse embedding."""
        doc = Document(content="Test")

        result = get_doc_sparse_embedding(doc)

        assert result is None

    def test_returns_none_with_empty_meta(self) -> None:
        """Test returns None when meta exists but has no sparse_embedding."""
        doc = Document(content="Test", meta={"other_key": "value"})

        result = get_doc_sparse_embedding(doc)

        assert result is None

    def test_custom_fallback_meta_key(self) -> None:
        """Test with custom fallback_meta_key."""
        sparse = SparseEmbedding(indices=[7, 8, 9], values=[0.7, 0.8, 0.9])
        doc = Document(content="Test", meta={"custom_sparse_key": sparse})

        result = get_doc_sparse_embedding(doc, fallback_meta_key="custom_sparse_key")

        assert result is sparse

    def test_standard_location_takes_precedence(self) -> None:
        """Test that doc.sparse_embedding takes precedence over meta."""
        sparse_standard = SparseEmbedding(indices=[1, 2], values=[0.1, 0.2])
        sparse_meta = SparseEmbedding(indices=[3, 4], values=[0.3, 0.4])
        doc = Document(
            content="Test",
            sparse_embedding=sparse_standard,
            meta={"sparse_embedding": sparse_meta},
        )

        result = get_doc_sparse_embedding(doc)

        assert result is sparse_standard

    def test_doc_without_meta_attribute(self) -> None:
        """Test handling document-like object without meta attribute."""

        @dataclass
        class NoMetaDoc:
            content: str
            sparse_embedding: SparseEmbedding | None = None

        sparse = SparseEmbedding(indices=[1, 2], values=[0.1, 0.2])
        doc = NoMetaDoc(content="Test", sparse_embedding=sparse)

        result = get_doc_sparse_embedding(doc)

        assert result is sparse

    def test_doc_without_meta_or_sparse_returns_none(self) -> None:
        """Test document without meta or sparse_embedding returns None."""

        @dataclass
        class MinimalDoc:
            content: str

        doc = MinimalDoc(content="Test")

        result = get_doc_sparse_embedding(doc)

        assert result is None

    def test_doc_with_none_sparse_falls_back_to_meta(self) -> None:
        """Test that None sparse_embedding falls back to meta."""
        sparse_meta = SparseEmbedding(indices=[5, 6], values=[0.5, 0.6])
        doc = Document(
            content="Test",
            sparse_embedding=None,
            meta={"sparse_embedding": sparse_meta},
        )

        result = get_doc_sparse_embedding(doc)

        assert result is sparse_meta

    def test_doc_without_sparse_embedding_attribute(self) -> None:
        """Test document without sparse_embedding attribute uses meta fallback."""

        @dataclass
        class NoSparseDoc:
            content: str
            meta: dict[str, Any] = field(default_factory=dict)

        sparse = SparseEmbedding(indices=[1, 2], values=[0.1, 0.2])
        doc = NoSparseDoc(content="Test", meta={"sparse_embedding": sparse})

        result = get_doc_sparse_embedding(doc)

        assert result is sparse


class TestRoundTrip:
    """Test suite for round-trip conversions.

    Tests cover:
    - Normalize -> to_format -> normalize consistency
    """

    def test_milvus_roundtrip(self) -> None:
        """Test Milvus format round-trip conversion."""
        original = {1: 0.5, 5: 0.8, 10: 0.3}

        sparse = normalize_sparse(original)
        assert sparse is not None
        back_to_milvus = to_milvus_sparse(sparse)

        assert back_to_milvus == original

    def test_pinecone_roundtrip(self) -> None:
        """Test Pinecone format round-trip conversion."""
        original = {"indices": [1, 5, 10], "values": [0.5, 0.8, 0.3]}

        sparse = normalize_sparse(original)
        assert sparse is not None
        back_to_pinecone = to_pinecone_sparse(sparse)

        assert back_to_pinecone == original

    def test_sparse_embedding_to_all_formats(self) -> None:
        """Test SparseEmbedding converts to all formats correctly."""
        sparse = SparseEmbedding(indices=[1, 2, 3], values=[0.1, 0.2, 0.3])

        milvus = to_milvus_sparse(sparse)
        pinecone = to_pinecone_sparse(sparse)
        qdrant = to_qdrant_sparse(sparse)

        assert milvus == {1: 0.1, 2: 0.2, 3: 0.3}
        assert pinecone == {"indices": [1, 2, 3], "values": [0.1, 0.2, 0.3]}
        assert qdrant.indices == [1, 2, 3]
        assert qdrant.values == [0.1, 0.2, 0.3]
