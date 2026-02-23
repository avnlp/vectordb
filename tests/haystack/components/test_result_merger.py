"""Comprehensive tests for Haystack result merger components.

This module tests the ResultMerger which combines results from multiple
retrieval or reasoning sources into unified output.
"""

import hashlib

import pytest
from haystack import Document

from vectordb.haystack.components.result_merger import ResultMerger


class TestResultMerger:
    """Test suite for ResultMerger component.

    Tests cover:
    - Stable document ID generation
    - RRF fusion with various inputs
    - N-way RRF fusion
    - Weighted fusion
    - Deduplication
    - Configuration validation
    - Edge cases and error handling
    """

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for merging."""
        return [
            Document(content="Document 1", meta={"source": "source1"}),
            Document(content="Document 2", meta={"source": "source2"}),
            Document(content="Document 3", meta={"source": "source3"}),
        ]

    @pytest.fixture
    def duplicate_documents(self):
        """Create documents with duplicate content."""
        return [
            Document(content="Duplicate content", meta={"source": "source1"}),
            Document(content="Duplicate content", meta={"source": "source2"}),
            Document(content="Unique content", meta={"source": "source3"}),
        ]

    def test_stable_doc_id_with_doc_id_meta(self, sample_documents):
        """Test stable document ID generation with explicit doc_id in meta."""
        doc = sample_documents[0]
        doc.meta["doc_id"] = "custom-doc-id-123"

        doc_id = ResultMerger.stable_doc_id(doc)

        assert doc_id == "custom-doc-id-123"

    def test_stable_doc_id_with_doc_id_field(self):
        """Test stable document ID generation with doc.id field."""
        doc = Document(content="Test content")
        doc.id = "doc-field-id-456"

        doc_id = ResultMerger.stable_doc_id(doc)

        assert doc_id == "doc-field-id-456"

    def test_stable_doc_id_from_content(self):
        """Test stable document ID generation from content hash."""
        # Create document without id to test content-based hashing
        doc = Document.__new__(Document)
        doc.id = None
        doc.content = "Test content for hashing"
        doc.meta = {}

        doc_id = ResultMerger.stable_doc_id(doc)

        # Should be SHA1 hash (the implementation uses SHA1)
        assert isinstance(doc_id, str)
        assert len(doc_id) == 40  # SHA1 produces 40-char hex string

        # Verify it's actually a SHA1 hash of normalized content
        expected_hash = hashlib.sha1(
            "test content for hashing".encode(),
            usedforsecurity=False,
        ).hexdigest()
        assert doc_id == expected_hash

    def test_stable_doc_id_consistency(self):
        """Test that stable doc ID is consistent across calls."""
        doc = Document(content="Consistent content")

        id1 = ResultMerger.stable_doc_id(doc)
        id2 = ResultMerger.stable_doc_id(doc)
        id3 = ResultMerger.stable_doc_id(doc)

        assert id1 == id2 == id3

    def test_stable_doc_id_case_insensitive(self):
        """Test that stable doc ID is case-insensitive for content."""
        # Create documents without id to test content-based hashing
        doc1 = Document.__new__(Document)
        doc1.id = None
        doc1.content = "Test Content"
        doc1.meta = {}

        doc2 = Document.__new__(Document)
        doc2.id = None
        doc2.content = "test content"
        doc2.meta = {}

        id1 = ResultMerger.stable_doc_id(doc1)
        id2 = ResultMerger.stable_doc_id(doc2)

        assert id1 == id2

    def test_stable_doc_id_whitespace_handling(self):
        """Test that stable doc ID handles whitespace consistently."""
        # Create documents without id to test content-based hashing
        doc1 = Document.__new__(Document)
        doc1.id = None
        doc1.content = "  Test content  "
        doc1.meta = {}

        doc2 = Document.__new__(Document)
        doc2.id = None
        doc2.content = "Test content"
        doc2.meta = {}

        id1 = ResultMerger.stable_doc_id(doc1)
        id2 = ResultMerger.stable_doc_id(doc2)

        assert id1 == id2

    def test_stable_doc_id_empty_content(self):
        """Test stable doc ID with empty content."""
        # Create document without id to test content-based hashing
        doc = Document.__new__(Document)
        doc.id = None
        doc.content = ""
        doc.meta = {}

        doc_id = ResultMerger.stable_doc_id(doc)

        assert isinstance(doc_id, str)
        assert len(doc_id) == 40  # SHA1 produces 40-char hex string

    def test_stable_doc_id_none_content(self):
        """Test stable doc ID with None content."""
        doc = Document(content=None)

        doc_id = ResultMerger.stable_doc_id(doc)

        # Should handle None gracefully
        assert isinstance(doc_id, str)

    def test_rrf_fusion_basic(self, sample_documents):
        """Test basic RRF fusion with dense and sparse documents."""
        dense_docs = [sample_documents[0], sample_documents[1]]
        sparse_docs = [sample_documents[2]]

        # Use top_k to ensure all documents are returned
        merged = ResultMerger.rrf_fusion(dense_docs, sparse_docs, top_k=3)

        assert isinstance(merged, list)
        # Documents have unique IDs, so all 3 should be returned
        assert len(merged) == 3
        assert all(isinstance(doc, Document) for doc in merged)

    def test_rrf_fusion_empty_sources(self):
        """Test RRF fusion with empty sources."""
        merged = ResultMerger.rrf_fusion([], [])

        assert isinstance(merged, list)
        assert len(merged) == 0

    def test_rrf_fusion_only_dense(self, sample_documents):
        """Test RRF fusion with only dense documents."""
        dense_docs = sample_documents

        merged = ResultMerger.rrf_fusion(dense_docs, [])

        assert len(merged) == 3
        # Should maintain order from dense docs
        assert merged[0].content == "Document 1"

    def test_rrf_fusion_only_sparse(self, sample_documents):
        """Test RRF fusion with only sparse documents."""
        sparse_docs = sample_documents

        merged = ResultMerger.rrf_fusion([], sparse_docs)

        assert len(merged) == 3

    def test_rrf_fusion_with_duplicates(self, duplicate_documents):
        """Test RRF fusion deduplicates documents."""
        dense_docs = [duplicate_documents[0], duplicate_documents[2]]
        sparse_docs = [duplicate_documents[1]]  # Duplicate of dense_docs[0]

        merged = ResultMerger.rrf_fusion(dense_docs, sparse_docs)

        # Should have 2 unique documents
        assert len(merged) == 2

    def test_rrf_fusion_top_k_limit(self, sample_documents):
        """Test RRF fusion with top_k limit."""
        dense_docs = sample_documents
        sparse_docs = sample_documents

        merged = ResultMerger.rrf_fusion(dense_docs, sparse_docs, top_k=2)

        assert len(merged) == 2

    def test_rrf_fusion_top_k_zero(self, sample_documents):
        """Test RRF fusion with top_k=0."""
        dense_docs = sample_documents
        sparse_docs = sample_documents

        merged = ResultMerger.rrf_fusion(dense_docs, sparse_docs, top_k=0)

        assert len(merged) == 0

    def test_rrf_fusion_custom_k(self, sample_documents):
        """Test RRF fusion with custom k parameter."""
        dense_docs = [sample_documents[0]]
        sparse_docs = [sample_documents[1]]

        # Different k values affect ranking
        # Use explicit top_k since default is max(len(dense), len(sparse)) = 1
        merged_k10 = ResultMerger.rrf_fusion(dense_docs, sparse_docs, k=10, top_k=2)
        merged_k100 = ResultMerger.rrf_fusion(dense_docs, sparse_docs, k=100, top_k=2)

        assert len(merged_k10) == 2
        assert len(merged_k100) == 2

    def test_rrf_fusion_many_basic(self, sample_documents):
        """Test N-way RRF fusion."""
        ranked_lists = [
            [sample_documents[0], sample_documents[1]],
            [sample_documents[1], sample_documents[2]],
            [sample_documents[0], sample_documents[2]],
        ]

        # Use explicit top_k to get all 3 unique documents
        merged = ResultMerger.rrf_fusion_many(ranked_lists, top_k=3)

        assert isinstance(merged, list)
        assert len(merged) == 3

    def test_rrf_fusion_many_empty(self):
        """Test N-way RRF fusion with empty input."""
        merged = ResultMerger.rrf_fusion_many([])

        assert isinstance(merged, list)
        assert len(merged) == 0

    def test_rrf_fusion_many_single_list(self, sample_documents):
        """Test N-way RRF fusion with single list."""
        ranked_lists = [sample_documents]

        merged = ResultMerger.rrf_fusion_many(ranked_lists)

        assert len(merged) == 3

    def test_rrf_fusion_many_with_top_k(self, sample_documents):
        """Test N-way RRF fusion with top_k limit."""
        ranked_lists = [
            sample_documents,
            list(reversed(sample_documents)),
        ]

        merged = ResultMerger.rrf_fusion_many(ranked_lists, top_k=2)

        assert len(merged) == 2

    def test_weighted_fusion_basic(self, sample_documents):
        """Test basic weighted fusion."""
        # Add scores to documents
        for i, doc in enumerate(sample_documents):
            doc.score = 0.9 - (i * 0.1)

        dense_docs = [sample_documents[0], sample_documents[1]]
        sparse_docs = [sample_documents[2]]

        # Use explicit top_k to get all 3 unique documents
        merged = ResultMerger.weighted_fusion(
            dense_docs, sparse_docs, dense_weight=0.7, sparse_weight=0.3, top_k=3
        )

        assert isinstance(merged, list)
        assert len(merged) == 3
        assert all(isinstance(doc, Document) for doc in merged)

    def test_weighted_fusion_empty_sources(self):
        """Test weighted fusion with empty sources."""
        merged = ResultMerger.weighted_fusion([], [])

        assert isinstance(merged, list)
        assert len(merged) == 0

    def test_weighted_fusion_only_dense(self, sample_documents):
        """Test weighted fusion with only dense documents."""
        for doc in sample_documents:
            doc.score = 0.8

        merged = ResultMerger.weighted_fusion(
            sample_documents, [], dense_weight=0.7, sparse_weight=0.3
        )

        assert len(merged) == 3

    def test_weighted_fusion_only_sparse(self, sample_documents):
        """Test weighted fusion with only sparse documents."""
        for doc in sample_documents:
            doc.score = 0.8

        merged = ResultMerger.weighted_fusion(
            [], sample_documents, dense_weight=0.7, sparse_weight=0.3
        )

        assert len(merged) == 3

    def test_weighted_fusion_weight_normalization(self, sample_documents):
        """Test weighted fusion normalizes weights that don't sum to 1."""
        # Create fresh documents with unique content to avoid ID collisions
        doc1 = Document(content="Unique doc 1", score=0.8)
        doc2 = Document(content="Unique doc 2", score=0.8)

        dense_docs = [doc1]
        sparse_docs = [doc2]

        # Weights sum to 2.0, should be normalized
        # Use explicit top_k to get all 2 unique documents
        merged = ResultMerger.weighted_fusion(
            dense_docs, sparse_docs, dense_weight=1.2, sparse_weight=0.8, top_k=2
        )

        assert len(merged) == 2

    def test_weighted_fusion_top_k(self, sample_documents):
        """Test weighted fusion with top_k limit."""
        for doc in sample_documents:
            doc.score = 0.8

        merged = ResultMerger.weighted_fusion(
            sample_documents[:2],
            [sample_documents[2]],
            top_k=2,
        )

        assert len(merged) == 2

    def test_weighted_fusion_no_scores(self):
        """Test weighted fusion with documents that have no scores."""
        doc1 = Document(content="Doc 1 no scores")
        doc2 = Document(content="Doc 2 no scores")

        # Use explicit top_k to get all 2 unique documents
        merged = ResultMerger.weighted_fusion([doc1], [doc2], top_k=2)

        assert len(merged) == 2

    def test_weighted_fusion_zero_scores(self):
        """Test weighted fusion with zero scores."""
        doc1 = Document(content="Doc 1 zero", score=0.0)
        doc2 = Document(content="Doc 2 zero", score=0.0)

        # Use explicit top_k to get all 2 unique documents
        merged = ResultMerger.weighted_fusion([doc1], [doc2], top_k=2)

        assert len(merged) == 2

    def test_weighted_fusion_identical_scores(self):
        """Test weighted fusion with identical scores."""
        doc1 = Document(content="Doc 1 identical", score=0.5)
        doc2 = Document(content="Doc 2 identical", score=0.5)

        # Use explicit top_k to get all 2 unique documents
        merged = ResultMerger.weighted_fusion([doc1], [doc2], top_k=2)

        assert len(merged) == 2

    def test_deduplicate_by_content_basic(self, duplicate_documents):
        """Test basic deduplication by content."""
        deduplicated = ResultMerger.deduplicate_by_content(duplicate_documents)

        assert len(deduplicated) == 2
        contents = [doc.content for doc in deduplicated]
        assert "Duplicate content" in contents
        assert "Unique content" in contents

    def test_deduplicate_by_content_empty(self):
        """Test deduplication with empty list."""
        deduplicated = ResultMerger.deduplicate_by_content([])

        assert isinstance(deduplicated, list)
        assert len(deduplicated) == 0

    def test_deduplicate_by_content_no_duplicates(self, sample_documents):
        """Test deduplication with no duplicates."""
        deduplicated = ResultMerger.deduplicate_by_content(sample_documents)

        assert len(deduplicated) == 3

    def test_deduplicate_by_content_all_duplicates(self):
        """Test deduplication with all duplicates."""
        docs = [
            Document(content="Same content"),
            Document(content="Same content"),
            Document(content="Same content"),
        ]

        deduplicated = ResultMerger.deduplicate_by_content(docs)

        assert len(deduplicated) == 1

    def test_deduplicate_by_content_case_insensitive(self):
        """Test deduplication is case-insensitive."""
        docs = [
            Document(content="Test Content"),
            Document(content="test content"),
            Document(content="TEST CONTENT"),
        ]

        deduplicated = ResultMerger.deduplicate_by_content(docs)

        # All should be considered the same
        assert len(deduplicated) == 1

    def test_deduplicate_by_content_whitespace_handling(self):
        """Test deduplication handles whitespace."""
        docs = [
            Document(content="  Test content  "),
            Document(content="Test content"),
        ]

        deduplicated = ResultMerger.deduplicate_by_content(docs)

        # Should be considered the same after stripping
        assert len(deduplicated) == 1

    def test_validate_fusion_config_success(self):
        """Test successful fusion configuration validation."""
        config = {"type": "rrf", "top_k": 10}

        result = ResultMerger.validate_fusion_config(config)

        assert result == config

    def test_validate_fusion_config_weighted(self):
        """Test validation with weighted fusion type."""
        config = {"type": "weighted", "top_k": 5}

        result = ResultMerger.validate_fusion_config(config)

        assert result == config

    def test_validate_fusion_config_default_type(self):
        """Test validation with default type."""
        config = {"top_k": 10}  # No type specified

        result = ResultMerger.validate_fusion_config(config)

        assert result == config

    def test_validate_fusion_config_invalid_type(self):
        """Test validation with invalid fusion type."""
        config = {"type": "invalid", "top_k": 10}

        with pytest.raises(ValueError, match="Unsupported fusion type"):
            ResultMerger.validate_fusion_config(config)

    def test_validate_fusion_config_invalid_top_k_type(self):
        """Test validation with invalid top_k type."""
        config = {"type": "rrf", "top_k": "ten"}

        with pytest.raises(ValueError, match="top_k must be > 0"):
            ResultMerger.validate_fusion_config(config)

    def test_validate_fusion_config_zero_top_k(self):
        """Test validation with zero top_k."""
        config = {"type": "rrf", "top_k": 0}

        with pytest.raises(ValueError, match="top_k must be > 0"):
            ResultMerger.validate_fusion_config(config)

    def test_validate_fusion_config_negative_top_k(self):
        """Test validation with negative top_k."""
        config = {"type": "rrf", "top_k": -5}

        with pytest.raises(ValueError, match="top_k must be > 0"):
            ResultMerger.validate_fusion_config(config)

    def test_validate_fusion_config_not_dict(self):
        """Test validation with non-dict input."""
        with pytest.raises(ValueError, match="Config must be dict"):
            ResultMerger.validate_fusion_config("not a dict")

    def test_validate_fusion_config_case_insensitive(self):
        """Test validation with case-insensitive type."""
        config = {"type": "RRF", "top_k": 10}

        result = ResultMerger.validate_fusion_config(config)

        assert result == config

    def test_rrf_fusion_preserves_document_meta(self, sample_documents):
        """Test that RRF fusion preserves document metadata."""
        dense_docs = [sample_documents[0]]
        sparse_docs = [sample_documents[1]]

        merged = ResultMerger.rrf_fusion(dense_docs, sparse_docs)

        for doc in merged:
            assert "source" in doc.meta

    def test_weighted_fusion_preserves_document_meta(self, sample_documents):
        """Test that weighted fusion preserves document metadata."""
        for doc in sample_documents:
            doc.score = 0.8

        merged = ResultMerger.weighted_fusion(
            [sample_documents[0]],
            [sample_documents[1]],
        )

        for doc in merged:
            assert "source" in doc.meta

    def test_rrf_fusion_score_boosting(self):
        """Test that documents appearing in both lists get higher scores."""
        doc = Document(content="Common doc", score=0.5)

        dense_docs = [doc, Document(content="Dense only", score=0.4)]
        sparse_docs = [doc, Document(content="Sparse only", score=0.3)]

        merged = ResultMerger.rrf_fusion(dense_docs, sparse_docs)

        # Common doc should appear first due to higher combined RRF score
        assert merged[0].content == "Common doc"

    def test_rrf_fusion_many_score_boosting(self):
        """Test N-way fusion score boosting."""
        doc_common = Document(content="Common")
        doc_rare = Document(content="Rare")

        ranked_lists = [
            [doc_common, doc_rare],
            [doc_common],
            [doc_common],
        ]

        merged = ResultMerger.rrf_fusion_many(ranked_lists)

        # Common doc should appear first
        assert merged[0].content == "Common"
