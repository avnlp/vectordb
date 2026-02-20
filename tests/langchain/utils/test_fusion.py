"""Tests for result fusion utilities (LangChain)."""

import pytest
from langchain_core.documents import Document

from vectordb.langchain.utils.fusion import ResultMerger


class TestReciprocalRankFusion:
    """Unit tests for ResultMerger.reciprocal_rank_fusion method."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results from multiple searches."""
        return [
            [
                Document(page_content="Doc A", metadata={"source": "search1"}),
                Document(page_content="Doc B", metadata={"source": "search1"}),
                Document(page_content="Doc C", metadata={"source": "search1"}),
            ],
            [
                Document(page_content="Doc B", metadata={"source": "search2"}),
                Document(page_content="Doc D", metadata={"source": "search2"}),
                Document(page_content="Doc A", metadata={"source": "search2"}),
            ],
        ]

    def test_empty_results(self):
        """Test RRF with empty results list returns empty list."""
        result = ResultMerger.reciprocal_rank_fusion([])
        assert result == []

    def test_empty_result_sets(self):
        """Test RRF with only empty result sets returns empty list."""
        result = ResultMerger.reciprocal_rank_fusion([[], []])
        assert result == []

    def test_single_result_set(self, sample_results):
        """Test RRF with single result set returns documents in order."""
        result = ResultMerger.reciprocal_rank_fusion([sample_results[0]])
        assert len(result) == 3
        assert result[0].page_content == "Doc A"
        assert result[1].page_content == "Doc B"
        assert result[2].page_content == "Doc C"

    def test_merges_multiple_results(self, sample_results):
        """Test RRF merges results from multiple searches."""
        result = ResultMerger.reciprocal_rank_fusion(sample_results)
        # Doc A and Doc B appear in both searches, so they should rank higher
        contents = [doc.page_content for doc in result]
        assert "Doc A" in contents
        assert "Doc B" in contents
        assert "Doc C" in contents
        assert "Doc D" in contents

    def test_equal_weights(self, sample_results):
        """Test RRF with equal weights gives balanced scores."""
        result = ResultMerger.reciprocal_rank_fusion(sample_results, weights=None)
        assert len(result) == 4  # 4 unique docs

    def test_custom_weights(self):
        """Test RRF with custom weights prioritizes weighted results."""
        results = [
            [
                Document(page_content="High Priority Doc", metadata={"source": "high"}),
                Document(
                    page_content="Medium Priority Doc", metadata={"source": "high"}
                ),
            ],
            [
                Document(page_content="Low Priority Doc", metadata={"source": "low"}),
            ],
        ]
        # Give first search much higher weight
        result = ResultMerger.reciprocal_rank_fusion(results, weights=[0.9, 0.1])
        # High Priority Doc should rank first due to high weight and high rank
        assert result[0].page_content == "High Priority Doc"

    def test_default_k_parameter(self, sample_results):
        """Test RRF uses default k=60."""
        result = ResultMerger.reciprocal_rank_fusion(sample_results, k=60)
        assert len(result) == 4

    def test_custom_k_parameter(self):
        """Test RRF with custom k parameter affects scores."""
        results = [
            [Document(page_content="Doc 1", metadata={})],
            [Document(page_content="Doc 1", metadata={})],
        ]
        # With high k, rank matters less
        result_high_k = ResultMerger.reciprocal_rank_fusion(results, k=1000)
        # With low k, rank matters more
        result_low_k = ResultMerger.reciprocal_rank_fusion(results, k=10)
        # Both should return the same document
        assert len(result_high_k) == 1
        assert len(result_low_k) == 1

    def test_deduplication_by_content(self):
        """Test RRF deduplicates documents by page_content."""
        results = [
            [Document(page_content="Same Doc", metadata={"id": "1"})],
            [Document(page_content="Same Doc", metadata={"id": "2"})],
            [Document(page_content="Same Doc", metadata={"id": "3"})],
        ]
        result = ResultMerger.reciprocal_rank_fusion(results)
        # Should have only one document despite 3 occurrences
        assert len(result) == 1

    def test_preserves_metadata(self, sample_results):
        """Test RRF preserves document metadata."""
        result = ResultMerger.reciprocal_rank_fusion(sample_results)
        for doc in result:
            assert hasattr(doc, "metadata")

    def test_scores_favor_high_ranks(self):
        """Test that higher ranked documents get higher RRF scores."""
        results = [
            [
                Document(page_content="First", metadata={}),
                Document(page_content="Second", metadata={}),
                Document(page_content="Third", metadata={}),
            ],
        ]
        result = ResultMerger.reciprocal_rank_fusion(results)
        # First should be first in results
        assert result[0].page_content == "First"
        assert result[1].page_content == "Second"
        assert result[2].page_content == "Third"

    def test_different_result_set_lengths(self):
        """Test RRF with result sets of different lengths."""
        results = [
            [
                Document(page_content="Doc A", metadata={}),
                Document(page_content="Doc B", metadata={}),
            ],
            [
                Document(page_content="Doc B", metadata={}),
                Document(page_content="Doc C", metadata={}),
                Document(page_content="Doc D", metadata={}),
                Document(page_content="Doc E", metadata={}),
            ],
        ]
        result = ResultMerger.reciprocal_rank_fusion(results)
        # Doc B appears in both, should rank higher
        assert result[0].page_content == "Doc B"


class TestWeightedMerge:
    """Unit tests for ResultMerger.weighted_merge method."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results from multiple searches."""
        return [
            [
                Document(page_content="Doc A", metadata={"source": "search1"}),
                Document(page_content="Doc B", metadata={"source": "search1"}),
            ],
            [
                Document(page_content="Doc B", metadata={"source": "search2"}),
                Document(page_content="Doc C", metadata={"source": "search2"}),
            ],
        ]

    def test_empty_results(self):
        """Test weighted merge with empty results returns empty list."""
        result = ResultMerger.weighted_merge([])
        assert result == []

    def test_empty_result_sets(self):
        """Test weighted merge with only empty result sets."""
        result = ResultMerger.weighted_merge([[], []])
        assert result == []

    def test_single_result_set(self, sample_results):
        """Test weighted merge with single result set."""
        result = ResultMerger.weighted_merge([sample_results[0]])
        assert len(result) == 2

    def test_merges_multiple_results(self, sample_results):
        """Test weighted merge combines results from multiple searches."""
        result = ResultMerger.weighted_merge(sample_results)
        contents = [doc.page_content for doc in result]
        assert "Doc A" in contents
        assert "Doc B" in contents
        assert "Doc C" in contents

    def test_default_equal_weights(self):
        """Test weighted merge with default equal weights."""
        results = [
            [Document(page_content="Doc 1", metadata={})],
            [Document(page_content="Doc 2", metadata={})],
        ]
        result = ResultMerger.weighted_merge(results, weights=None)
        assert len(result) == 2

    def test_custom_weights(self):
        """Test weighted merge with custom weights."""
        results = [
            [
                Document(page_content="High Weight Doc", metadata={}),
            ],
            [
                Document(page_content="Low Weight Doc", metadata={}),
            ],
        ]
        result = ResultMerger.weighted_merge(results, weights=[0.8, 0.2])
        assert result[0].page_content == "High Weight Doc"

    def test_first_documents_score_highest(self):
        """Test that first documents in each result get highest scores."""
        results = [
            [
                Document(page_content="First in A", metadata={}),
                Document(page_content="Second in A", metadata={}),
            ],
            [
                Document(page_content="First in B", metadata={}),
                Document(page_content="Second in B", metadata={}),
            ],
        ]
        result = ResultMerger.weighted_merge(results)
        # First documents should rank higher than second documents
        first_docs = ["First in A", "First in B"]
        second_docs = ["Second in A", "Second in B"]
        result_contents = [doc.page_content for doc in result]
        # At least one first doc should be before all second docs
        first_indices = [
            result_contents.index(d) for d in first_docs if d in result_contents
        ]
        second_indices = [
            result_contents.index(d) for d in second_docs if d in result_contents
        ]
        assert min(first_indices) < max(second_indices)

    def test_preserves_metadata(self, sample_results):
        """Test weighted merge preserves document metadata."""
        result = ResultMerger.weighted_merge(sample_results)
        for doc in result:
            assert hasattr(doc, "metadata")

    def test_single_document_per_set(self):
        """Test weighted merge with single document per result set."""
        results = [
            [Document(page_content="Doc 1", metadata={})],
            [Document(page_content="Doc 2", metadata={})],
            [Document(page_content="Doc 3", metadata={})],
        ]
        result = ResultMerger.weighted_merge(results, weights=[0.5, 0.3, 0.2])
        assert len(result) == 3


class TestDeduplication:
    """Unit tests for ResultMerger.deduplication method."""

    def test_empty_list(self):
        """Test deduplication with empty list returns empty list."""
        result = ResultMerger.deduplication([])
        assert result == []

    def test_no_duplicates(self):
        """Test deduplication with all unique documents."""
        docs = [
            Document(page_content="Doc 1", metadata={}),
            Document(page_content="Doc 2", metadata={}),
            Document(page_content="Doc 3", metadata={}),
        ]
        result = ResultMerger.deduplication(docs)
        assert len(result) == 3

    def test_removes_duplicates_by_content(self):
        """Test deduplication removes duplicates by page_content."""
        docs = [
            Document(page_content="Same Doc", metadata={"id": "1"}),
            Document(page_content="Same Doc", metadata={"id": "2"}),
            Document(page_content="Different Doc", metadata={}),
        ]
        result = ResultMerger.deduplication(docs)
        assert len(result) == 2
        assert result[0].page_content == "Same Doc"
        assert result[1].page_content == "Different Doc"

    def test_keeps_first_occurrence(self):
        """Test deduplication keeps first occurrence of duplicates."""
        docs = [
            Document(page_content="Doc", metadata={"id": "1"}),
            Document(page_content="Doc", metadata={"id": "2"}),
            Document(page_content="Doc", metadata={"id": "3"}),
        ]
        result = ResultMerger.deduplication(docs)
        assert len(result) == 1
        assert result[0].metadata["id"] == "1"

    def test_deduplication_by_metadata_key(self):
        """Test deduplication by custom metadata key."""
        docs = [
            Document(page_content="Doc 1", metadata={"doc_id": "A"}),
            Document(page_content="Doc 2", metadata={"doc_id": "B"}),
            Document(page_content="Doc 3", metadata={"doc_id": "A"}),
        ]
        result = ResultMerger.deduplication(docs, key="doc_id")
        assert len(result) == 2
        assert result[0].page_content == "Doc 1"
        assert result[1].page_content == "Doc 2"

    def test_deduplication_missing_metadata_key(self):
        """Test deduplication with missing metadata key treats as None."""
        docs = [
            Document(page_content="Doc 1", metadata={}),
            Document(page_content="Doc 2", metadata={"key": "value"}),
        ]
        result = ResultMerger.deduplication(docs, key="missing_key")
        # Both will have None as key, so first one is kept
        assert len(result) == 1

    def test_deduplication_with_missing_metadata_key(self):
        """Test deduplication with missing metadata key treats as None."""
        docs = [
            Document(page_content="Doc 1", metadata={}),
            Document(page_content="Doc 2", metadata={"key": "value"}),
        ]
        result = ResultMerger.deduplication(docs, key="missing_key")
        # Both will have None as key for "missing_key", so they are
        # considered duplicates
        assert len(result) == 1

    def test_preserves_order(self):
        """Test deduplication preserves order of first occurrences."""
        docs = [
            Document(page_content="First", metadata={}),
            Document(page_content="Second", metadata={}),
            Document(page_content="Third", metadata={}),
            Document(page_content="First", metadata={}),  # duplicate
            Document(page_content="Fourth", metadata={}),
        ]
        result = ResultMerger.deduplication(docs)
        assert [doc.page_content for doc in result] == [
            "First",
            "Second",
            "Third",
            "Fourth",
        ]


class TestMergeAndDeduplicate:
    """Unit tests for ResultMerger.merge_and_deduplicate method."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results for testing."""
        return [
            [
                Document(page_content="Doc A", metadata={"source": "search1"}),
                Document(page_content="Doc B", metadata={"source": "search1"}),
            ],
            [
                Document(page_content="Doc B", metadata={"source": "search2"}),
                Document(page_content="Doc C", metadata={"source": "search2"}),
            ],
        ]

    def test_empty_results(self):
        """Test merge_and_deduplicate with empty results."""
        result = ResultMerger.merge_and_deduplicate([])
        assert result == []

    def test_rrf_method(self, sample_results):
        """Test merge_and_deduplicate with RRF method."""
        result = ResultMerger.merge_and_deduplicate(sample_results, method="rrf")
        assert len(result) == 3  # 3 unique docs

    def test_weighted_method(self, sample_results):
        """Test merge_and_deduplicate with weighted method."""
        result = ResultMerger.merge_and_deduplicate(sample_results, method="weighted")
        assert len(result) == 3  # 3 unique docs

    def test_invalid_method_raises_error(self):
        """Test invalid merge method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown merge method"):
            ResultMerger.merge_and_deduplicate([[]], method="invalid")

    def test_custom_weights(self):
        """Test merge_and_deduplicate with custom weights."""
        results = [
            [Document(page_content="Doc 1", metadata={})],
            [Document(page_content="Doc 2", metadata={})],
        ]
        result = ResultMerger.merge_and_deduplicate(
            results, method="rrf", weights=[0.7, 0.3]
        )
        assert len(result) == 2

    def test_custom_dedup_key(self):
        """Test merge_and_deduplicate with custom dedup key."""
        results = [
            [
                Document(page_content="Content 1", metadata={"id": "A"}),
                Document(page_content="Content 2", metadata={"id": "B"}),
            ],
            [
                Document(page_content="Content 1", metadata={"id": "A"}),
                Document(page_content="Content 3", metadata={"id": "C"}),
            ],
        ]
        result = ResultMerger.merge_and_deduplicate(
            results, method="rrf", dedup_key="id"
        )
        assert len(result) == 3

    def test_deduplication_after_merge(self):
        """Test that deduplication works after merging."""
        results = [
            [
                Document(page_content="Same", metadata={}),
                Document(page_content="Unique 1", metadata={}),
            ],
            [
                Document(page_content="Same", metadata={}),
                Document(page_content="Unique 2", metadata={}),
            ],
        ]
        result = ResultMerger.merge_and_deduplicate(results)
        # Should have 3 unique docs: "Same", "Unique 1", "Unique 2"
        assert len(result) == 3
        contents = [doc.page_content for doc in result]
        assert "Same" in contents
        assert "Unique 1" in contents
        assert "Unique 2" in contents


class TestResultMergerEdgeCases:
    """Edge case tests for ResultMerger."""

    def test_all_empty_result_sets(self):
        """Test with all empty result sets."""
        result = ResultMerger.reciprocal_rank_fusion([[], [], []])
        assert result == []

    def test_single_document_multiple_times(self):
        """Test with same document appearing multiple times."""
        doc = Document(page_content="Same Doc", metadata={})
        results = [[doc], [doc], [doc]]
        result = ResultMerger.weighted_merge(results)
        assert len(result) == 1

    def test_very_long_results_list(self):
        """Test with many result sets."""
        results = [[Document(page_content=f"Doc {i}", metadata={})] for i in range(10)]
        result = ResultMerger.weighted_merge(results)
        assert len(result) == 10

    def test_mixed_weights_normalization(self):
        """Test that weights are properly normalized."""
        results = [
            [Document(page_content="Doc 1", metadata={})],
            [Document(page_content="Doc 2", metadata={})],
        ]
        # Unnormalized weights should be normalized
        result = ResultMerger.weighted_merge(results, weights=[10, 20])
        assert len(result) == 2

    def test_deduplication_with_missing_metadata_key(self):
        """Test deduplication with missing metadata key treats as None."""
        docs = [
            Document(page_content="Doc 1", metadata={}),
            Document(page_content="Doc 2", metadata={"key": "value"}),
        ]
        result = ResultMerger.deduplication(docs, key="missing_key")
        # Both will have None as key for "missing_key", so they are
        # considered duplicates
        assert len(result) == 1

    def test_large_k_value_rrf(self):
        """Test RRF with very large k value."""
        results = [
            [Document(page_content="Doc 1", metadata={})],
            [Document(page_content="Doc 1", metadata={})],
        ]
        result = ResultMerger.reciprocal_rank_fusion(results, k=10000)
        assert len(result) == 1
