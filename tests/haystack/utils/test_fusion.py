"""Tests for result fusion utilities."""

import pytest
from haystack import Document

from vectordb.haystack.utils.fusion import ResultMerger


class TestResultMerger:
    """Tests for ResultMerger class."""

    @pytest.fixture
    def sample_dense_results(self) -> list[Document]:
        """Create sample dense retrieval results."""
        return [
            Document(content="Dense doc 1", id="d1", meta={"score": 0.95}),
            Document(content="Dense doc 2", id="d2", meta={"score": 0.85}),
            Document(content="Dense doc 3", id="d3", meta={"score": 0.75}),
        ]

    @pytest.fixture
    def sample_sparse_results(self) -> list[Document]:
        """Create sample sparse retrieval results."""
        return [
            Document(content="Sparse doc 2", id="d2", meta={"score": 0.90}),
            Document(content="Sparse doc 4", id="d4", meta={"score": 0.80}),
            Document(content="Sparse doc 5", id="d5", meta={"score": 0.70}),
        ]


class TestFuseRRF(TestResultMerger):
    """Tests for fuse_rrf method."""

    def test_fuse_rrf_basic(
        self,
        sample_dense_results: list[Document],
        sample_sparse_results: list[Document],
    ) -> None:
        """Test basic RRF fusion."""
        result = ResultMerger.fuse_rrf(
            sample_dense_results, sample_sparse_results, top_k=5
        )
        assert len(result) <= 5
        assert all(isinstance(doc, Document) for doc in result)

    def test_fuse_rrf_respects_top_k(
        self,
        sample_dense_results: list[Document],
        sample_sparse_results: list[Document],
    ) -> None:
        """Test that RRF fusion respects top_k."""
        result = ResultMerger.fuse_rrf(
            sample_dense_results, sample_sparse_results, top_k=2
        )
        assert len(result) == 2

    def test_fuse_rrf_default_k(
        self,
        sample_dense_results: list[Document],
        sample_sparse_results: list[Document],
    ) -> None:
        """Test RRF fusion with default k parameter."""
        result = ResultMerger.fuse_rrf(sample_dense_results, sample_sparse_results)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_fuse_rrf_custom_k(
        self,
        sample_dense_results: list[Document],
        sample_sparse_results: list[Document],
    ) -> None:
        """Test RRF fusion with custom k parameter."""
        result1 = ResultMerger.fuse_rrf(
            sample_dense_results, sample_sparse_results, k=30.0
        )
        result2 = ResultMerger.fuse_rrf(
            sample_dense_results, sample_sparse_results, k=100.0
        )
        # Different k values may produce different orderings
        assert len(result1) > 0
        assert len(result2) > 0

    def test_fuse_rrf_deduplication(self) -> None:
        """Test that RRF fusion deduplicates results."""
        # Create results with overlapping documents
        dense = [
            Document(content="Doc A", id="same"),
            Document(content="Doc B", id="dense_only"),
        ]
        sparse = [
            Document(content="Doc A", id="same"),
            Document(content="Doc C", id="sparse_only"),
        ]
        result = ResultMerger.fuse_rrf(dense, sparse, top_k=10)
        # The "same" document should appear once with combined score
        ids = [doc.id for doc in result]
        assert ids.count("same") == 1

    def test_fuse_rrf_empty_dense(self, sample_sparse_results: list[Document]) -> None:
        """Test RRF fusion with empty dense results."""
        result = ResultMerger.fuse_rrf([], sample_sparse_results, top_k=5)
        assert len(result) == 3  # All sparse results
        assert all(doc.id in ["d2", "d4", "d5"] for doc in result)

    def test_fuse_rrf_empty_sparse(self, sample_dense_results: list[Document]) -> None:
        """Test RRF fusion with empty sparse results."""
        result = ResultMerger.fuse_rrf(sample_dense_results, [], top_k=5)
        assert len(result) == 3  # All dense results
        assert all(doc.id in ["d1", "d2", "d3"] for doc in result)

    def test_fuse_rrf_both_empty(self) -> None:
        """Test RRF fusion with both empty lists."""
        result = ResultMerger.fuse_rrf([], [], top_k=5)
        assert result == []

    def test_fuse_rrf_single_document(self) -> None:
        """Test RRF fusion with single document."""
        dense = [Document(content="Single doc", id="single")]
        result = ResultMerger.fuse_rrf(dense, [], top_k=5)
        assert len(result) == 1
        assert result[0].id == "single"

    def test_fuse_rrf_overlapping_documents_score_boost(self) -> None:
        """Test that overlapping documents get boosted scores."""
        dense = [
            Document(content="Overlap doc", id="overlap"),
            Document(content="Dense only", id="dense_only"),
        ]
        sparse = [
            Document(content="Overlap doc", id="overlap"),
            Document(content="Sparse only", id="sparse_only"),
        ]
        result = ResultMerger.fuse_rrf(dense, sparse, top_k=3)
        # Overlapping document should be ranked first (highest combined RRF score)
        assert result[0].id == "overlap"

    def test_fuse_rrf_non_overlapping_results(self) -> None:
        """Test RRF fusion with completely non-overlapping results."""
        dense = [
            Document(content="Dense A", id="da"),
            Document(content="Dense B", id="db"),
        ]
        sparse = [
            Document(content="Sparse A", id="sa"),
            Document(content="Sparse B", id="sb"),
        ]
        result = ResultMerger.fuse_rrf(dense, sparse, top_k=10)
        assert len(result) == 4
        ids = {doc.id for doc in result}
        assert ids == {"da", "db", "sa", "sb"}

    def test_fuse_rrf_documents_without_ids(self) -> None:
        """Test RRF fusion with documents that have no IDs (fallback to content)."""
        dense = [
            Document(content="This is a document without an ID for dense search"),
            Document(content="Another dense document without ID"),
        ]
        sparse = [
            Document(content="This is a document without an ID for dense search"),
            Document(content="Sparse document without ID"),
        ]
        result = ResultMerger.fuse_rrf(dense, sparse, top_k=10)
        # First doc in both lists has same content[:50], should be deduplicated
        assert len(result) == 3

    def test_fuse_rrf_top_k_truncation(self) -> None:
        """Test that results are truncated to top_k."""
        dense = [Document(content=f"Dense {i}", id=f"d{i}") for i in range(10)]
        sparse = [Document(content=f"Sparse {i}", id=f"s{i}") for i in range(10)]
        result = ResultMerger.fuse_rrf(dense, sparse, top_k=5)
        assert len(result) == 5

    def test_fuse_rrf_various_k_values(self) -> None:
        """Test RRF with various k parameter values."""
        dense = [Document(content="Doc 1", id="d1")]
        sparse = [Document(content="Doc 2", id="s1")]

        # Small k emphasizes position more
        result_small_k = ResultMerger.fuse_rrf(dense, sparse, k=1.0)
        # Large k reduces position emphasis
        result_large_k = ResultMerger.fuse_rrf(dense, sparse, k=1000.0)

        assert len(result_small_k) == 2
        assert len(result_large_k) == 2

    def test_fuse_rrf_large_result_sets(self) -> None:
        """Test RRF with large result sets."""
        dense = [Document(content=f"Dense doc {i}", id=f"d{i}") for i in range(100)]
        sparse = [Document(content=f"Sparse doc {i}", id=f"s{i}") for i in range(100)]
        result = ResultMerger.fuse_rrf(dense, sparse, top_k=50)
        assert len(result) == 50

    def test_fuse_rrf_duplicate_ids_in_same_list(self) -> None:
        """Test RRF when same ID appears multiple times in one list."""
        dense = [
            Document(content="First", id="dup"),
            Document(content="Second", id="dup"),
            Document(content="Third", id="unique"),
        ]
        sparse: list[Document] = []
        result = ResultMerger.fuse_rrf(dense, sparse, top_k=10)
        # Duplicate IDs should be merged (second overwrites first)
        ids = [doc.id for doc in result]
        assert ids.count("dup") == 1


class TestFuseWeighted(TestResultMerger):
    """Tests for fuse_weighted method."""

    def test_fuse_weighted_basic(
        self,
        sample_dense_results: list[Document],
        sample_sparse_results: list[Document],
    ) -> None:
        """Test basic weighted fusion."""
        result = ResultMerger.fuse_weighted(
            sample_dense_results, sample_sparse_results, top_k=5
        )
        assert len(result) <= 5
        assert all(isinstance(doc, Document) for doc in result)

    def test_fuse_weighted_respects_top_k(
        self,
        sample_dense_results: list[Document],
        sample_sparse_results: list[Document],
    ) -> None:
        """Test that weighted fusion respects top_k."""
        result = ResultMerger.fuse_weighted(
            sample_dense_results, sample_sparse_results, top_k=2
        )
        assert len(result) == 2

    def test_fuse_weighted_custom_weights(
        self,
        sample_dense_results: list[Document],
        sample_sparse_results: list[Document],
    ) -> None:
        """Test weighted fusion with custom weights."""
        result1 = ResultMerger.fuse_weighted(
            sample_dense_results,
            sample_sparse_results,
            dense_weight=0.9,
            sparse_weight=0.1,
        )
        result2 = ResultMerger.fuse_weighted(
            sample_dense_results,
            sample_sparse_results,
            dense_weight=0.1,
            sparse_weight=0.9,
        )
        # Different weights may produce different results
        assert len(result1) > 0
        assert len(result2) > 0

    def test_fuse_weighted_weight_normalization(
        self,
        sample_dense_results: list[Document],
        sample_sparse_results: list[Document],
    ) -> None:
        """Test that weighted fusion normalizes weights correctly."""
        # These should produce equivalent results regardless of initial scale
        result1 = ResultMerger.fuse_weighted(
            sample_dense_results,
            sample_sparse_results,
            dense_weight=0.7,
            sparse_weight=0.3,
        )
        result2 = ResultMerger.fuse_weighted(
            sample_dense_results,
            sample_sparse_results,
            dense_weight=7.0,
            sparse_weight=3.0,
        )
        # Same normalized weights should produce same order
        assert [doc.id for doc in result1] == [doc.id for doc in result2]

    def test_fuse_weighted_deduplication(self) -> None:
        """Test that weighted fusion deduplicates results."""
        dense = [
            Document(content="Doc A", id="same"),
            Document(content="Doc B", id="dense_only"),
        ]
        sparse = [
            Document(content="Doc A", id="same"),
            Document(content="Doc C", id="sparse_only"),
        ]
        result = ResultMerger.fuse_weighted(dense, sparse, top_k=10)
        ids = [doc.id for doc in result]
        assert ids.count("same") == 1

    def test_fuse_weighted_empty_dense(
        self, sample_sparse_results: list[Document]
    ) -> None:
        """Test weighted fusion with empty dense results."""
        result = ResultMerger.fuse_weighted([], sample_sparse_results, top_k=5)
        assert len(result) == 3

    def test_fuse_weighted_empty_sparse(
        self, sample_dense_results: list[Document]
    ) -> None:
        """Test weighted fusion with empty sparse results."""
        result = ResultMerger.fuse_weighted(sample_dense_results, [], top_k=5)
        assert len(result) == 3

    def test_fuse_weighted_both_empty(self) -> None:
        """Test weighted fusion with both empty lists."""
        result = ResultMerger.fuse_weighted([], [], top_k=5)
        assert result == []

    def test_fuse_weighted_single_document(self) -> None:
        """Test weighted fusion with single document."""
        dense = [Document(content="Single doc", id="single")]
        result = ResultMerger.fuse_weighted(dense, [], top_k=5)
        assert len(result) == 1
        assert result[0].id == "single"

    def test_fuse_weighted_overlapping_documents_score_boost(self) -> None:
        """Test that overlapping documents get boosted scores."""
        dense = [
            Document(content="Overlap doc", id="overlap"),
            Document(content="Dense only", id="dense_only"),
        ]
        sparse = [
            Document(content="Overlap doc", id="overlap"),
            Document(content="Sparse only", id="sparse_only"),
        ]
        result = ResultMerger.fuse_weighted(dense, sparse, top_k=3)
        # Overlapping document should be ranked first
        assert result[0].id == "overlap"

    def test_fuse_weighted_non_overlapping_results(self) -> None:
        """Test weighted fusion with completely non-overlapping results."""
        dense = [
            Document(content="Dense A", id="da"),
            Document(content="Dense B", id="db"),
        ]
        sparse = [
            Document(content="Sparse A", id="sa"),
            Document(content="Sparse B", id="sb"),
        ]
        result = ResultMerger.fuse_weighted(dense, sparse, top_k=10)
        assert len(result) == 4
        ids = {doc.id for doc in result}
        assert ids == {"da", "db", "sa", "sb"}

    def test_fuse_weighted_documents_without_ids(self) -> None:
        """Test weighted fusion with documents that have no IDs."""
        dense = [
            Document(content="This is a document without an ID for dense search"),
            Document(content="Another dense document without ID"),
        ]
        sparse = [
            Document(content="This is a document without an ID for dense search"),
            Document(content="Sparse document without ID"),
        ]
        result = ResultMerger.fuse_weighted(dense, sparse, top_k=10)
        # First doc in both lists has same content[:50], should be deduplicated
        assert len(result) == 3

    def test_fuse_weighted_top_k_truncation(self) -> None:
        """Test that results are truncated to top_k."""
        dense = [Document(content=f"Dense {i}", id=f"d{i}") for i in range(10)]
        sparse = [Document(content=f"Sparse {i}", id=f"s{i}") for i in range(10)]
        result = ResultMerger.fuse_weighted(dense, sparse, top_k=5)
        assert len(result) == 5

    def test_fuse_weighted_dense_only_weight(self) -> None:
        """Test weighted fusion with only dense weight."""
        dense = [Document(content="Dense doc", id="d1")]
        sparse = [Document(content="Sparse doc", id="s1")]
        result = ResultMerger.fuse_weighted(
            dense, sparse, dense_weight=1.0, sparse_weight=0.0
        )
        # Should still work, sparse weight normalized
        assert len(result) == 2

    def test_fuse_weighted_sparse_only_weight(self) -> None:
        """Test weighted fusion with only sparse weight."""
        dense = [Document(content="Dense doc", id="d1")]
        sparse = [Document(content="Sparse doc", id="s1")]
        result = ResultMerger.fuse_weighted(
            dense, sparse, dense_weight=0.0, sparse_weight=1.0
        )
        # Should still work, dense weight normalized
        assert len(result) == 2

    def test_fuse_weighted_equal_weights(self) -> None:
        """Test weighted fusion with equal weights."""
        dense = [
            Document(content="Dense 1", id="d1"),
            Document(content="Dense 2", id="d2"),
        ]
        sparse = [
            Document(content="Sparse 1", id="s1"),
            Document(content="Sparse 2", id="s2"),
        ]
        result = ResultMerger.fuse_weighted(
            dense, sparse, dense_weight=0.5, sparse_weight=0.5, top_k=10
        )
        # With equal weights, first positions from each should have equal scores
        assert len(result) == 4

    def test_fuse_weighted_large_result_sets(self) -> None:
        """Test weighted fusion with large result sets."""
        dense = [Document(content=f"Dense doc {i}", id=f"d{i}") for i in range(100)]
        sparse = [Document(content=f"Sparse doc {i}", id=f"s{i}") for i in range(100)]
        result = ResultMerger.fuse_weighted(dense, sparse, top_k=50)
        assert len(result) == 50

    def test_fuse_weighted_score_calculation(self) -> None:
        """Test that weighted scores are calculated correctly."""
        # When dense has higher weight, dense first-ranked should win
        dense = [Document(content="Dense first", id="d1")]
        sparse = [Document(content="Sparse first", id="s1")]
        result = ResultMerger.fuse_weighted(
            dense, sparse, dense_weight=0.9, sparse_weight=0.1
        )
        # Dense doc should be ranked first due to higher weight
        assert result[0].id == "d1"

        # Flip weights
        result2 = ResultMerger.fuse_weighted(
            dense, sparse, dense_weight=0.1, sparse_weight=0.9
        )
        # Sparse doc should be ranked first
        assert result2[0].id == "s1"


class TestFuse(TestResultMerger):
    """Tests for the fuse dispatcher method."""

    def test_fuse_rrf_strategy(
        self,
        sample_dense_results: list[Document],
        sample_sparse_results: list[Document],
    ) -> None:
        """Test fuse with rrf strategy."""
        result = ResultMerger.fuse(
            sample_dense_results, sample_sparse_results, top_k=5, strategy="rrf"
        )
        assert len(result) <= 5
        assert all(isinstance(doc, Document) for doc in result)

    def test_fuse_weighted_strategy(
        self,
        sample_dense_results: list[Document],
        sample_sparse_results: list[Document],
    ) -> None:
        """Test fuse with weighted strategy."""
        result = ResultMerger.fuse(
            sample_dense_results, sample_sparse_results, top_k=5, strategy="weighted"
        )
        assert len(result) <= 5
        assert all(isinstance(doc, Document) for doc in result)

    def test_fuse_rrf_with_custom_k(
        self,
        sample_dense_results: list[Document],
        sample_sparse_results: list[Document],
    ) -> None:
        """Test fuse rrf strategy with custom k parameter."""
        result = ResultMerger.fuse(
            sample_dense_results,
            sample_sparse_results,
            top_k=5,
            strategy="rrf",
            k=30.0,
        )
        assert len(result) <= 5

    def test_fuse_weighted_with_custom_weights(
        self,
        sample_dense_results: list[Document],
        sample_sparse_results: list[Document],
    ) -> None:
        """Test fuse weighted strategy with custom weights."""
        result = ResultMerger.fuse(
            sample_dense_results,
            sample_sparse_results,
            top_k=5,
            strategy="weighted",
            dense_weight=0.8,
            sparse_weight=0.2,
        )
        assert len(result) <= 5

    def test_fuse_unknown_strategy(
        self,
        sample_dense_results: list[Document],
        sample_sparse_results: list[Document],
    ) -> None:
        """Test fuse with unknown strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown fusion strategy"):
            ResultMerger.fuse(
                sample_dense_results,
                sample_sparse_results,
                strategy="unknown_strategy",
            )

    def test_fuse_default_strategy(
        self,
        sample_dense_results: list[Document],
        sample_sparse_results: list[Document],
    ) -> None:
        """Test fuse uses rrf as default strategy."""
        # Default strategy should be 'rrf'
        result_default = ResultMerger.fuse(
            sample_dense_results, sample_sparse_results, top_k=5
        )
        result_rrf = ResultMerger.fuse_rrf(
            sample_dense_results, sample_sparse_results, top_k=5
        )
        # Should produce same results
        assert [doc.id for doc in result_default] == [doc.id for doc in result_rrf]

    def test_fuse_empty_inputs(self) -> None:
        """Test fuse with empty inputs."""
        result_rrf = ResultMerger.fuse([], [], strategy="rrf")
        result_weighted = ResultMerger.fuse([], [], strategy="weighted")
        assert result_rrf == []
        assert result_weighted == []

    def test_fuse_preserves_document_metadata(self) -> None:
        """Test that fuse preserves document metadata."""
        dense = [
            Document(
                content="Doc with meta",
                id="d1",
                meta={"source": "dense", "score": 0.9},
            )
        ]
        sparse = [
            Document(
                content="Another doc",
                id="s1",
                meta={"source": "sparse", "score": 0.8},
            )
        ]
        result = ResultMerger.fuse(dense, sparse, strategy="rrf")
        # Check metadata is preserved
        doc_with_meta = next((d for d in result if d.id == "d1"), None)
        assert doc_with_meta is not None
        assert doc_with_meta.meta["source"] == "dense"

    def test_fuse_case_sensitivity_strategy(
        self,
        sample_dense_results: list[Document],
        sample_sparse_results: list[Document],
    ) -> None:
        """Test that strategy parameter is case-sensitive."""
        # Uppercase should fail
        with pytest.raises(ValueError, match="Unknown fusion strategy"):
            ResultMerger.fuse(
                sample_dense_results, sample_sparse_results, strategy="RRF"
            )
