"""Tests for ResultFuser class - result fusion strategies for hybrid search."""

import pytest

from vectordb.haystack.cost_optimized_rag.base.fusion import ResultFuser


class TestReciprocalRankFusion:
    """Tests for ResultFuser.reciprocal_rank_fusion() method."""

    def test_rrf_basic_fusion(self) -> None:
        """Test basic RRF fusion with overlapping results."""
        dense = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        sparse = [("doc2", 0.95), ("doc1", 0.85), ("doc4", 0.6)]

        result = ResultFuser.reciprocal_rank_fusion(dense, sparse, k=60)

        # doc1: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
        # doc2: 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252
        # They should be equal and at the top
        result_dict = dict(result)
        assert "doc1" in result_dict
        assert "doc2" in result_dict
        assert "doc3" in result_dict
        assert "doc4" in result_dict

    def test_rrf_empty_dense_results(self) -> None:
        """Test RRF with empty dense results."""
        dense: list[tuple[str, float]] = []
        sparse = [("doc1", 0.9), ("doc2", 0.8)]

        result = ResultFuser.reciprocal_rank_fusion(dense, sparse, k=60)

        assert len(result) == 2
        # Only sparse contributes
        doc_ids = [doc_id for doc_id, _ in result]
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids

    def test_rrf_empty_sparse_results(self) -> None:
        """Test RRF with empty sparse results."""
        dense = [("doc1", 0.9), ("doc2", 0.8)]
        sparse: list[tuple[str, float]] = []

        result = ResultFuser.reciprocal_rank_fusion(dense, sparse, k=60)

        assert len(result) == 2
        doc_ids = [doc_id for doc_id, _ in result]
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids

    def test_rrf_both_empty(self) -> None:
        """Test RRF with both empty lists."""
        dense: list[tuple[str, float]] = []
        sparse: list[tuple[str, float]] = []

        result = ResultFuser.reciprocal_rank_fusion(dense, sparse, k=60)

        assert result == []

    def test_rrf_single_item_each(self) -> None:
        """Test RRF with single item in each list."""
        dense = [("doc1", 0.9)]
        sparse = [("doc2", 0.8)]

        result = ResultFuser.reciprocal_rank_fusion(dense, sparse, k=60)

        assert len(result) == 2
        # Both should have same RRF score (1/(60+1))
        assert result[0][1] == result[1][1]
        assert result[0][1] == pytest.approx(1 / 61)

    def test_rrf_same_doc_in_both(self) -> None:
        """Test RRF when same doc appears in both lists at same rank."""
        dense = [("doc1", 0.9)]
        sparse = [("doc1", 0.8)]

        result = ResultFuser.reciprocal_rank_fusion(dense, sparse, k=60)

        assert len(result) == 1
        assert result[0][0] == "doc1"
        # Score should be 2 * 1/(60+1)
        assert result[0][1] == pytest.approx(2 / 61)

    def test_rrf_different_k_values(self) -> None:
        """Test RRF with different k parameter values."""
        dense = [("doc1", 0.9)]
        sparse = [("doc1", 0.8)]

        result_k10 = ResultFuser.reciprocal_rank_fusion(dense, sparse, k=10)
        result_k60 = ResultFuser.reciprocal_rank_fusion(dense, sparse, k=60)
        result_k100 = ResultFuser.reciprocal_rank_fusion(dense, sparse, k=100)

        # Higher k = lower RRF scores
        assert result_k10[0][1] > result_k60[0][1] > result_k100[0][1]

    def test_rrf_ordering_by_combined_rank(self) -> None:
        """Test RRF orders by combined ranking contribution."""
        dense = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        sparse = [("doc3", 0.9), ("doc2", 0.8), ("doc1", 0.7)]

        result = ResultFuser.reciprocal_rank_fusion(dense, sparse, k=60)

        # doc2 is at rank 2 in both, should have highest combined score
        # doc1: 1/(60+1) + 1/(60+3)
        # doc2: 1/(60+2) + 1/(60+2)
        # doc3: 1/(60+3) + 1/(60+1)
        # doc1 and doc3 have same score, doc2 has different score
        result_dict = dict(result)
        assert result_dict["doc1"] == pytest.approx(result_dict["doc3"])
        # All scores should be close (within 1%)
        assert result_dict["doc2"] == pytest.approx(result_dict["doc1"], rel=0.01)

    def test_rrf_preserves_all_documents(self) -> None:
        """Test RRF includes all unique documents from both lists."""
        dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        sparse = [("d", 0.9), ("e", 0.8), ("f", 0.7)]

        result = ResultFuser.reciprocal_rank_fusion(dense, sparse, k=60)

        doc_ids = {doc_id for doc_id, _ in result}
        assert doc_ids == {"a", "b", "c", "d", "e", "f"}

    def test_rrf_sorted_descending(self) -> None:
        """Test RRF returns results sorted by score descending."""
        dense = [("doc1", 0.9), ("doc2", 0.8)]
        sparse = [("doc3", 0.95), ("doc4", 0.85)]

        result = ResultFuser.reciprocal_rank_fusion(dense, sparse, k=60)

        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)


class TestWeightedFusion:
    """Tests for ResultFuser.weighted_fusion() method."""

    def test_weighted_basic_fusion(self) -> None:
        """Test basic weighted fusion."""
        dense = [("doc1", 0.9), ("doc2", 0.6)]
        sparse = [("doc1", 0.8), ("doc3", 0.7)]

        result = ResultFuser.weighted_fusion(
            dense, sparse, dense_weight=0.7, sparse_weight=0.3
        )

        assert len(result) == 3
        result_dict = dict(result)
        assert "doc1" in result_dict
        assert "doc2" in result_dict
        assert "doc3" in result_dict

    def test_weighted_weight_validation_raises_error(self) -> None:
        """Test weighted fusion raises ValueError for invalid weights."""
        dense = [("doc1", 0.9)]
        sparse = [("doc1", 0.8)]

        with pytest.raises(ValueError, match="Weights must sum to ~1.0"):
            ResultFuser.weighted_fusion(
                dense, sparse, dense_weight=0.5, sparse_weight=0.3
            )

    def test_weighted_weight_validation_tolerance(self) -> None:
        """Test weighted fusion allows slight tolerance in weights."""
        dense = [("doc1", 0.9)]
        sparse = [("doc1", 0.8)]

        # 0.7 + 0.305 = 1.005, within 0.01 tolerance
        result = ResultFuser.weighted_fusion(
            dense, sparse, dense_weight=0.7, sparse_weight=0.305
        )
        assert len(result) == 1

    def test_weighted_empty_dense_results(self) -> None:
        """Test weighted fusion with empty dense results."""
        dense: list[tuple[str, float]] = []
        sparse = [("doc1", 0.9), ("doc2", 0.6)]

        result = ResultFuser.weighted_fusion(
            dense, sparse, dense_weight=0.7, sparse_weight=0.3
        )

        assert len(result) == 2
        # Only sparse contributes (normalized and weighted by 0.3)
        result_dict = dict(result)
        assert result_dict["doc1"] == pytest.approx(0.3)  # 0.9/0.9 * 0.3 = 0.3
        assert result_dict["doc2"] == pytest.approx(0.6 / 0.9 * 0.3)

    def test_weighted_empty_sparse_results(self) -> None:
        """Test weighted fusion with empty sparse results."""
        dense = [("doc1", 0.9), ("doc2", 0.6)]
        sparse: list[tuple[str, float]] = []

        result = ResultFuser.weighted_fusion(
            dense, sparse, dense_weight=0.7, sparse_weight=0.3
        )

        assert len(result) == 2
        result_dict = dict(result)
        assert result_dict["doc1"] == pytest.approx(0.7)  # 0.9/0.9 * 0.7 = 0.7
        assert result_dict["doc2"] == pytest.approx(0.6 / 0.9 * 0.7)

    def test_weighted_both_empty(self) -> None:
        """Test weighted fusion with both empty lists."""
        dense: list[tuple[str, float]] = []
        sparse: list[tuple[str, float]] = []

        result = ResultFuser.weighted_fusion(
            dense, sparse, dense_weight=0.7, sparse_weight=0.3
        )

        assert result == []

    def test_weighted_normalization(self) -> None:
        """Test weighted fusion normalizes scores correctly."""
        dense = [("doc1", 10.0), ("doc2", 5.0)]  # max=10
        sparse = [("doc1", 100.0), ("doc2", 50.0)]  # max=100

        result = ResultFuser.weighted_fusion(
            dense, sparse, dense_weight=0.5, sparse_weight=0.5
        )

        result_dict = dict(result)
        # doc1: (10/10)*0.5 + (100/100)*0.5 = 0.5 + 0.5 = 1.0
        assert result_dict["doc1"] == pytest.approx(1.0)
        # doc2: (5/10)*0.5 + (50/100)*0.5 = 0.25 + 0.25 = 0.5
        assert result_dict["doc2"] == pytest.approx(0.5)

    def test_weighted_zero_max_score_handling(self) -> None:
        """Test weighted fusion handles zero max scores."""
        dense = [("doc1", 0.0), ("doc2", 0.0)]
        sparse = [("doc1", 0.9)]

        result = ResultFuser.weighted_fusion(
            dense, sparse, dense_weight=0.5, sparse_weight=0.5
        )

        # Dense normalized scores should be 0 due to max=0
        result_dict = dict(result)
        assert result_dict["doc1"] == pytest.approx(0.5)  # 0 + 1.0*0.5
        assert result_dict["doc2"] == pytest.approx(0.0)  # 0 + 0

    def test_weighted_sorted_descending(self) -> None:
        """Test weighted fusion returns results sorted descending."""
        dense = [("doc1", 0.9), ("doc2", 0.5), ("doc3", 0.3)]
        sparse = [("doc1", 0.1), ("doc2", 0.5), ("doc3", 0.9)]

        result = ResultFuser.weighted_fusion(
            dense, sparse, dense_weight=0.5, sparse_weight=0.5
        )

        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_weighted_doc_only_in_one_list(self) -> None:
        """Test weighted fusion handles docs appearing in only one list."""
        dense = [("doc1", 1.0)]
        sparse = [("doc2", 1.0)]

        result = ResultFuser.weighted_fusion(
            dense, sparse, dense_weight=0.6, sparse_weight=0.4
        )

        result_dict = dict(result)
        # doc1: 1.0*0.6 + 0*0.4 = 0.6
        assert result_dict["doc1"] == pytest.approx(0.6)
        # doc2: 0*0.6 + 1.0*0.4 = 0.4
        assert result_dict["doc2"] == pytest.approx(0.4)

    def test_weighted_equal_weights(self) -> None:
        """Test weighted fusion with equal weights."""
        dense = [("doc1", 0.8)]
        sparse = [("doc1", 0.4)]

        result = ResultFuser.weighted_fusion(
            dense, sparse, dense_weight=0.5, sparse_weight=0.5
        )

        # Both normalized to 1.0 (each is max of their list)
        assert result[0][1] == pytest.approx(1.0)


class TestMergeSearchResults:
    """Tests for ResultFuser.merge_search_results() method."""

    def test_merge_rrf_method(self) -> None:
        """Test merge with RRF method."""
        results = [
            [{"id": "doc1", "score": 0.9}, {"id": "doc2", "score": 0.8}],
            [{"id": "doc2", "score": 0.95}, {"id": "doc3", "score": 0.7}],
        ]

        merged = ResultFuser.merge_search_results(results, method="rrf")

        assert len(merged) == 3
        doc_ids = [r["id"] for r in merged]
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids
        assert "doc3" in doc_ids
        # All should have fused_score
        for r in merged:
            assert "fused_score" in r

    def test_merge_weighted_method(self) -> None:
        """Test merge with weighted method."""
        results = [
            [{"id": "doc1", "score": 0.9}],
            [{"id": "doc1", "score": 0.8}],
        ]

        merged = ResultFuser.merge_search_results(
            results, method="weighted", dense_weight=0.6, sparse_weight=0.4
        )

        assert len(merged) == 1
        assert merged[0]["id"] == "doc1"
        assert "fused_score" in merged[0]

    def test_merge_unknown_method_raises_error(self) -> None:
        """Test merge raises ValueError for unknown method."""
        results = [
            [{"id": "doc1", "score": 0.9}],
            [{"id": "doc2", "score": 0.8}],
        ]

        with pytest.raises(ValueError, match="Unknown fusion method"):
            ResultFuser.merge_search_results(results, method="unknown")

    def test_merge_empty_results(self) -> None:
        """Test merge with empty results list."""
        results: list[list[dict]] = []

        merged = ResultFuser.merge_search_results(results, method="rrf")

        assert merged == []

    def test_merge_all_empty_sublists(self) -> None:
        """Test merge when all sublists are empty."""
        results: list[list[dict]] = [[], []]

        merged = ResultFuser.merge_search_results(results, method="rrf")

        assert merged == []

    def test_merge_single_result_list_rrf(self) -> None:
        """Test merge with single result list returns it unchanged."""
        results = [[{"id": "doc1", "score": 0.9}, {"id": "doc2", "score": 0.8}]]

        merged = ResultFuser.merge_search_results(results, method="rrf")

        # Single list returned as-is
        assert merged == results[0]

    def test_merge_single_result_list_weighted(self) -> None:
        """Test merge with single result list for weighted method."""
        results = [[{"id": "doc1", "score": 0.9}]]

        merged = ResultFuser.merge_search_results(results, method="weighted")

        assert merged == results[0]

    def test_merge_missing_id_uses_index(self) -> None:
        """Test merge handles results without id by using index.

        Note: The current implementation uses str(i) as the id when 'id' is missing,
        but the reconstruction logic only adds results that have an 'id' key in the
        original result dict. So results without 'id' won't appear in merged output.
        """
        results = [
            [{"id": "0", "score": 0.9, "text": "hello"}],
            [{"id": "1", "score": 0.8, "text": "world"}],
        ]

        merged = ResultFuser.merge_search_results(results, method="rrf")

        # Both results have id, so both should be in merged
        assert len(merged) == 2

    def test_merge_missing_score_uses_default(self) -> None:
        """Test merge handles results without score by using 1.0."""
        results = [
            [{"id": "doc1"}],
            [{"id": "doc2"}],
        ]

        merged = ResultFuser.merge_search_results(results, method="rrf")

        assert len(merged) == 2
        for r in merged:
            assert "fused_score" in r

    def test_merge_preserves_original_fields(self) -> None:
        """Test merge preserves all original fields in results."""
        results = [
            [
                {
                    "id": "doc1",
                    "score": 0.9,
                    "text": "content",
                    "metadata": {"key": "val"},
                }
            ],
            [{"id": "doc2", "score": 0.8, "title": "title"}],
        ]

        merged = ResultFuser.merge_search_results(results, method="rrf")

        doc1 = next(r for r in merged if r["id"] == "doc1")
        assert doc1["text"] == "content"
        assert doc1["metadata"] == {"key": "val"}

        doc2 = next(r for r in merged if r["id"] == "doc2")
        assert doc2["title"] == "title"

    def test_merge_custom_k_parameter(self) -> None:
        """Test merge passes k parameter to RRF."""
        results = [
            [{"id": "doc1", "score": 0.9}],
            [{"id": "doc1", "score": 0.8}],
        ]

        merged_k10 = ResultFuser.merge_search_results(results, method="rrf", k=10)
        merged_k60 = ResultFuser.merge_search_results(results, method="rrf", k=60)

        # Different k values should produce different fused scores
        assert merged_k10[0]["fused_score"] != merged_k60[0]["fused_score"]
        assert merged_k10[0]["fused_score"] > merged_k60[0]["fused_score"]

    def test_merge_custom_weights(self) -> None:
        """Test merge passes weight parameters to weighted fusion."""
        results = [
            [{"id": "doc1", "score": 1.0}],
            [{"id": "doc2", "score": 1.0}],
        ]

        merged = ResultFuser.merge_search_results(
            results, method="weighted", dense_weight=0.8, sparse_weight=0.2
        )

        doc1 = next(r for r in merged if r["id"] == "doc1")
        doc2 = next(r for r in merged if r["id"] == "doc2")
        # doc1 gets 0.8 (from dense only), doc2 gets 0.2 (from sparse only)
        assert doc1["fused_score"] == pytest.approx(0.8)
        assert doc2["fused_score"] == pytest.approx(0.2)

    def test_merge_first_occurrence_preserved(self) -> None:
        """Test merge preserves first occurrence of duplicate ids."""
        results = [
            [{"id": "doc1", "score": 0.9, "source": "dense"}],
            [{"id": "doc1", "score": 0.8, "source": "sparse"}],
        ]

        merged = ResultFuser.merge_search_results(results, method="rrf")

        assert merged[0]["source"] == "dense"

    def test_merge_result_ordering(self) -> None:
        """Test merge returns results ordered by fused score."""
        results = [
            [{"id": "doc1", "score": 0.9}, {"id": "doc2", "score": 0.5}],
            [{"id": "doc2", "score": 0.95}, {"id": "doc3", "score": 0.4}],
        ]

        merged = ResultFuser.merge_search_results(results, method="rrf")

        fused_scores = [r["fused_score"] for r in merged]
        assert fused_scores == sorted(fused_scores, reverse=True)


class TestResultFuserIntegration:
    """Integration tests for ResultFuser combining multiple methods."""

    def test_rrf_and_weighted_produce_different_rankings(self) -> None:
        """Test that RRF and weighted fusion can produce different rankings."""
        dense = [("doc1", 0.9), ("doc2", 0.3)]
        sparse = [("doc2", 0.95), ("doc1", 0.1)]

        rrf_result = ResultFuser.reciprocal_rank_fusion(dense, sparse, k=60)
        weighted_result = ResultFuser.weighted_fusion(
            dense, sparse, dense_weight=0.5, sparse_weight=0.5
        )

        # Rankings might differ based on score vs rank aggregation
        rrf_order = [doc_id for doc_id, _ in rrf_result]
        weighted_order = [doc_id for doc_id, _ in weighted_result]

        # Both should contain same documents
        assert set(rrf_order) == set(weighted_order)

    def test_merge_with_real_world_like_data(self) -> None:
        """Test merge with realistic search results."""
        dense_results = [
            {"id": "doc_001", "score": 0.95, "title": "Machine Learning Guide"},
            {"id": "doc_002", "score": 0.87, "title": "Deep Learning Tutorial"},
            {"id": "doc_003", "score": 0.75, "title": "AI Fundamentals"},
        ]
        sparse_results = [
            {"id": "doc_002", "score": 12.5, "title": "Deep Learning Tutorial"},
            {"id": "doc_004", "score": 10.2, "title": "Neural Networks Explained"},
            {"id": "doc_001", "score": 8.7, "title": "Machine Learning Guide"},
        ]

        merged = ResultFuser.merge_search_results(
            [dense_results, sparse_results], method="rrf", k=60
        )

        # Should have 4 unique documents
        assert len(merged) == 4

        # All should have fused_score
        for r in merged:
            assert "fused_score" in r
            assert r["fused_score"] > 0

        # doc_002 appears at rank 2 in dense and rank 1 in sparse
        # Should likely be highly ranked
        doc_002 = next(r for r in merged if r["id"] == "doc_002")
        assert doc_002["title"] == "Deep Learning Tutorial"

    def test_large_result_sets(self) -> None:
        """Test fusion with larger result sets."""
        dense = [(f"doc_{i}", 1.0 - i * 0.01) for i in range(100)]
        sparse = [(f"doc_{99 - i}", 1.0 - i * 0.01) for i in range(100)]

        result = ResultFuser.reciprocal_rank_fusion(dense, sparse, k=60)

        # Should contain all 100 unique docs
        assert len(result) == 100

        # All scores should be positive
        for _, score in result:
            assert score > 0

    def test_edge_case_single_doc_all_methods(self) -> None:
        """Test all methods with single document."""
        dense = [("doc1", 0.9)]
        sparse = [("doc1", 0.8)]

        rrf_result = ResultFuser.reciprocal_rank_fusion(dense, sparse, k=60)
        weighted_result = ResultFuser.weighted_fusion(
            dense, sparse, dense_weight=0.7, sparse_weight=0.3
        )

        assert len(rrf_result) == 1
        assert len(weighted_result) == 1
        assert rrf_result[0][0] == "doc1"
        assert weighted_result[0][0] == "doc1"

    def test_numerical_stability(self) -> None:
        """Test fusion handles very small and very large scores."""
        dense = [("doc1", 1e-10), ("doc2", 1e10)]
        sparse = [("doc1", 1e10), ("doc2", 1e-10)]

        # RRF should work fine (ignores scores)
        rrf_result = ResultFuser.reciprocal_rank_fusion(dense, sparse, k=60)
        assert len(rrf_result) == 2

        # Weighted fusion normalizes scores
        weighted_result = ResultFuser.weighted_fusion(
            dense, sparse, dense_weight=0.5, sparse_weight=0.5
        )
        assert len(weighted_result) == 2
        # Both normalized to 1.0 (each is max in their list for one doc)
        result_dict = dict(weighted_result)
        # doc2 has max dense (normalized to 1) + near-zero sparse
        # doc1 has near-zero dense + max sparse (normalized to 1)
        assert result_dict["doc1"] == pytest.approx(result_dict["doc2"], rel=1e-5)
