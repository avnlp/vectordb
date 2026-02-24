"""Tests for result fusion utilities."""

import hashlib

from haystack import Document

from vectordb.haystack.query_enhancement.utils.fusion import (
    deduplicate_by_content,
    rrf_fusion_many,
    stable_doc_id,
)


class TestStableDocId:
    """Tests for stable_doc_id function."""

    def test_returns_doc_id_from_meta(self) -> None:
        """Should return doc.meta['doc_id'] when set."""
        doc = Document(content="test content", meta={"doc_id": "custom-id-123"})
        assert stable_doc_id(doc) == "custom-id-123"

    def test_returns_doc_id_when_meta_doc_id_is_integer(self) -> None:
        """Should convert non-string doc_id to string."""
        doc = Document(content="test content", meta={"doc_id": 42})
        assert stable_doc_id(doc) == "42"

    def test_fallback_to_doc_id(self) -> None:
        """Should return doc.id when meta doc_id not set."""
        doc = Document(content="test content", id="fallback-id")
        assert stable_doc_id(doc) == "fallback-id"

    def test_fallback_to_doc_id_when_meta_empty(self) -> None:
        """Should return doc.id when meta is empty dict."""
        doc = Document(content="test content", id="fallback-id", meta={})
        assert stable_doc_id(doc) == "fallback-id"

    def test_hash_content_when_no_ids(self) -> None:
        """Should hash content when neither doc_id nor id set."""
        doc = Document(content="test content")
        # Manually clear the id to simulate no id scenario
        doc.id = ""
        result = stable_doc_id(doc)
        expected = hashlib.sha1(
            "test content".encode(), usedforsecurity=False
        ).hexdigest()
        assert result == expected

    def test_empty_content_hashes(self) -> None:
        """Should hash empty content when no ids and content is empty."""
        doc = Document(content="")
        doc.id = ""
        result = stable_doc_id(doc)
        expected = hashlib.sha1(b"", usedforsecurity=False).hexdigest()
        assert result == expected

    def test_none_content_hashes_as_empty(self) -> None:
        """Should handle None content gracefully."""
        doc = Document(content=None)  # type: ignore[arg-type]
        doc.id = ""
        result = stable_doc_id(doc)
        expected = hashlib.sha1(b"", usedforsecurity=False).hexdigest()
        assert result == expected

    def test_whitespace_content_normalizes(self) -> None:
        """Should normalize whitespace in content before hashing."""
        doc1 = Document(content="  test content  ")
        doc1.id = ""
        doc2 = Document(content="test content")
        doc2.id = ""
        # Both should produce same hash after strip().lower()
        assert stable_doc_id(doc1) == stable_doc_id(doc2)

    def test_case_normalization(self) -> None:
        """Should normalize case in content before hashing."""
        doc1 = Document(content="TEST CONTENT")
        doc1.id = ""
        doc2 = Document(content="test content")
        doc2.id = ""
        assert stable_doc_id(doc1) == stable_doc_id(doc2)

    def test_special_characters_in_content(self) -> None:
        """Should handle special characters in content."""
        doc = Document(content="Test with émojis 🎉 and spëcial chars!")
        doc.id = ""
        result = stable_doc_id(doc)
        expected_content = "test with émojis 🎉 and spëcial chars!"
        expected = hashlib.sha1(
            expected_content.encode(), usedforsecurity=False
        ).hexdigest()
        assert result == expected

    def test_deterministic_hashing(self) -> None:
        """Same document should produce same hash across multiple calls."""
        doc = Document(content="deterministic test")
        doc.id = ""
        results = [stable_doc_id(doc) for _ in range(10)]
        assert len(set(results)) == 1  # All should be identical

    def test_priority_meta_over_id(self) -> None:
        """meta['doc_id'] should take priority over doc.id."""
        doc = Document(
            content="test", id="regular-id", meta={"doc_id": "meta-priority-id"}
        )
        assert stable_doc_id(doc) == "meta-priority-id"


class TestRrfFusionMany:
    """Tests for rrf_fusion_many function."""

    def test_empty_ranked_lists(self) -> None:
        """Should return empty list for empty input."""
        assert rrf_fusion_many([]) == []

    def test_single_list_passthrough(self) -> None:
        """Should return documents from single list."""
        docs = [
            Document(content="doc1", meta={"doc_id": "1"}),
            Document(content="doc2", meta={"doc_id": "2"}),
        ]
        result = rrf_fusion_many([docs])
        assert len(result) == 2
        assert result[0].meta["doc_id"] == "1"
        assert result[1].meta["doc_id"] == "2"

    def test_two_lists_fusion(self) -> None:
        """Should fuse two ranked lists correctly."""
        list1 = [
            Document(content="doc1", meta={"doc_id": "1"}),
            Document(content="doc2", meta={"doc_id": "2"}),
        ]
        list2 = [
            Document(content="doc2", meta={"doc_id": "2"}),
            Document(content="doc3", meta={"doc_id": "3"}),
        ]
        result = rrf_fusion_many([list1, list2])
        # doc2 appears in both lists, should have highest score
        assert result[0].meta["doc_id"] == "2"

    def test_multiple_lists_fusion(self) -> None:
        """Should fuse 3+ ranked lists correctly."""
        list1 = [Document(content="doc1", meta={"doc_id": "1"})]
        list2 = [Document(content="doc2", meta={"doc_id": "2"})]
        list3 = [Document(content="doc1", meta={"doc_id": "1"})]
        list4 = [Document(content="doc1", meta={"doc_id": "1"})]
        result = rrf_fusion_many([list1, list2, list3, list4])
        # doc1 appears in 3 lists, doc2 in 1
        assert result[0].meta["doc_id"] == "1"
        assert result[1].meta["doc_id"] == "2"

    def test_non_overlapping_lists(self) -> None:
        """Should handle lists with no overlapping documents."""
        list1 = [Document(content="doc1", meta={"doc_id": "1"})]
        list2 = [Document(content="doc2", meta={"doc_id": "2"})]
        result = rrf_fusion_many([list1, list2])
        # Both have same RRF score (1/61), order depends on iteration
        assert len(result) == 2
        doc_ids = {r.meta["doc_id"] for r in result}
        assert doc_ids == {"1", "2"}

    def test_custom_k_value(self) -> None:
        """Should use custom k value in RRF calculation."""
        list1 = [
            Document(content="doc1", meta={"doc_id": "1"}),
            Document(content="doc2", meta={"doc_id": "2"}),
        ]
        list2 = [
            Document(content="doc2", meta={"doc_id": "2"}),
            Document(content="doc1", meta={"doc_id": "1"}),
        ]
        # With k=0, rank differences matter more
        result_k0 = rrf_fusion_many([list1, list2], k=0)
        # Both docs appear at rank 1 and 2, scores should be equal
        # doc1: 1/1 + 1/2 = 1.5, doc2: 1/1 + 1/2 = 1.5
        assert len(result_k0) == 2

    def test_top_k_truncation(self) -> None:
        """Should truncate results to top_k."""
        docs = [Document(content=f"doc{i}", meta={"doc_id": str(i)}) for i in range(10)]
        result = rrf_fusion_many([docs], top_k=3)
        assert len(result) == 3

    def test_top_k_none_returns_all_unique_docs(self) -> None:
        """top_k=None should return all unique documents."""
        list1 = [Document(content=f"doc{i}", meta={"doc_id": str(i)}) for i in range(5)]
        list2 = [
            Document(content=f"doc{i}", meta={"doc_id": str(i)}) for i in range(10, 13)
        ]
        result = rrf_fusion_many([list1, list2], top_k=None)
        # All unique documents (5 + 3 = 8)
        assert len(result) == 8

    def test_score_calculation_accuracy(self) -> None:
        """Should calculate RRF scores correctly."""
        # Create a scenario where we can predict the winner
        # doc1 at rank 1 in two lists: score = 2 * 1/(60+1) = 2/61
        # doc2 at rank 1 in one list: score = 1/(60+1) = 1/61
        list1 = [Document(content="doc1", meta={"doc_id": "1"})]
        list2 = [Document(content="doc1", meta={"doc_id": "1"})]
        list3 = [Document(content="doc2", meta={"doc_id": "2"})]
        result = rrf_fusion_many([list1, list2, list3])
        assert result[0].meta["doc_id"] == "1"
        assert result[1].meta["doc_id"] == "2"

    def test_document_ordering_highest_score_first(self) -> None:
        """Should order documents by highest RRF score first."""
        # doc1: appears at rank 1 in all 3 lists
        # doc2: appears at rank 2 in all 3 lists
        # doc3: appears at rank 3 in all 3 lists
        list1 = [
            Document(content="doc1", meta={"doc_id": "1"}),
            Document(content="doc2", meta={"doc_id": "2"}),
            Document(content="doc3", meta={"doc_id": "3"}),
        ]
        list2 = [
            Document(content="doc1", meta={"doc_id": "1"}),
            Document(content="doc2", meta={"doc_id": "2"}),
            Document(content="doc3", meta={"doc_id": "3"}),
        ]
        list3 = [
            Document(content="doc1", meta={"doc_id": "1"}),
            Document(content="doc2", meta={"doc_id": "2"}),
            Document(content="doc3", meta={"doc_id": "3"}),
        ]
        result = rrf_fusion_many([list1, list2, list3])
        assert [r.meta["doc_id"] for r in result] == ["1", "2", "3"]

    def test_accumulates_scores_across_lists(self) -> None:
        """Document appearing in multiple lists should accumulate scores."""
        # doc_common appears in both lists at rank 1
        # doc_unique1 appears only in list1 at rank 2
        # doc_unique2 appears only in list2 at rank 2
        list1 = [
            Document(content="common", meta={"doc_id": "common"}),
            Document(content="unique1", meta={"doc_id": "unique1"}),
        ]
        list2 = [
            Document(content="common", meta={"doc_id": "common"}),
            Document(content="unique2", meta={"doc_id": "unique2"}),
        ]
        result = rrf_fusion_many([list1, list2])
        # common should be first due to accumulated score
        assert result[0].meta["doc_id"] == "common"

    def test_keeps_first_occurrence_of_document(self) -> None:
        """Should keep first occurrence when document appears in multiple lists."""
        doc1_v1 = Document(
            content="doc1", meta={"doc_id": "1", "source": "list1", "extra": "v1"}
        )
        doc1_v2 = Document(
            content="doc1", meta={"doc_id": "1", "source": "list2", "extra": "v2"}
        )
        result = rrf_fusion_many([[doc1_v1], [doc1_v2]])
        assert result[0].meta["source"] == "list1"
        assert result[0].meta["extra"] == "v1"

    def test_five_lists_fusion(self) -> None:
        """Should handle 5 ranked lists."""
        lists = [
            [Document(content=f"doc{i}", meta={"doc_id": str(i)})] for i in range(5)
        ]
        result = rrf_fusion_many(lists)
        assert len(result) == 5


class TestDeduplicateByContent:
    """Tests for deduplicate_by_content function."""

    def test_empty_list(self) -> None:
        """Should return empty list for empty input."""
        assert deduplicate_by_content([]) == []

    def test_single_document(self) -> None:
        """Should return single document unchanged."""
        doc = Document(content="test", meta={"doc_id": "1"})
        result = deduplicate_by_content([doc])
        assert len(result) == 1
        assert result[0] == doc

    def test_no_duplicates(self) -> None:
        """Should keep all documents when no duplicates."""
        docs = [
            Document(content="doc1", meta={"doc_id": "1"}),
            Document(content="doc2", meta={"doc_id": "2"}),
            Document(content="doc3", meta={"doc_id": "3"}),
        ]
        result = deduplicate_by_content(docs)
        assert len(result) == 3

    def test_removes_duplicates_by_doc_id(self) -> None:
        """Should remove duplicates based on stable_doc_id."""
        docs = [
            Document(content="doc1", meta={"doc_id": "1"}),
            Document(content="doc1 different content", meta={"doc_id": "1"}),
            Document(content="doc2", meta={"doc_id": "2"}),
        ]
        result = deduplicate_by_content(docs)
        assert len(result) == 2
        doc_ids = [r.meta["doc_id"] for r in result]
        assert doc_ids == ["1", "2"]

    def test_removes_duplicates_by_content_hash(self) -> None:
        """Should remove duplicates when content hashes match."""
        doc1 = Document(content="same content")
        doc1.id = ""
        doc2 = Document(content="same content")
        doc2.id = ""
        doc3 = Document(content="different content")
        doc3.id = ""
        result = deduplicate_by_content([doc1, doc2, doc3])
        assert len(result) == 2

    def test_preserves_order_first_occurrence(self) -> None:
        """Should preserve order and keep first occurrence."""
        docs = [
            Document(content="first", meta={"doc_id": "1", "order": "first"}),
            Document(content="second", meta={"doc_id": "2", "order": "second"}),
            Document(content="first duplicate", meta={"doc_id": "1", "order": "third"}),
        ]
        result = deduplicate_by_content(docs)
        assert len(result) == 2
        assert result[0].meta["order"] == "first"
        assert result[1].meta["order"] == "second"

    def test_deduplication_by_stable_doc_id(self) -> None:
        """Should use stable_doc_id for deduplication."""
        # Test with doc.id fallback
        doc1 = Document(content="content1", id="same-id")
        doc2 = Document(content="content2", id="same-id")
        result = deduplicate_by_content([doc1, doc2])
        assert len(result) == 1
        assert result[0].content == "content1"

    def test_varying_ids_and_content(self) -> None:
        """Should handle documents with varying id configurations."""
        doc1 = Document(content="content1", meta={"doc_id": "meta-id"})
        doc2 = Document(content="content2", id="regular-id")
        doc3 = Document(content="content3")
        doc3.id = ""  # Will use content hash
        doc4 = Document(content="content3")  # Different id but same content as doc3
        doc4.id = ""  # Will hash to same value

        result = deduplicate_by_content([doc1, doc2, doc3, doc4])
        assert len(result) == 3  # doc4 is duplicate of doc3

    def test_all_duplicates(self) -> None:
        """Should return single document when all are duplicates."""
        docs = [
            Document(content="same", meta={"doc_id": "1"}),
            Document(content="same", meta={"doc_id": "1"}),
            Document(content="same", meta={"doc_id": "1"}),
        ]
        result = deduplicate_by_content(docs)
        assert len(result) == 1

    def test_whitespace_normalized_duplicates(self) -> None:
        """Should treat whitespace-normalized content as duplicates."""
        doc1 = Document(content="  test  content  ")
        doc1.id = ""
        doc2 = Document(content="test content")
        doc2.id = ""
        result = deduplicate_by_content([doc1, doc2])
        assert len(result) == 1
