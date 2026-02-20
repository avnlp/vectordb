"""Tests for filter utilities."""

from haystack import Document

from vectordb.haystack.utils import DocumentFilter


class TestDocumentFilter:
    """Tests for DocumentFilter class."""

    def test_normalize_none(self) -> None:
        """Test normalize with None returns None."""
        result = DocumentFilter.normalize(None)
        assert result is None

    def test_normalize_empty_dict(self) -> None:
        """Test normalize with empty dict returns None."""
        result = DocumentFilter.normalize({})
        assert result is None

    def test_normalize_dict(self) -> None:
        """Test normalize with dict."""
        filters = {"source": "wiki", "year": 2023}
        result = DocumentFilter.normalize(filters)
        assert result == filters

    def test_apply_no_filters(self, sample_documents: list[Document]) -> None:
        """Test filtering with no filters returns all documents."""
        result = DocumentFilter.apply(sample_documents, {})
        assert len(result) == len(sample_documents)

    def test_apply_exact_match(self, sample_documents: list[Document]) -> None:
        """Test exact match filtering."""
        filters = {"source": "wiki"}
        result = DocumentFilter.apply(sample_documents, filters)
        assert len(result) == 1
        assert result[0].meta["source"] == "wiki"

    def test_apply_multiple_filters(self, sample_documents: list[Document]) -> None:
        """Test multiple filter conditions."""
        filters = {"source": "wiki", "id": "1"}
        result = DocumentFilter.apply(sample_documents, filters)
        assert len(result) == 1
        assert result[0].meta["id"] == "1"

    def test_apply_no_match(self, sample_documents: list[Document]) -> None:
        """Test filtering with no matching documents."""
        filters = {"source": "nonexistent"}
        result = DocumentFilter.apply(sample_documents, filters)
        assert len(result) == 0

    def test_apply_gt_operator(self, sample_documents: list[Document]) -> None:
        """Test $gt (greater than) operator."""
        docs = [
            Document(content="Doc 1", meta={"year": 2020}),
            Document(content="Doc 2", meta={"year": 2022}),
            Document(content="Doc 3", meta={"year": 2024}),
        ]
        filters = {"year": {"$gt": 2021}}
        result = DocumentFilter.apply(docs, filters)
        assert len(result) == 2
        assert all(doc.meta["year"] > 2021 for doc in result)

    def test_apply_gte_operator(self, sample_documents: list[Document]) -> None:
        """Test $gte (greater than or equal) operator."""
        docs = [
            Document(content="Doc 1", meta={"year": 2020}),
            Document(content="Doc 2", meta={"year": 2022}),
            Document(content="Doc 3", meta={"year": 2024}),
        ]
        filters = {"year": {"$gte": 2022}}
        result = DocumentFilter.apply(docs, filters)
        assert len(result) == 2

    def test_apply_lt_operator(self, sample_documents: list[Document]) -> None:
        """Test $lt (less than) operator."""
        docs = [
            Document(content="Doc 1", meta={"year": 2020}),
            Document(content="Doc 2", meta={"year": 2022}),
            Document(content="Doc 3", meta={"year": 2024}),
        ]
        filters = {"year": {"$lt": 2023}}
        result = DocumentFilter.apply(docs, filters)
        assert len(result) == 2

    def test_apply_lte_operator(self, sample_documents: list[Document]) -> None:
        """Test $lte (less than or equal) operator."""
        docs = [
            Document(content="Doc 1", meta={"year": 2020}),
            Document(content="Doc 2", meta={"year": 2022}),
            Document(content="Doc 3", meta={"year": 2024}),
        ]
        filters = {"year": {"$lte": 2022}}
        result = DocumentFilter.apply(docs, filters)
        assert len(result) == 2

    def test_apply_ne_operator(self, sample_documents: list[Document]) -> None:
        """Test $ne (not equal) operator."""
        docs = [
            Document(content="Doc 1", meta={"source": "wiki"}),
            Document(content="Doc 2", meta={"source": "paper"}),
            Document(content="Doc 3", meta={"source": "wiki"}),
        ]
        filters = {"source": {"$ne": "wiki"}}
        result = DocumentFilter.apply(docs, filters)
        assert len(result) == 1
        assert result[0].meta["source"] == "paper"

    def test_apply_in_operator(self, sample_documents: list[Document]) -> None:
        """Test $in operator."""
        docs = [
            Document(content="Doc 1", meta={"source": "wiki"}),
            Document(content="Doc 2", meta={"source": "paper"}),
            Document(content="Doc 3", meta={"source": "blog"}),
        ]
        filters = {"source": {"$in": ["wiki", "paper"]}}
        result = DocumentFilter.apply(docs, filters)
        assert len(result) == 2

    def test_apply_nin_operator(self, sample_documents: list[Document]) -> None:
        """Test $nin (not in) operator."""
        docs = [
            Document(content="Doc 1", meta={"source": "wiki"}),
            Document(content="Doc 2", meta={"source": "paper"}),
            Document(content="Doc 3", meta={"source": "blog"}),
        ]
        filters = {"source": {"$nin": ["wiki", "paper"]}}
        result = DocumentFilter.apply(docs, filters)
        assert len(result) == 1
        assert result[0].meta["source"] == "blog"

    def test_apply_with_missing_metadata(
        self, sample_documents: list[Document]
    ) -> None:
        """Test filtering documents with missing metadata."""
        docs = [
            Document(content="Doc 1", meta={"source": "wiki"}),
            Document(content="Doc 2", meta={}),
            Document(content="Doc 3", meta={"source": "paper"}),
        ]
        filters = {"source": "wiki"}
        result = DocumentFilter.apply(docs, filters)
        assert len(result) == 1

    def test_apply_none_metadata(self) -> None:
        """Test filtering documents with None metadata."""
        docs = [
            Document(content="Doc 1", meta={"source": "wiki"}),
            Document(content="Doc 2", meta=None),
        ]
        filters = {"source": "wiki"}
        result = DocumentFilter.apply(docs, filters)
        assert len(result) == 1
