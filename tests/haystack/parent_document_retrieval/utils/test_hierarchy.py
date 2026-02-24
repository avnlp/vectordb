"""Tests for document hierarchy utilities in parent document retrieval."""

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document

from vectordb.haystack.parent_document_retrieval.utils.hierarchy import (
    create_hierarchy,
    filter_by_level,
)


class TestCreateHierarchy:
    """Test suite for create_hierarchy function."""

    @patch(
        "vectordb.haystack.parent_document_retrieval.utils.hierarchy.HierarchicalDocumentSplitter"
    )
    def test_create_hierarchy_default_parameters(
        self, mock_splitter_class: MagicMock, sample_documents: list[Document]
    ) -> None:
        """Test create_hierarchy with default parameters."""
        mock_splitter = MagicMock()
        mock_splitter_class.return_value = mock_splitter

        # Mock return documents with hierarchical metadata
        mock_docs = [
            Document(
                content="Parent 1",
                meta={"level": 1, "doc_idx": 0, "children_ids": ["child1", "child2"]},
            ),
            Document(content="Child 1", meta={"level": 2, "doc_idx": 0}),
            Document(content="Child 2", meta={"level": 2, "doc_idx": 0}),
            Document(
                content="Parent 2",
                meta={"level": 1, "doc_idx": 1, "children_ids": ["child3"]},
            ),
            Document(content="Child 3", meta={"level": 2, "doc_idx": 1}),
        ]
        mock_splitter.run.return_value = {"documents": mock_docs}

        result = create_hierarchy(sample_documents)

        # Verify splitter was called with correct parameters
        mock_splitter_class.assert_called_once_with(
            block_sizes={100, 25},
            split_overlap=5,
            split_by="word",
        )
        mock_splitter.run.assert_called_once_with(sample_documents)

        # Verify result structure
        assert "parents" in result
        assert "leaves" in result
        assert len(result["parents"]) == 2
        assert len(result["leaves"]) == 3  # All child docs are leaves

    @patch(
        "vectordb.haystack.parent_document_retrieval.utils.hierarchy.HierarchicalDocumentSplitter"
    )
    def test_create_hierarchy_custom_parameters(
        self, mock_splitter_class: MagicMock, sample_documents: list[Document]
    ) -> None:
        """Test create_hierarchy with custom parameters."""
        mock_splitter = MagicMock()
        mock_splitter_class.return_value = mock_splitter
        mock_splitter.run.return_value = {"documents": []}

        create_hierarchy(
            sample_documents,
            parent_size_words=150,
            child_size_words=30,
            split_overlap=10,
        )

        mock_splitter_class.assert_called_once_with(
            block_sizes={150, 30},
            split_overlap=10,
            split_by="word",
        )

    @patch(
        "vectordb.haystack.parent_document_retrieval.utils.hierarchy.HierarchicalDocumentSplitter"
    )
    def test_create_hierarchy_single_document(
        self, mock_splitter_class: MagicMock, sample_documents: list[Document]
    ) -> None:
        """Test create_hierarchy with a single document."""
        mock_splitter = MagicMock()
        mock_splitter_class.return_value = mock_splitter

        single_doc = [sample_documents[0]]
        mock_docs = [
            Document(content="Parent", meta={"level": 1, "children_ids": ["child1"]}),
            Document(content="Child", meta={"level": 2}),
        ]
        mock_splitter.run.return_value = {"documents": mock_docs}

        result = create_hierarchy(single_doc)

        assert len(result["parents"]) == 1
        assert len(result["leaves"]) == 1

    @patch(
        "vectordb.haystack.parent_document_retrieval.utils.hierarchy.HierarchicalDocumentSplitter"
    )
    def test_create_hierarchy_empty_documents_list(
        self, mock_splitter_class: MagicMock, empty_documents_list: list[Document]
    ) -> None:
        """Test create_hierarchy with empty documents list."""
        mock_splitter = MagicMock()
        mock_splitter_class.return_value = mock_splitter
        mock_splitter.run.return_value = {"documents": []}

        result = create_hierarchy(empty_documents_list)

        assert result["parents"] == []
        assert result["leaves"] == []
        mock_splitter.run.assert_called_once_with(empty_documents_list)

    @patch(
        "vectordb.haystack.parent_document_retrieval.utils.hierarchy.HierarchicalDocumentSplitter"
    )
    def test_create_hierarchy_parent_level_identification(
        self, mock_splitter_class: MagicMock, sample_documents: list[Document]
    ) -> None:
        """Test that parent documents are correctly identified by level=1."""
        mock_splitter = MagicMock()
        mock_splitter_class.return_value = mock_splitter

        mock_docs = [
            Document(content="Parent 1", meta={"level": 1, "doc_idx": 0}),
            Document(content="Parent 2", meta={"level": 1, "doc_idx": 1}),
            Document(content="Child 1", meta={"level": 2, "doc_idx": 0}),
            Document(content="Child 2", meta={"level": 2, "doc_idx": 0}),
        ]
        mock_splitter.run.return_value = {"documents": mock_docs}

        result = create_hierarchy(sample_documents)

        # Should only include documents with level=1 as parents
        assert len(result["parents"]) == 2
        for parent in result["parents"]:
            assert parent.meta.get("level") == 1

    @patch(
        "vectordb.haystack.parent_document_retrieval.utils.hierarchy.HierarchicalDocumentSplitter"
    )
    def test_create_hierarchy_leaf_identification(
        self, mock_splitter_class: MagicMock, sample_documents: list[Document]
    ) -> None:
        """Test that leaf documents are correctly identified by missing children_ids."""
        mock_splitter = MagicMock()
        mock_splitter_class.return_value = mock_splitter

        mock_docs = [
            Document(content="Parent", meta={"level": 1, "children_ids": ["child1"]}),
            Document(content="Child", meta={"level": 2}),  # No children_ids
            Document(content="Grandchild", meta={"level": 3}),  # No children_ids
        ]
        mock_splitter.run.return_value = {"documents": mock_docs}

        result = create_hierarchy(sample_documents)

        # Should include documents without children_ids as leaves
        assert len(result["leaves"]) == 2
        for leaf in result["leaves"]:
            assert not leaf.meta.get("children_ids")

    @patch(
        "vectordb.haystack.parent_document_retrieval.utils.hierarchy.HierarchicalDocumentSplitter"
    )
    def test_create_hierarchy_documents_without_meta(
        self, mock_splitter_class: MagicMock, sample_documents: list[Document]
    ) -> None:
        """Test create_hierarchy with documents missing metadata.

        Documents without level/children_ids won't be classified as parents.
        """
        mock_splitter = MagicMock()
        mock_splitter_class.return_value = mock_splitter

        mock_docs = [
            Document(content="Doc without meta"),  # No meta dict
            Document(content="Doc with empty meta", meta={}),  # Empty meta dict
        ]
        mock_splitter.run.return_value = {"documents": mock_docs}

        result = create_hierarchy(sample_documents)

        # Should handle gracefully - documents won't be classified as parents
        assert len(result["parents"]) == 0
        assert len(result["leaves"]) == 2  # Both docs have no children_ids

    @patch(
        "vectordb.haystack.parent_document_retrieval.utils.hierarchy.HierarchicalDocumentSplitter"
    )
    def test_create_hierarchy_parameter_validation(
        self, mock_splitter_class: MagicMock, sample_documents: list[Document]
    ) -> None:
        """Test create_hierarchy with various parameter values."""
        mock_splitter = MagicMock()
        mock_splitter_class.return_value = mock_splitter
        mock_splitter.run.return_value = {"documents": []}

        # Test with parent_size_words smaller than child_size_words
        create_hierarchy(
            sample_documents,
            parent_size_words=25,
            child_size_words=50,
            split_overlap=5,
        )

        mock_splitter_class.assert_called_with(
            block_sizes={25, 50},
            split_overlap=5,
            split_by="word",
        )

    @patch(
        "vectordb.haystack.parent_document_retrieval.utils.hierarchy.HierarchicalDocumentSplitter"
    )
    def test_create_hierarchy_large_overlap(
        self, mock_splitter_class: MagicMock, sample_documents: list[Document]
    ) -> None:
        """Test create_hierarchy with large split_overlap value."""
        mock_splitter = MagicMock()
        mock_splitter_class.return_value = mock_splitter
        mock_splitter.run.return_value = {"documents": []}

        create_hierarchy(
            sample_documents,
            parent_size_words=100,
            child_size_words=25,
            split_overlap=50,  # Large overlap
        )

        mock_splitter_class.assert_called_with(
            block_sizes={100, 25},
            split_overlap=50,
            split_by="word",
        )

    @patch(
        "vectordb.haystack.parent_document_retrieval.utils.hierarchy.HierarchicalDocumentSplitter"
    )
    def test_create_hierarchy_zero_overlap(
        self, mock_splitter_class: MagicMock, sample_documents: list[Document]
    ) -> None:
        """Test create_hierarchy with zero split_overlap."""
        mock_splitter = MagicMock()
        mock_splitter_class.return_value = mock_splitter
        mock_splitter.run.return_value = {"documents": []}

        create_hierarchy(
            sample_documents,
            split_overlap=0,
        )

        mock_splitter_class.assert_called_with(
            block_sizes={100, 25},
            split_overlap=0,
            split_by="word",
        )


class TestFilterByLevel:
    """Test suite for filter_by_level function."""

    def test_filter_by_level_parents(
        self, sample_hierarchical_documents: list[Document]
    ) -> None:
        """Test filtering for parent documents (level=1)."""
        result = filter_by_level(sample_hierarchical_documents, 1)

        assert len(result) == 2
        for doc in result:
            assert doc.meta.get("level") == 1

    def test_filter_by_level_children(
        self, sample_hierarchical_documents: list[Document]
    ) -> None:
        """Test filtering for child documents (level=2)."""
        result = filter_by_level(sample_hierarchical_documents, 2)

        assert len(result) == 2
        for doc in result:
            assert doc.meta.get("level") == 2

    def test_filter_by_level_nonexistent_level(
        self, sample_hierarchical_documents: list[Document]
    ) -> None:
        """Test filtering for level that doesn't exist in documents."""
        result = filter_by_level(sample_hierarchical_documents, 3)

        assert result == []

    def test_filter_by_level_mixed_documents(
        self, sample_hierarchical_documents: list[Document]
    ) -> None:
        """Test filtering with mixed level documents."""
        # Add a document with different level
        sample_hierarchical_documents.append(
            Document(content="Level 3 doc", meta={"level": 3})
        )

        level_1_docs = filter_by_level(sample_hierarchical_documents, 1)
        level_2_docs = filter_by_level(sample_hierarchical_documents, 2)
        level_3_docs = filter_by_level(sample_hierarchical_documents, 3)

        assert len(level_1_docs) == 2
        assert len(level_2_docs) == 2
        assert len(level_3_docs) == 1

    def test_filter_by_level_empty_documents_list(
        self, empty_documents_list: list[Document]
    ) -> None:
        """Test filtering with empty documents list."""
        result = filter_by_level(empty_documents_list, 1)

        assert result == []

    def test_filter_by_level_documents_without_meta(
        self, sample_documents: list[Document]
    ) -> None:
        """Test filtering with documents that have no metadata."""
        result = filter_by_level(sample_documents, 1)

        assert result == []

    def test_filter_by_level_documents_without_level_key(
        self, sample_documents: list[Document]
    ) -> None:
        """Test filtering with documents that have meta but no level key."""
        docs_with_meta_but_no_level = [
            Document(content="Doc 1", meta={"source": "test"}),
            Document(content="Doc 2", meta={"doc_id": "123"}),
        ]

        result = filter_by_level(docs_with_meta_but_no_level, 1)

        assert result == []

    def test_filter_by_level_string_level_values(self) -> None:
        """Test filtering when level values are strings instead of integers."""
        docs_with_string_levels = [
            Document(content="Doc 1", meta={"level": "1"}),
            Document(content="Doc 2", meta={"level": "2"}),
            Document(content="Doc 3", meta={"level": 1}),  # Mixed types
        ]

        result = filter_by_level(docs_with_string_levels, 1)

        # Should only match the integer level
        assert len(result) == 1
        assert result[0].meta.get("level") == 1

    def test_filter_by_level_negative_level_values(self) -> None:
        """Test filtering with negative level values."""
        docs_with_negative_levels = [
            Document(content="Doc 1", meta={"level": -1}),
            Document(content="Doc 2", meta={"level": 1}),
            Document(content="Doc 3", meta={"level": 0}),
        ]

        result = filter_by_level(docs_with_negative_levels, 1)

        assert len(result) == 1
        assert result[0].meta.get("level") == 1

    def test_filter_by_level_zero_level(self) -> None:
        """Test filtering for level 0."""
        docs_with_zero_level = [
            Document(content="Doc 1", meta={"level": 0}),
            Document(content="Doc 2", meta={"level": 1}),
            Document(content="Doc 3", meta={"level": 2}),
        ]

        result = filter_by_level(docs_with_zero_level, 0)

        assert len(result) == 1
        assert result[0].meta.get("level") == 0

    @pytest.mark.parametrize("level", [1, 2, 3, 0, -1])
    def test_filter_by_level_parameterized(self, level: int) -> None:
        """Test filter_by_level with various level values."""
        docs = [
            Document(content=f"Level {i} doc", meta={"level": i})
            for i in range(-2, 5)  # -2 to 4
        ]

        result = filter_by_level(docs, level)

        expected_count = 1 if -2 <= level <= 4 else 0
        assert len(result) == expected_count
        if expected_count > 0:
            assert result[0].meta.get("level") == level
