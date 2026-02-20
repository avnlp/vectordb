"""Tests for chroma_collections module."""

from unittest.mock import MagicMock, patch

from vectordb.dataloaders import (
    ARCLoader,
    EarningsCallsLoader,
    FactScoreLoader,
    PopQALoader,
    TriviaQALoader,
)
from vectordb.haystack.namespaces.chroma_collections import (
    generate_embeddings,
    get_dataloader_map,
)


class TestGetDataloaderMap:
    """Test suite for get_dataloader_map function."""

    def test_get_dataloader_map_returns_dict(self):
        """Test that get_dataloader_map returns a dictionary."""
        result = get_dataloader_map()
        assert isinstance(result, dict)

    def test_get_dataloader_map_has_correct_keys(self):
        """Test that get_dataloader_map has the expected keys."""
        result = get_dataloader_map()
        expected_keys = {"triviaqa", "arc", "popqa", "factscore", "earnings_calls"}
        assert set(result.keys()) == expected_keys

    def test_get_dataloader_map_has_correct_values(self):
        """Test that get_dataloader_map has the correct dataloader classes."""
        result = get_dataloader_map()

        assert result["triviaqa"] == TriviaQALoader
        assert result["arc"] == ARCLoader
        assert result["popqa"] == PopQALoader
        assert result["factscore"] == FactScoreLoader
        assert result["earnings_calls"] == EarningsCallsLoader

    def test_get_dataloader_map_returns_class_objects_not_instances(self):
        """Test that get_dataloader_map returns class objects, not instances."""
        result = get_dataloader_map()

        for dataloader_class in result.values():
            assert isinstance(dataloader_class, type)
            assert callable(dataloader_class)


class TestGenerateEmbeddings:
    """Test suite for generate_embeddings function."""

    @patch(
        "vectordb.haystack.namespaces.chroma_collections.SentenceTransformersDocumentEmbedder"
    )
    def test_generate_embeddings_basic_functionality(self, mock_embedder_class):
        """Test basic functionality of generate_embeddings."""
        # Setup mock
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.warm_up.return_value = None

        # Mock the run method to return documents with embeddings
        def mock_run(documents):
            # Create mock documents with embeddings
            result_docs = []
            for doc in documents:
                mock_result_doc = MagicMock()
                mock_result_doc.content = doc.content
                mock_result_doc.embedding = [0.1, 0.2, 0.3]
                result_docs.append(mock_result_doc)
            return {"documents": result_docs}

        mock_embedder_instance.run.side_effect = mock_run
        mock_embedder_class.return_value = mock_embedder_instance

        # Create mock documents
        mock_doc1 = MagicMock()
        mock_doc1.content = "Test document 1"
        mock_doc1.embedding = None  # Initially no embedding

        mock_doc2 = MagicMock()
        mock_doc2.content = "Test document 2"
        mock_doc2.embedding = None  # Initially no embedding

        mock_doc3 = MagicMock()
        mock_doc3.content = "Test document 3"
        mock_doc3.embedding = None  # Initially no embedding

        mock_doc4 = MagicMock()
        mock_doc4.content = "Test document 4"
        mock_doc4.embedding = None  # Initially no embedding

        # Call the function
        result_split1, result_split2 = generate_embeddings(
            dense_model="test-model",
            haystack_documents_split1=[mock_doc1, mock_doc2],
            haystack_documents_split2=[mock_doc3, mock_doc4],
        )

        # Assertions
        assert len(result_split1) == 2
        assert len(result_split2) == 2
        mock_embedder_class.assert_called_once_with(model="test-model")
        assert mock_embedder_instance.warm_up.called
        assert (
            mock_embedder_instance.run.call_count == 2
        )  # Called twice, once for each split

    @patch(
        "vectordb.haystack.namespaces.chroma_collections.SentenceTransformersDocumentEmbedder"
    )
    def test_generate_embeddings_empty_lists(self, mock_embedder_class):
        """Test generate_embeddings with empty document lists."""
        # Setup mock
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.warm_up.return_value = None
        mock_embedder_instance.run.return_value = {"documents": []}
        mock_embedder_class.return_value = mock_embedder_instance

        # Call with empty lists
        result_split1, result_split2 = generate_embeddings(
            dense_model="test-model",
            haystack_documents_split1=[],
            haystack_documents_split2=[],
        )

        # Assertions
        assert result_split1 == []
        assert result_split2 == []
        assert mock_embedder_instance.warm_up.called
        assert mock_embedder_instance.run.call_count == 2

    @patch(
        "vectordb.haystack.namespaces.chroma_collections.SentenceTransformersDocumentEmbedder"
    )
    def test_generate_embeddings_single_list(self, mock_embedder_class):
        """Test generate_embeddings with one empty and one populated list."""
        # Setup mock
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.warm_up.return_value = None

        # Mock the run method
        def mock_run(documents):
            if not documents:  # Empty list
                return {"documents": []}
            # Create mock documents with embeddings
            result_docs = []
            for doc in documents:
                mock_result_doc = MagicMock()
                mock_result_doc.content = doc.content
                mock_result_doc.embedding = [0.1, 0.2, 0.3]
                result_docs.append(mock_result_doc)
            return {"documents": result_docs}

        mock_embedder_instance.run.side_effect = mock_run
        mock_embedder_class.return_value = mock_embedder_instance

        # Create mock documents for one list only
        mock_doc1 = MagicMock()
        mock_doc1.content = "Test document 1"
        mock_doc1.embedding = None  # Initially no embedding

        mock_doc2 = MagicMock()
        mock_doc2.content = "Test document 2"
        mock_doc2.embedding = None  # Initially no embedding

        # Call with one empty and one populated list
        result_split1, result_split2 = generate_embeddings(
            dense_model="test-model",
            haystack_documents_split1=[mock_doc1, mock_doc2],
            haystack_documents_split2=[],
        )

        # Assertions
        assert len(result_split1) == 2
        assert result_split2 == []  # Should be an empty list, not a dict
        assert mock_embedder_instance.warm_up.called
        assert mock_embedder_instance.run.call_count == 2

    @patch(
        "vectordb.haystack.namespaces.chroma_collections.SentenceTransformersDocumentEmbedder"
    )
    def test_generate_embeddings_different_models(self, mock_embedder_class):
        """Test generate_embeddings with different model names."""
        # Setup mock
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.warm_up.return_value = None
        mock_embedder_instance.run.return_value = {"documents": []}
        mock_embedder_class.return_value = mock_embedder_instance

        # Test with different model
        model_name = "different-test-model"
        generate_embeddings(
            dense_model=model_name,
            haystack_documents_split1=[],
            haystack_documents_split2=[],
        )

        # Check that the correct model was passed to the constructor
        mock_embedder_class.assert_called_with(model=model_name)

    def test_generate_embeddings_signature(self):
        """Test that generate_embeddings has the expected signature."""
        import inspect

        sig = inspect.signature(generate_embeddings)
        params = list(sig.parameters.keys())

        assert params == [
            "dense_model",
            "haystack_documents_split1",
            "haystack_documents_split2",
        ]

        # Check that it returns a tuple
        import typing

        hints = typing.get_type_hints(generate_embeddings)
        assert "return" in hints
        # The return type hint should indicate a tuple of two lists
