"""Tests for Haystack Earnings Calls dataset loader.

This module tests the Haystack-specific EarningsCallDataloader implementation
which loads the Earnings Calls dataset with Haystack components.
"""

from unittest.mock import MagicMock, patch

from haystack import Document

from vectordb.dataloaders.haystack.earnings_calls import EarningsCallDataloader


class TestHaystackEarningsCallDataloaderInitialization:
    """Test suite for Haystack EarningsCallDataloader initialization.

    Tests cover:
    - Default configuration
    - Custom dataset name
    - Custom split
    - Custom text splitter
    - Parameter handling
    """

    def test_earnings_call_dataloader_init_defaults(self) -> None:
        """Test Earnings Call dataloader with default parameters."""
        with patch(
            "vectordb.dataloaders.haystack.earnings_calls.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = EarningsCallDataloader()

            assert loader.dataset is not None
            assert loader.data is None
            assert loader.corpus == []
            assert loader.text_splitter is not None  # Default RecursiveDocumentSplitter

    def test_earnings_call_dataloader_init_custom_split(self) -> None:
        """Test Earnings Call dataloader with custom split."""
        with patch(
            "vectordb.dataloaders.haystack.earnings_calls.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = EarningsCallDataloader(split="validation")

            assert loader.dataset is not None

    def test_earnings_call_dataloader_init_custom_dataset_name(self) -> None:
        """Test Earnings Call dataloader with custom dataset name."""
        with patch(
            "vectordb.dataloaders.haystack.earnings_calls.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = EarningsCallDataloader(dataset_name="custom_earnings")

            assert loader.dataset is not None

    def test_earnings_call_dataloader_init_custom_text_splitter(
        self, mock_recursive_document_splitter
    ) -> None:
        """Test Earnings Call dataloader with custom text splitter."""
        with patch(
            "vectordb.dataloaders.haystack.earnings_calls.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = EarningsCallDataloader(
                text_splitter=mock_recursive_document_splitter
            )

            assert loader.text_splitter == mock_recursive_document_splitter

    def test_earnings_call_dataloader_has_text_splitter_default(self) -> None:
        """Test that text splitter defaults to RecursiveDocumentSplitter."""
        with patch(
            "vectordb.dataloaders.haystack.earnings_calls.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_load.return_value = mock_dataset

            loader = EarningsCallDataloader()

            assert loader.text_splitter is not None


class TestHaystackEarningsCallDataloaderLoadData:
    """Test suite for Haystack Earnings Calls dataset loading.

    Tests cover:
    - Loading dataset
    - Data format and structure
    - Metadata preservation
    - Question and answer handling
    - Text splitting integration
    """

    def test_earnings_call_load_data_returns_list(
        self, haystack_earnings_calls_sample_rows, mock_recursive_document_splitter
    ) -> None:
        """Test that load_data returns a list."""
        with patch(
            "vectordb.dataloaders.haystack.earnings_calls.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_earnings_calls_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = EarningsCallDataloader(
                text_splitter=mock_recursive_document_splitter
            )
            result = loader.load_data()

            assert isinstance(result, list)

    def test_earnings_call_load_data_correct_structure(
        self, haystack_earnings_calls_sample_rows, mock_recursive_document_splitter
    ) -> None:
        """Test that loaded data has correct structure."""
        # Mock the dataset and corpus dataset separately
        with patch(
            "vectordb.dataloaders.haystack.earnings_calls.load_dataset"
        ) as mock_load:
            # Mock the main dataset
            mock_main_dataset = MagicMock()
            mock_main_dataset.__iter__ = MagicMock(
                return_value=iter(
                    [
                        {
                            "question": "What was the revenue?",
                            "answer": "10 billion",
                            "date": "2024-01-15",
                            "transcript": "Q4 earnings call transcript for Acme Corp.",
                            "q": "2023-Q4",
                            "ticker": "ACME",
                        }
                    ]
                )
            )

            # Mock the corpus dataset - based on the actual implementation
            mock_corpus_dataset = MagicMock()
            mock_corpus_dataset.__iter__ = MagicMock(
                return_value=iter(
                    [
                        {
                            "symbol": "ACME",
                            "year": "2023",
                            "quarter": 4,  # Quarter is an integer, not string
                            "date": "2024-01-15",
                            "transcript": [
                                {
                                    "speaker": "CEO",
                                    "text": "Our revenue was 10 billion.",
                                }
                            ],
                        }
                    ]
                )
            )

            # Configure load_dataset to return different values based on the arguments
            def side_effect(dataset_name, *args, **kwargs):
                # Check if the call is for the corpus dataset (second arg is "corpus")
                if args and args[0] == "corpus":
                    return mock_corpus_dataset
                return mock_main_dataset

            mock_load.side_effect = side_effect

            loader = EarningsCallDataloader(
                text_splitter=mock_recursive_document_splitter
            )
            result = loader.load_data()

            assert len(result) > 0
            item = result[0]
            # Check that the expected fields are present in the result
            assert "question" in item
            assert "answer" in item
            assert "date" in item
            assert "context" in item
            assert "year" in item
            assert "quarter" in item
            assert "ticker" in item

    def test_earnings_call_load_data_preserves_question(
        self, haystack_earnings_calls_sample_rows, mock_recursive_document_splitter
    ) -> None:
        """Test that question is preserved in the result."""
        with patch(
            "vectordb.dataloaders.haystack.earnings_calls.load_dataset"
        ) as mock_load:
            # Mock the main dataset
            mock_main_dataset = MagicMock()
            mock_main_dataset.__iter__ = MagicMock(
                return_value=iter(
                    [
                        {
                            "question": "What was the revenue?",
                            "answer": "10 billion",
                            "date": "2024-01-15",
                            "transcript": "Q4 earnings call transcript for Acme Corp.",
                            "q": "2023-Q4",
                            "ticker": "ACME",
                        }
                    ]
                )
            )

            # Mock the corpus dataset - based on the actual implementation
            mock_corpus_dataset = MagicMock()
            mock_corpus_dataset.__iter__ = MagicMock(
                return_value=iter(
                    [
                        {
                            "symbol": "ACME",
                            "year": "2023",
                            "quarter": 4,  # Quarter is an integer, not string
                            "date": "2024-01-15",
                            "transcript": [
                                {
                                    "speaker": "CEO",
                                    "text": "Our revenue was 10 billion.",
                                }
                            ],
                        }
                    ]
                )
            )

            # Configure load_dataset to return different values based on the arguments
            def side_effect(dataset_name, *args, **kwargs):
                # Check if the call is for the corpus dataset (second arg is "corpus")
                if args and args[0] == "corpus":
                    return mock_corpus_dataset
                return mock_main_dataset

            mock_load.side_effect = side_effect

            loader = EarningsCallDataloader(
                text_splitter=mock_recursive_document_splitter
            )
            result = loader.load_data()

            assert len(result) > 0
            item = result[0]
            assert "question" in item
            assert "What was the revenue?" in item["question"]

    def test_earnings_call_load_data_preserves_answer(
        self, haystack_earnings_calls_sample_rows, mock_recursive_document_splitter
    ) -> None:
        """Test that answer is preserved in the result."""
        with patch(
            "vectordb.dataloaders.haystack.earnings_calls.load_dataset"
        ) as mock_load:
            # Mock the main dataset
            mock_main_dataset = MagicMock()
            mock_main_dataset.__iter__ = MagicMock(
                return_value=iter(
                    [
                        {
                            "question": "What was the revenue?",
                            "answer": "10 billion",
                            "date": "2024-01-15",
                            "transcript": "Q4 earnings call transcript for Acme Corp.",
                            "q": "2023-Q4",
                            "ticker": "ACME",
                        }
                    ]
                )
            )

            # Mock the corpus dataset - based on the actual implementation
            mock_corpus_dataset = MagicMock()
            mock_corpus_dataset.__iter__ = MagicMock(
                return_value=iter(
                    [
                        {
                            "symbol": "ACME",
                            "year": "2023",
                            "quarter": 4,  # Quarter is an integer, not string
                            "date": "2024-01-15",
                            "transcript": [
                                {
                                    "speaker": "CEO",
                                    "text": "Our revenue was 10 billion.",
                                }
                            ],
                        }
                    ]
                )
            )

            # Configure load_dataset to return different values based on the arguments
            def side_effect(dataset_name, *args, **kwargs):
                # Check if the call is for the corpus dataset (second arg is "corpus")
                if args and args[0] == "corpus":
                    return mock_corpus_dataset
                return mock_main_dataset

            mock_load.side_effect = side_effect

            loader = EarningsCallDataloader(
                text_splitter=mock_recursive_document_splitter
            )
            result = loader.load_data()

            assert len(result) > 0
            item = result[0]
            assert "answer" in item
            assert "10 billion" in item["answer"]

    def test_earnings_call_load_data_preserves_date_and_ticker(
        self, haystack_earnings_calls_sample_rows, mock_recursive_document_splitter
    ) -> None:
        """Test that date and ticker are preserved in the result."""
        with patch(
            "vectordb.dataloaders.haystack.earnings_calls.load_dataset"
        ) as mock_load:
            # Mock the main dataset
            mock_main_dataset = MagicMock()
            mock_main_dataset.__iter__ = MagicMock(
                return_value=iter(
                    [
                        {
                            "question": "What was the revenue?",
                            "answer": "10 billion",
                            "date": "2024-01-15",
                            "transcript": "Q4 earnings call transcript for Acme Corp.",
                            "q": "2023-Q4",
                            "ticker": "ACME",
                        }
                    ]
                )
            )

            # Mock the corpus dataset - based on the actual implementation
            mock_corpus_dataset = MagicMock()
            mock_corpus_dataset.__iter__ = MagicMock(
                return_value=iter(
                    [
                        {
                            "symbol": "ACME",
                            "year": "2023",
                            "quarter": 4,  # Quarter is an integer, not string
                            "date": "2024-01-15",
                            "transcript": [
                                {
                                    "speaker": "CEO",
                                    "text": "Our revenue was 10 billion.",
                                }
                            ],
                        }
                    ]
                )
            )

            # Configure load_dataset to return different values based on the arguments
            def side_effect(dataset_name, *args, **kwargs):
                # Check if the call is for the corpus dataset (second arg is "corpus")
                if args and args[0] == "corpus":
                    return mock_corpus_dataset
                return mock_main_dataset

            mock_load.side_effect = side_effect

            loader = EarningsCallDataloader(
                text_splitter=mock_recursive_document_splitter
            )
            result = loader.load_data()

            assert len(result) > 0
            item = result[0]
            assert "date" in item
            assert "ticker" in item
            assert "year-quarter" in item  # The q field is stored as year-quarter

    def test_earnings_call_load_data_caches_data(
        self, haystack_earnings_calls_sample_rows, mock_recursive_document_splitter
    ) -> None:
        """Test that load_data caches the result."""
        with patch(
            "vectordb.dataloaders.haystack.earnings_calls.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_earnings_calls_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = EarningsCallDataloader(
                text_splitter=mock_recursive_document_splitter
            )
            result1 = loader.load_data()
            result2 = loader.load_data()

            assert result1 == result2
            # Dataset should only be iterated once
            assert mock_dataset.__iter__.call_count == 1

    def test_earnings_call_load_data_handles_multiple_rows(
        self, haystack_earnings_calls_sample_rows, mock_recursive_document_splitter
    ) -> None:
        """Test handling multiple dataset rows."""
        with patch(
            "vectordb.dataloaders.haystack.earnings_calls.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_earnings_calls_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = EarningsCallDataloader(
                text_splitter=mock_recursive_document_splitter
            )
            result = loader.load_data()

            assert len(result) >= len(haystack_earnings_calls_sample_rows)


class TestHaystackEarningsCallDataloaderGetDocuments:
    """Test suite for Haystack Document conversion.

    Tests cover:
    - Converting corpus to Haystack Documents
    - Document structure
    - Metadata preservation
    - Error handling
    """

    def test_earnings_call_get_documents_returns_haystack_docs(
        self, haystack_earnings_calls_sample_rows, mock_recursive_document_splitter
    ) -> None:
        """Test that get_documents returns Haystack Documents."""
        with patch(
            "vectordb.dataloaders.haystack.earnings_calls.load_dataset"
        ) as mock_load:
            # Mock the main dataset
            mock_main_dataset = MagicMock()
            mock_main_dataset.__iter__ = MagicMock(
                return_value=iter(
                    [
                        {
                            "question": "What was the revenue?",
                            "answer": "10 billion",
                            "date": "2024-01-15",
                            "transcript": "Q4 earnings call transcript for Acme Corp.",
                            "q": "2023-Q4",
                            "ticker": "ACME",
                        }
                    ]
                )
            )

            # Mock the corpus dataset - based on the actual implementation
            mock_corpus_dataset = MagicMock()
            mock_corpus_dataset.__iter__ = MagicMock(
                return_value=iter(
                    [
                        {
                            "symbol": "ACME",
                            "year": "2023",
                            "quarter": 4,  # Quarter is an integer, not string
                            "date": "2024-01-15",
                            "transcript": [
                                {
                                    "speaker": "CEO",
                                    "text": "Our revenue was 10 billion.",
                                }
                            ],
                        }
                    ]
                )
            )

            # Configure load_dataset to return different values based on the arguments
            def side_effect(dataset_name, *args, **kwargs):
                # Check if the call is for the corpus dataset (second arg is "corpus")
                if args and args[0] == "corpus":
                    return mock_corpus_dataset
                return mock_main_dataset

            mock_load.side_effect = side_effect

            loader = EarningsCallDataloader(
                text_splitter=mock_recursive_document_splitter
            )
            # Note: get_documents internally loads the corpus, so we don't need to
            # call load_data first
            documents = loader.get_documents()

            assert isinstance(documents, list)
            assert len(documents) > 0
            assert all(isinstance(doc, Document) for doc in documents)

    def test_earnings_call_get_documents_preserves_content(
        self, haystack_earnings_calls_sample_rows, mock_recursive_document_splitter
    ) -> None:
        """Test that get_documents preserves content."""
        with patch(
            "vectordb.dataloaders.haystack.earnings_calls.load_dataset"
        ) as mock_load:
            # Mock the main dataset
            mock_main_dataset = MagicMock()
            mock_main_dataset.__iter__ = MagicMock(
                return_value=iter(
                    [
                        {
                            "question": "What was the revenue?",
                            "answer": "10 billion",
                            "date": "2024-01-15",
                            "transcript": "Q4 earnings call transcript for Acme Corp.",
                            "q": "2023-Q4",
                            "ticker": "ACME",
                        }
                    ]
                )
            )

            # Mock the corpus dataset - based on the actual implementation
            mock_corpus_dataset = MagicMock()
            mock_corpus_dataset.__iter__ = MagicMock(
                return_value=iter(
                    [
                        {
                            "symbol": "ACME",
                            "year": "2023",
                            "quarter": 4,  # Quarter is an integer, not string
                            "date": "2024-01-15",
                            "transcript": [
                                {
                                    "speaker": "CEO",
                                    "text": "Our revenue was 10 billion.",
                                }
                            ],
                        }
                    ]
                )
            )

            # Configure load_dataset to return different values based on the arguments
            def side_effect(dataset_name, *args, **kwargs):
                # Check if the call is for the corpus dataset (second arg is "corpus")
                if args and args[0] == "corpus":
                    return mock_corpus_dataset
                return mock_main_dataset

            mock_load.side_effect = side_effect

            loader = EarningsCallDataloader(
                text_splitter=mock_recursive_document_splitter
            )
            documents = loader.get_documents()

            assert len(documents) > 0
            assert all(doc.content for doc in documents)

    def test_earnings_call_get_documents_preserves_metadata(
        self, haystack_earnings_calls_sample_rows, mock_recursive_document_splitter
    ) -> None:
        """Test that get_documents preserves metadata."""
        with patch(
            "vectordb.dataloaders.haystack.earnings_calls.load_dataset"
        ) as mock_load:
            # Mock the main dataset
            mock_main_dataset = MagicMock()
            mock_main_dataset.__iter__ = MagicMock(
                return_value=iter(
                    [
                        {
                            "question": "What was the revenue?",
                            "answer": "10 billion",
                            "date": "2024-01-15",
                            "transcript": "Q4 earnings call transcript for Acme Corp.",
                            "q": "2023-Q4",
                            "ticker": "ACME",
                        }
                    ]
                )
            )

            # Mock the corpus dataset - based on the actual implementation
            mock_corpus_dataset = MagicMock()
            mock_corpus_dataset.__iter__ = MagicMock(
                return_value=iter(
                    [
                        {
                            "symbol": "ACME",
                            "year": "2023",
                            "quarter": 4,  # Quarter is an integer, not string
                            "date": "2024-01-15",
                            "transcript": [
                                {
                                    "speaker": "CEO",
                                    "text": "Our revenue was 10 billion.",
                                }
                            ],
                        }
                    ]
                )
            )

            # Configure load_dataset to return different values based on the arguments
            def side_effect(dataset_name, *args, **kwargs):
                # Check if the call is for the corpus dataset (second arg is "corpus")
                if args and args[0] == "corpus":
                    return mock_corpus_dataset
                return mock_main_dataset

            mock_load.side_effect = side_effect

            loader = EarningsCallDataloader(
                text_splitter=mock_recursive_document_splitter
            )
            documents = loader.get_documents()

            assert len(documents) > 0
            assert all(doc.meta for doc in documents)

    def test_earnings_call_get_documents_works_when_corpus_dataset_empty(
        self, mock_recursive_document_splitter
    ) -> None:
        """Test that get_documents works when corpus dataset is empty.

        Unlike other loaders, this doesn't raise a Corpus empty error.
        """
        with patch(
            "vectordb.dataloaders.haystack.earnings_calls.load_dataset"
        ) as mock_load:
            # Mock both the main dataset and the corpus dataset
            mock_main_dataset = MagicMock()
            mock_main_dataset.__iter__ = MagicMock(
                return_value=iter([])
            )  # Empty main dataset
            mock_corpus_dataset = MagicMock()
            mock_corpus_dataset.__iter__ = MagicMock(
                return_value=iter([])
            )  # Empty corpus dataset

            # Configure load_dataset to return different values based on the arguments
            def side_effect(dataset_name, *args, **kwargs):
                # Check if the call is for the corpus dataset (second arg is "corpus")
                if args and args[0] == "corpus":
                    return mock_corpus_dataset
                return mock_main_dataset

            mock_load.side_effect = side_effect

            loader = EarningsCallDataloader(
                text_splitter=mock_recursive_document_splitter
            )

            # get_documents should not raise an error even when corpus dataset is
            # empty (it processes the corpus_dataset in the method itself)
            documents = loader.get_documents()

            # Should return an empty list since corpus dataset is empty
            assert documents == []


class TestHaystackEarningsCallDataloaderIntegration:
    """Test suite for full Earnings Call loader workflow.

    Tests cover:
    - Complete load and conversion pipeline
    - Data consistency
    - Multiple items handling
    """

    def test_earnings_call_full_workflow(
        self, haystack_earnings_calls_sample_rows, mock_recursive_document_splitter
    ) -> None:
        """Test complete load_data -> get_documents workflow."""
        with patch(
            "vectordb.dataloaders.haystack.earnings_calls.load_dataset"
        ) as mock_load:
            # Mock the main dataset
            mock_main_dataset = MagicMock()
            mock_main_dataset.__iter__ = MagicMock(
                return_value=iter(
                    [
                        {
                            "question": "What was the revenue?",
                            "answer": "10 billion",
                            "date": "2024-01-15",
                            "transcript": "Q4 earnings call transcript for Acme Corp.",
                            "q": "2023-Q4",
                            "ticker": "ACME",
                        }
                    ]
                )
            )

            # Mock the corpus dataset - based on the actual implementation
            mock_corpus_dataset = MagicMock()
            mock_corpus_dataset.__iter__ = MagicMock(
                return_value=iter(
                    [
                        {
                            "symbol": "ACME",
                            "year": "2023",
                            "quarter": 4,  # Quarter is an integer, not string
                            "date": "2024-01-15",
                            "transcript": [
                                {
                                    "speaker": "CEO",
                                    "text": "Our revenue was 10 billion.",
                                }
                            ],
                        }
                    ]
                )
            )

            # Configure load_dataset to return different values based on the arguments
            def side_effect(dataset_name, *args, **kwargs):
                # Check if the call is for the corpus dataset (second arg is "corpus")
                if args and args[0] == "corpus":
                    return mock_corpus_dataset
                return mock_main_dataset

            mock_load.side_effect = side_effect

            loader = EarningsCallDataloader(
                text_splitter=mock_recursive_document_splitter
            )
            corpus = loader.load_data()
            documents = loader.get_documents()

            assert len(corpus) > 0
            assert len(documents) > 0
            # Note: The length of documents and corpus may not be equal since they
            # come from different datasets

    def test_earnings_call_handles_multiple_rows(
        self, haystack_earnings_calls_sample_rows, mock_recursive_document_splitter
    ) -> None:
        """Test handling multiple dataset rows."""
        with patch(
            "vectordb.dataloaders.haystack.earnings_calls.load_dataset"
        ) as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(
                return_value=iter(haystack_earnings_calls_sample_rows)
            )
            mock_load.return_value = mock_dataset

            loader = EarningsCallDataloader(
                text_splitter=mock_recursive_document_splitter
            )
            result = loader.load_data()

            assert len(result) >= len(haystack_earnings_calls_sample_rows)
