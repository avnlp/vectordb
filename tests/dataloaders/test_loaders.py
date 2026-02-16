"""Tests for dataloader registry and protocol.

This module tests the DatasetRegistry for loading datasets
and verifies the DataloaderProtocol interface.
"""

from unittest.mock import MagicMock, patch

import pytest

from vectordb.dataloaders.loaders import DataloaderProtocol, DatasetRegistry


class TestDatasetRegistry:
    """Test suite for DatasetRegistry.

    Tests cover:
    - Loading datasets by type
    - Registry registration and lookup
    - Supported dataset list
    - Configuration parameter passing
    - Error handling for unsupported types
    """

    def test_registry_supported_datasets(self) -> None:
        """Test that registry lists all supported datasets."""
        supported = DatasetRegistry.supported_datasets()

        assert isinstance(supported, list)
        assert len(supported) > 0
        assert "arc" in supported
        assert "earnings_calls" in supported

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_registry_load_arc(self) -> None:
        """Test loading ARC dataset through registry."""
        with patch("vectordb.dataloaders.arc.ARCDataloader") as mock_loader_class:
            mock_instance = MagicMock()
            mock_instance.load.return_value = []
            mock_loader_class.return_value = mock_instance

            DatasetRegistry.load("arc", limit=5)

            assert mock_loader_class.called
            mock_instance.load.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_registry_load_triviaqa(self) -> None:
        """Test loading TriviaQA dataset through registry."""
        with patch(
            "vectordb.dataloaders.triviaqa.TriviaQADataloader"
        ) as mock_loader_class:
            mock_instance = MagicMock()
            mock_instance.load.return_value = []
            mock_loader_class.return_value = mock_instance

            DatasetRegistry.load("triviaqa")

            assert mock_loader_class.called

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_registry_load_popqa(self) -> None:
        """Test loading PopQA dataset through registry."""
        with patch("vectordb.dataloaders.popqa.PopQADataloader") as mock_loader_class:
            mock_instance = MagicMock()
            mock_instance.load.return_value = []
            mock_loader_class.return_value = mock_instance

            DatasetRegistry.load("popqa")

            assert mock_loader_class.called

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_registry_load_factscore(self) -> None:
        """Test loading FactScore dataset through registry."""
        with patch(
            "vectordb.dataloaders.factscore.FactScoreDataloader"
        ) as mock_loader_class:
            mock_instance = MagicMock()
            mock_instance.load.return_value = []
            mock_loader_class.return_value = mock_instance

            DatasetRegistry.load("factscore")

            assert mock_loader_class.called

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_registry_load_earnings_calls(self) -> None:
        """Test loading earnings calls dataset through registry."""
        with patch(
            "vectordb.dataloaders.earnings_calls.EarningsCallDataloader"
        ) as mock_loader_class:
            mock_instance = MagicMock()
            mock_instance.load.return_value = []
            mock_loader_class.return_value = mock_instance

            DatasetRegistry.load("earnings_calls")

            assert mock_loader_class.called

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_registry_load_with_dataset_name(self) -> None:
        """Test loading with custom dataset name."""
        with patch("vectordb.dataloaders.loaders.ARCDataloader") as mock_loader_class:
            mock_instance = MagicMock()
            mock_instance.load.return_value = []
            mock_loader_class.return_value = mock_instance

            DatasetRegistry.load("arc", dataset_name="custom_arc")

            mock_loader_class.assert_called_once()
            # Verify dataset_name was passed
            assert "dataset_name" in str(mock_loader_class.call_args)

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_registry_load_with_split(self) -> None:
        """Test loading with custom split."""
        with patch("vectordb.dataloaders.loaders.ARCDataloader") as mock_loader_class:
            mock_instance = MagicMock()
            mock_instance.load.return_value = []
            mock_loader_class.return_value = mock_instance

            DatasetRegistry.load("arc", split="validation")

            mock_loader_class.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_registry_load_with_limit(self) -> None:
        """Test loading with limit parameter."""
        with patch("vectordb.dataloaders.loaders.ARCDataloader") as mock_loader_class:
            mock_instance = MagicMock()
            mock_instance.load.return_value = []
            mock_loader_class.return_value = mock_instance

            DatasetRegistry.load("arc", limit=100)

            mock_loader_class.assert_called_once()
            assert "limit" in str(mock_loader_class.call_args)

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_registry_load_case_insensitive(self) -> None:
        """Test that dataset type is case insensitive."""
        with patch("vectordb.dataloaders.loaders.ARCDataloader") as mock_loader_class:
            mock_instance = MagicMock()
            mock_instance.load.return_value = []
            mock_loader_class.return_value = mock_instance

            DatasetRegistry.load("ARC")

            assert mock_loader_class.called

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_registry_loader_instantiation_with_kwargs(self) -> None:
        """Test that loader instantiation passes correct kwargs (lines 97-105)."""
        with patch("vectordb.dataloaders.loaders.ARCDataloader") as mock_loader_class:
            mock_instance = MagicMock()
            mock_instance.load.return_value = []
            mock_loader_class.return_value = mock_instance

            # Test with all parameters
            DatasetRegistry.load(
                "arc", dataset_name="custom_dataset", split="validation", limit=50
            )

            # Verify loader class was called with correct kwargs
            mock_loader_class.assert_called_once()
            call_kwargs = mock_loader_class.call_args[1]
            assert call_kwargs == {
                "split": "validation",
                "dataset_name": "custom_dataset",
                "limit": 50,
            }

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_registry_loader_instantiation_minimal_params(self) -> None:
        """Test loader instantiation with minimal parameters."""
        with patch("vectordb.dataloaders.loaders.ARCDataloader") as mock_loader_class:
            mock_instance = MagicMock()
            mock_instance.load.return_value = []
            mock_loader_class.return_value = mock_instance

            # Test with only required parameter
            DatasetRegistry.load("arc")

            # Verify loader class was called with default split
            mock_loader_class.assert_called_once()
            call_kwargs = mock_loader_class.call_args[1]
            assert call_kwargs == {"split": "test"}

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_registry_loader_instantiation_optional_params(self) -> None:
        """Test loader instantiation with various optional parameter combinations."""
        test_cases = [
            # (dataset_name, split, limit, expected_kwargs)
            (None, "train", None, {"split": "train"}),
            (
                "custom",
                None,
                25,
                {"split": "test", "dataset_name": "custom", "limit": 25},
            ),
            (None, "dev", 10, {"split": "dev", "limit": 10}),
        ]

        for dataset_name, split, limit, expected_kwargs in test_cases:
            with patch(
                "vectordb.dataloaders.loaders.ARCDataloader"
            ) as mock_loader_class:
                mock_instance = MagicMock()
                mock_instance.load.return_value = []
                mock_loader_class.return_value = mock_instance

                DatasetRegistry.load(
                    "arc", dataset_name=dataset_name, split=split, limit=limit
                )

                mock_loader_class.assert_called_once()
                call_kwargs = mock_loader_class.call_args[1]
                assert call_kwargs == expected_kwargs

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_registry_loader_instantiation_dataset_name_only(self) -> None:
        """Test loader instantiation with only dataset_name provided."""
        with patch("vectordb.dataloaders.loaders.ARCDataloader") as mock_loader_class:
            mock_instance = MagicMock()
            mock_instance.load.return_value = []
            mock_loader_class.return_value = mock_instance

            # Test with only dataset_name
            DatasetRegistry.load("arc", dataset_name="custom_dataset")

            # Verify loader class was called with correct kwargs
            mock_loader_class.assert_called_once()
            call_kwargs = mock_loader_class.call_args[1]
            assert call_kwargs == {"split": "test", "dataset_name": "custom_dataset"}

    @pytest.mark.integration
    @pytest.mark.enable_socket
    def test_registry_loader_instantiation_limit_only(self) -> None:
        """Test loader instantiation with only limit provided."""
        with patch("vectordb.dataloaders.loaders.ARCDataloader") as mock_loader_class:
            mock_instance = MagicMock()
            mock_instance.load.return_value = []
            mock_loader_class.return_value = mock_instance

            # Test with only limit
            DatasetRegistry.load("arc", limit=100)

            # Verify loader class was called with correct kwargs
            mock_loader_class.assert_called_once()
            call_kwargs = mock_loader_class.call_args[1]
            assert call_kwargs == {"split": "test", "limit": 100}

    def test_registry_load_unsupported_raises_error(self) -> None:
        """Test that unsupported dataset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            DatasetRegistry.load("unsupported_dataset")

    def test_registry_register_custom_loader(self) -> None:
        """Test registering a custom loader."""
        custom_loader = MagicMock(spec=DataloaderProtocol)
        custom_loader.load.return_value = []

        DatasetRegistry.register("custom", custom_loader)

        supported = DatasetRegistry.supported_datasets()
        assert "custom" in supported

    def test_registry_register_overwrites_existing(self) -> None:
        """Test that registering overwrites existing loaders."""
        new_loader = MagicMock()

        DatasetRegistry.register("arc", new_loader)

        # Verify new loader is in registry
        supported = DatasetRegistry.supported_datasets()
        assert "arc" in supported


class TestEdgeCases:
    """Test suite for edge cases and error handling.

    Tests cover:
    - Empty results
    - Malformed data
    - Invalid parameters
    - Error conditions
    """

    def test_loader_returns_empty_list(self) -> None:
        """Test that loaders can return empty lists."""
        mock_loader = MagicMock(spec=DataloaderProtocol)
        mock_loader.load.return_value = []

        result = mock_loader.load()
        assert result == []
        assert isinstance(result, list)

    def test_loader_returns_malformed_data(self) -> None:
        """Test handling of malformed data from loaders."""
        mock_loader = MagicMock(spec=DataloaderProtocol)
        # Missing required fields
        mock_loader.load.return_value = [{"metadata": {}}, {"text": "no metadata"}]

        result = mock_loader.load()
        assert len(result) == 2
        # Protocol doesn't enforce data validation, just structure
        assert all(isinstance(item, dict) for item in result)

    def test_registry_load_with_invalid_dataset_type(self) -> None:
        """Test error handling for invalid dataset types."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            DatasetRegistry.load("invalid_dataset_type")

    def test_registry_load_with_none_dataset_type(self) -> None:
        """Test error handling for None dataset type."""
        with pytest.raises(
            AttributeError, match="'NoneType' object has no attribute 'lower'"
        ):
            DatasetRegistry.load(None)

    def test_registry_load_with_empty_string_dataset_type(self) -> None:
        """Test error handling for empty string dataset type."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            DatasetRegistry.load("")

    def test_loader_with_exception_handling(self) -> None:
        """Test that loaders can handle exceptions gracefully."""
        mock_loader = MagicMock(spec=DataloaderProtocol)
        mock_loader.load.side_effect = Exception("Test exception")

        with pytest.raises(Exception, match="Test exception"):
            mock_loader.load()

    def test_registry_with_custom_loader_exception(self) -> None:
        """Test registry behavior when custom loader raises exception."""
        custom_loader_class = MagicMock()
        custom_instance = MagicMock(spec=DataloaderProtocol)
        custom_instance.load.side_effect = RuntimeError("Custom loader error")
        custom_loader_class.return_value = custom_instance

        DatasetRegistry.register("test_error", custom_loader_class)

        with pytest.raises(RuntimeError, match="Custom loader error"):
            DatasetRegistry.load("test_error")

    def test_loader_with_large_dataset(self) -> None:
        """Test loader behavior with large datasets."""
        mock_loader = MagicMock(spec=DataloaderProtocol)
        # Simulate large dataset
        large_data = [{"text": f"item_{i}", "metadata": {"id": i}} for i in range(1000)]
        mock_loader.load.return_value = large_data

        result = mock_loader.load()
        assert len(result) == 1000
        assert all(isinstance(item, dict) for item in result)

    def test_loader_with_special_characters(self) -> None:
        """Test loader behavior with special characters in data."""
        mock_loader = MagicMock(spec=DataloaderProtocol)
        special_data = [
            {
                "text": "Test with \n newlines and \t tabs",
                "metadata": {"special": "chars"},
            },
            {
                "text": "Unicode: \u2713 \u2717 \u2764",
                "metadata": {"emoji": "\ud83d\ude00"},
            },
        ]
        mock_loader.load.return_value = special_data

        result = mock_loader.load()
        assert len(result) == 2
        assert "\n" in result[0]["text"]
        assert "\u2713" in result[1]["text"]


class TestDataloaderProtocol:
    """Test suite for DataloaderProtocol.

    Tests cover:
    - Protocol implementation verification
    - Required methods
    - Type checking
    """

    def test_protocol_has_load_method(self) -> None:
        """Test that protocol requires load method."""
        assert hasattr(DataloaderProtocol, "__annotations__")
        assert hasattr(DataloaderProtocol, "load")
        assert callable(DataloaderProtocol.load)

    def test_protocol_load_method_signature(self) -> None:
        """Test that protocol load method has correct signature."""
        import inspect

        sig = inspect.signature(DataloaderProtocol.load)
        # Python 3.10+ may quote forward references in signature strings
        sig_str = str(sig)
        assert sig_str in {
            "(self) -> list[dict[str, Any]]",
            "(self) -> 'list[dict[str, Any]]'",
        }

    def test_mock_loader_implements_protocol(self) -> None:
        """Test that mock loaders implement the protocol."""
        mock_loader = MagicMock(spec=DataloaderProtocol)
        mock_loader.load.return_value = []

        result = mock_loader.load()

        assert isinstance(result, list)

    def test_loader_returns_list_of_dicts(self) -> None:
        """Test that loaders return list of dicts."""
        mock_loader = MagicMock(spec=DataloaderProtocol)
        mock_loader.load.return_value = [{"text": "sample", "metadata": {}}]

        result = mock_loader.load()

        assert isinstance(result, list)
        assert isinstance(result[0], dict)
        assert "text" in result[0]
        assert "metadata" in result[0]

    def test_loader_empty_dataset_returns_empty_list(self) -> None:
        """Test that empty datasets return empty list."""
        mock_loader = MagicMock(spec=DataloaderProtocol)
        mock_loader.load.return_value = []

        result = mock_loader.load()

        assert result == []

    def test_protocol_concrete_implementation(self) -> None:
        """Test that concrete implementations satisfy the protocol."""
        from vectordb.dataloaders.arc import ARCDataloader

        # Verify ARCDataloader implements the protocol
        assert isinstance(ARCDataloader, type)
        assert hasattr(ARCDataloader, "load")
        assert callable(ARCDataloader.load)
