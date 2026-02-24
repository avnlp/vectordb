"""Tests for configuration utilities in parent document retrieval."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from vectordb.haystack.parent_document_retrieval.utils.config import (
    load_parent_doc_config,
)


class TestLoadParentDocConfig:
    """Test suite for load_parent_doc_config function."""

    def test_load_config_from_dict_valid(self, valid_config_dict: dict) -> None:
        """Test loading configuration from a valid dictionary."""
        result = load_parent_doc_config(valid_config_dict)

        assert result == valid_config_dict
        assert "database" in result
        assert "embeddings" in result
        assert "dataloader" in result

    def test_load_config_from_path_string(self, tmp_config_file: Path) -> None:
        """Test loading configuration from a valid file path string."""
        result = load_parent_doc_config(str(tmp_config_file))

        assert "database" in result
        assert "embeddings" in result
        assert "dataloader" in result
        assert result["database"]["type"] == "pinecone"

    def test_load_config_from_path_object(self, tmp_config_file: Path) -> None:
        """Test loading configuration from a valid Path object."""
        result = load_parent_doc_config(tmp_config_file)

        assert "database" in result
        assert "embeddings" in result
        assert "dataloader" in result
        assert result["database"]["type"] == "pinecone"

    @patch("vectordb.haystack.parent_document_retrieval.utils.config.base_load_config")
    def test_load_config_calls_base_function(self, mock_base_load: MagicMock) -> None:
        """Test that base_load_config is called with correct parameters."""
        mock_base_load.return_value = {
            "database": {"type": "pinecone"},
            "embeddings": {"model": "test-model"},
            "dataloader": {"type": "arc"},
        }

        load_parent_doc_config("test_path.yaml")

        mock_base_load.assert_called_once_with("test_path.yaml")

    def test_missing_database_section_raises_error(self) -> None:
        """Test ValueError when database section is missing."""
        config_without_db = {
            "embeddings": {"model": "test-model"},
            "dataloader": {"type": "arc"},
        }

        with pytest.raises(
            ValueError, match="Config missing required section: database"
        ):
            load_parent_doc_config(config_without_db)

    def test_missing_embeddings_section_raises_error(self) -> None:
        """Test ValueError when embeddings section is missing."""
        config_without_embeddings = {
            "database": {"type": "pinecone"},
            "dataloader": {"type": "arc"},
        }

        with pytest.raises(
            ValueError, match="Config missing required section: embeddings"
        ):
            load_parent_doc_config(config_without_embeddings)

    def test_missing_dataloader_section_raises_error(self) -> None:
        """Test ValueError when dataloader section is missing."""
        config_without_dataloader = {
            "database": {"type": "pinecone"},
            "embeddings": {"model": "test-model"},
        }

        with pytest.raises(
            ValueError, match="Config missing required section: dataloader"
        ):
            load_parent_doc_config(config_without_dataloader)

    def test_multiple_missing_sections_raises_error_for_first(self) -> None:
        """Test ValueError reports first missing section when multiple are missing."""
        incomplete_config = {"database": {"type": "pinecone"}}

        # Should raise error for embeddings (first missing section alphabetically)
        with pytest.raises(
            ValueError, match="Config missing required section: embeddings"
        ):
            load_parent_doc_config(incomplete_config)

    def test_empty_config_dict_raises_error(self) -> None:
        """Test ValueError when empty config dictionary is provided."""
        with pytest.raises(
            ValueError, match="Config missing required section: database"
        ):
            load_parent_doc_config({})

    @patch("vectordb.haystack.parent_document_retrieval.utils.config.base_load_config")
    def test_file_not_found_propagates(self, mock_base_load: MagicMock) -> None:
        """Test FileNotFoundError from base_load_config is propagated."""
        mock_base_load.side_effect = FileNotFoundError("File not found")

        with pytest.raises(FileNotFoundError, match="File not found"):
            load_parent_doc_config("nonexistent_file.yaml")

    @patch("vectordb.haystack.parent_document_retrieval.utils.config.base_load_config")
    def test_yaml_error_propagates(self, mock_base_load: MagicMock) -> None:
        """Test YAMLError from base_load_config is propagated."""
        mock_base_load.side_effect = yaml.YAMLError("Invalid YAML")

        with pytest.raises(yaml.YAMLError, match="Invalid YAML"):
            load_parent_doc_config("invalid.yaml")

    def test_config_with_all_required_sections_succeeds(self) -> None:
        """Test that config with all required sections passes validation."""
        minimal_valid_config = {
            "database": {"type": "pinecone"},
            "embeddings": {"model": "test-model"},
            "dataloader": {"type": "arc"},
        }

        result = load_parent_doc_config(minimal_valid_config)

        assert result == minimal_valid_config
        assert "database" in result
        assert "embeddings" in result
        assert "dataloader" in result

    @pytest.mark.parametrize(
        "required_section", ["database", "embeddings", "dataloader"]
    )
    def test_each_required_section_validation(self, required_section: str) -> None:
        """Test validation for each required section individually."""
        config = {
            "database": {"type": "pinecone"},
            "embeddings": {"model": "test-model"},
            "dataloader": {"type": "arc"},
        }

        # Remove the required section being tested
        del config[required_section]

        with pytest.raises(
            ValueError, match=f"Config missing required section: {required_section}"
        ):
            load_parent_doc_config(config)

    def test_type_checking_for_dict_input(self, valid_config_dict: dict) -> None:
        """Test that dict input is properly handled without calling base_load_config."""
        with patch(
            "vectordb.haystack.parent_document_retrieval.utils.config.base_load_config"
        ) as mock_base_load:
            result = load_parent_doc_config(valid_config_dict)

            # Should not call base_load_config for dict input
            mock_base_load.assert_not_called()
            assert result == valid_config_dict

    def test_config_structure_preserved(self, valid_config_dict: dict) -> None:
        """Test that the original config structure is preserved."""
        result = load_parent_doc_config(valid_config_dict)

        # Check that all original keys and nested structure are preserved
        assert result.keys() == valid_config_dict.keys()
        for key, value in valid_config_dict.items():
            assert result[key] == value

    def test_additional_config_sections_allowed(self) -> None:
        """Test that additional sections beyond required ones are allowed."""
        config_with_extra = {
            "database": {"type": "pinecone"},
            "embeddings": {"model": "test-model"},
            "dataloader": {"type": "arc"},
            "extra_section": {"some": "value"},
            "another_extra": {"more": "values"},
        }

        result = load_parent_doc_config(config_with_extra)

        # Should preserve extra sections
        assert result == config_with_extra
        assert "extra_section" in result
        assert "another_extra" in result
