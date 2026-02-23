"""Tests for common metadata filtering utilities."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import yaml
from haystack import Document

from vectordb.haystack.metadata_filtering.common import (
    FilterCondition,
    FilteredQueryResult,
    FilterSpec,
    Timer,
    TimingMetrics,
    filter_spec_to_canonical_dict,
    load_documents_from_config,
    load_metadata_filtering_config,
    parse_filter_from_config,
)


class TestTimer:
    """Test Timer context manager."""

    def test_timer_basic(self) -> None:
        """Test basic timer functionality."""
        with Timer() as timer:
            timer.start_time = timer.start_time

        assert timer.elapsed_ms > 0
        assert timer.start_time > 0
        assert timer.end_time > 0
        assert timer.end_time >= timer.start_time

    def test_timer_uninitialized(self) -> None:
        """Test timer returns 0 if not used."""
        timer = Timer()
        assert timer.elapsed_ms == 0.0


class TestConfigLoading:
    """Test configuration loading utilities."""

    def test_load_dict_config(self) -> None:
        """Test loading config from dictionary."""
        config = {"test": "value"}
        loaded = load_metadata_filtering_config(config)
        assert loaded == config

    def test_load_yaml_file(self) -> None:
        """Test loading config from YAML file."""
        config_data = {
            "embeddings": {"model": "test-model"},
            "search": {"top_k": 10},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            loaded = load_metadata_filtering_config(temp_path)
            assert loaded == config_data
        finally:
            os.unlink(temp_path)

    def test_load_missing_file(self) -> None:
        """Test error on missing config file."""
        with pytest.raises(FileNotFoundError):
            load_metadata_filtering_config("/nonexistent/path.yaml")

    def test_load_env_vars(self) -> None:
        """Test environment variable resolution."""
        os.environ["TEST_VAR"] = "test_value"
        os.environ["DEFAULT_VAR"] = "real_value"

        config = {
            "value": "${TEST_VAR}",
            "with_default": "${MISSING:-fallback}",
            "real_default": "${DEFAULT_VAR:-ignored}",
        }

        loaded = load_metadata_filtering_config(config)
        assert loaded["value"] == "test_value"
        assert loaded["with_default"] == "fallback"
        assert loaded["real_default"] == "real_value"


class TestFilterTypes:
    """Test filter-related data types."""

    def test_filter_condition_valid(self) -> None:
        """Test valid filter condition."""
        cond = FilterCondition(field="score", operator="gte", value=0.9)
        assert cond.field == "score"
        assert cond.operator == "gte"
        assert cond.value == 0.9

    def test_filter_condition_invalid_operator(self) -> None:
        """Test invalid filter operator."""
        with pytest.raises(ValueError, match="Invalid operator"):
            FilterCondition(field="score", operator="invalid", value=0.9)

    def test_filter_spec(self) -> None:
        """Test filter specification."""
        conditions = [
            FilterCondition(field="category", operator="eq", value="science"),
            FilterCondition(field="score", operator="gte", value=0.9),
        ]
        spec = FilterSpec(conditions=conditions)
        assert len(spec.conditions) == 2

    def test_timing_metrics_selectivity(self) -> None:
        """Test timing metrics selectivity calculation."""
        metrics = TimingMetrics(
            pre_filter_ms=10.0,
            vector_search_ms=50.0,
            total_ms=60.0,
            num_candidates=50,
            num_total_docs=1000,
        )
        assert metrics.selectivity == 0.05

    def test_timing_metrics_selectivity_zero_docs(self) -> None:
        """Test selectivity with zero total docs."""
        metrics = TimingMetrics(
            pre_filter_ms=10.0,
            vector_search_ms=50.0,
            total_ms=60.0,
            num_candidates=0,
            num_total_docs=0,
        )
        assert metrics.selectivity == 0.0


class TestFilterParsing:
    """Test filter parsing from configuration."""

    def test_parse_empty_filter(self) -> None:
        """Test parsing empty filter configuration."""
        config = {"metadata_filtering": {}}
        spec = parse_filter_from_config(config)
        assert spec.conditions == []

    def test_parse_single_condition(self) -> None:
        """Test parsing single filter condition."""
        config = {
            "metadata_filtering": {
                "test_filters": [
                    {
                        "name": "test",
                        "conditions": [
                            {"field": "category", "operator": "eq", "value": "science"}
                        ],
                    }
                ]
            }
        }
        spec = parse_filter_from_config(config)
        assert len(spec.conditions) == 1
        assert spec.conditions[0].field == "category"
        assert spec.conditions[0].operator == "eq"
        assert spec.conditions[0].value == "science"

    def test_parse_multiple_conditions(self) -> None:
        """Test parsing multiple filter conditions."""
        config = {
            "metadata_filtering": {
                "test_filters": [
                    {
                        "name": "test",
                        "conditions": [
                            {"field": "category", "operator": "eq", "value": "science"},
                            {"field": "score", "operator": "gte", "value": 0.9},
                        ],
                    }
                ]
            }
        }
        spec = parse_filter_from_config(config)
        assert len(spec.conditions) == 2


class TestFilterSpecConversion:
    """Test converting FilterSpec to canonical dict format."""

    def test_empty_spec(self) -> None:
        """Test empty filter specification."""
        spec = FilterSpec(conditions=[])
        result = filter_spec_to_canonical_dict(spec)
        assert result == {}

    def test_single_condition(self) -> None:
        """Test single condition conversion."""
        spec = FilterSpec(
            conditions=[FilterCondition(field="score", operator="gte", value=0.9)]
        )
        result = filter_spec_to_canonical_dict(spec)
        assert result == {
            "field": "score",
            "operator": "gte",
            "value": 0.9,
        }

    def test_multiple_conditions(self) -> None:
        """Test multiple conditions conversion with AND."""
        spec = FilterSpec(
            conditions=[
                FilterCondition(field="category", operator="eq", value="science"),
                FilterCondition(field="score", operator="gte", value=0.9),
            ]
        )
        result = filter_spec_to_canonical_dict(spec)
        assert result["operator"] == "and"
        assert len(result["conditions"]) == 2


class TestDocumentLoading:
    """Test document loading from configuration."""

    @patch(
        "vectordb.haystack.metadata_filtering.common.dataloader.DataloaderCatalog.create"
    )
    def test_load_documents_success(self, mock_create: MagicMock) -> None:
        """Test successful document loading."""
        sample_documents = [
            Document(content="Document 1", meta={"id": 1}),
            Document(content="Document 2", meta={"id": 2}),
        ]
        mock_dataset = MagicMock()
        mock_dataset.to_haystack.return_value = sample_documents
        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_dataset
        mock_create.return_value = mock_loader

        config = {
            "dataloader": {
                "type": "triviaqa",
                "split": "test",
                "limit": 2,
            }
        }

        docs = load_documents_from_config(config)
        assert len(docs) == 2
        assert docs[0].content == "Document 1"
        assert docs[1].content == "Document 2"
        mock_create.assert_called_once()

    def test_load_documents_missing_config(self) -> None:
        """Test error when dataloader config missing."""
        config = {}
        with pytest.raises(ValueError, match="dataloader"):
            load_documents_from_config(config)

    def test_load_documents_missing_type(self) -> None:
        """Test error when dataset type missing."""
        config = {"dataloader": {"split": "test"}}
        with pytest.raises(ValueError, match="type"):
            load_documents_from_config(config)


class TestFilteredQueryResult:
    """Test FilteredQueryResult data class."""

    def test_create_result(self) -> None:
        """Test creating FilteredQueryResult."""
        doc = Document(content="Test content")
        timing = TimingMetrics(
            pre_filter_ms=10.0,
            vector_search_ms=50.0,
            total_ms=60.0,
            num_candidates=100,
            num_total_docs=1000,
        )

        result = FilteredQueryResult(
            document=doc,
            relevance_score=0.95,
            rank=1,
            filter_matched=True,
            timing=timing,
        )

        assert result.document == doc
        assert result.relevance_score == 0.95
        assert result.rank == 1
        assert result.filter_matched is True
        assert result.timing == timing
