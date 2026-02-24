"""Shared fixtures for parent document retrieval utils tests."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from haystack import Document


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for testing."""
    return [
        Document(
            content="Machine learning is a subset of artificial intelligence that focuses on neural networks.",
            meta={"source": "wiki", "doc_id": "doc_1"},
        ),
        Document(
            content="Deep learning uses multiple layers to progressively extract higher-level features from raw input.",
            meta={"source": "paper", "doc_id": "doc_2"},
        ),
        Document(
            content="Natural language processing enables computers to understand and process human language.",
            meta={"source": "blog", "doc_id": "doc_3"},
        ),
    ]


@pytest.fixture
def sample_hierarchical_documents() -> list[Document]:
    """Create sample documents with hierarchical metadata."""
    return [
        Document(
            content="Parent document 1 about machine learning concepts and applications.",
            meta={
                "level": 1,
                "doc_idx": 0,
                "parent_idx": 0,
                "source_id": "source_1",
                "children_ids": ["child_1", "child_2"],
            },
        ),
        Document(
            content="Parent document 2 covering deep learning architectures and training methods.",
            meta={
                "level": 1,
                "doc_idx": 1,
                "parent_idx": 0,
                "source_id": "source_2",
                "children_ids": ["child_3", "child_4"],
            },
        ),
        Document(
            content="Child document 1 focusing on basic ML algorithms.",
            meta={
                "level": 2,
                "parent_id": "parent_1",
                "doc_idx": 0,
                "parent_idx": 0,
                "child_idx": 0,
                "source_id": "source_1",
            },
        ),
        Document(
            content="Child document 2 discussing ML applications in industry.",
            meta={
                "level": 2,
                "parent_id": "parent_1",
                "doc_idx": 0,
                "parent_idx": 0,
                "child_idx": 1,
                "source_id": "source_1",
            },
        ),
    ]


@pytest.fixture
def valid_config_dict() -> dict:
    """Create a valid configuration dictionary."""
    return {
        "database": {
            "type": "pinecone",
            "api_key": "test-key",
            "index_name": "test-index",
        },
        "embeddings": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
        },
        "dataloader": {
            "type": "arc",
            "split": "test",
            "index_limit": 10,
        },
        "chunking": {
            "parent_chunk_size_words": 100,
            "child_chunk_size_words": 25,
            "split_overlap": 5,
        },
    }


@pytest.fixture
def invalid_config_dict() -> dict:
    """Create an invalid configuration dictionary missing required sections."""
    return {
        "database": {
            "type": "pinecone",
            "api_key": "test-key",
        },
        # Missing embeddings section
        "dataloader": {
            "type": "arc",
            "split": "test",
        },
    }


@pytest.fixture
def mock_hierarchical_splitter(
    sample_hierarchical_documents: list[Document],
) -> MagicMock:
    """Create a mock hierarchical document splitter."""
    splitter = MagicMock()
    splitter.run.return_value = {"documents": sample_hierarchical_documents}
    return splitter


@pytest.fixture
def empty_documents_list() -> list[Document]:
    """Create an empty documents list for edge case testing."""
    return []


@pytest.fixture
def sample_content() -> str:
    """Sample content for ID generation testing."""
    return "This is a sample document content for testing ID generation."


@pytest.fixture
def sample_source_id() -> str:
    """Sample source ID for testing."""
    return "source_123"


@pytest.fixture
def tmp_config_file(tmp_path: Path, valid_config_dict: dict) -> Path:
    """Create a temporary YAML config file."""
    import yaml

    config_file = tmp_path / "test_config.yaml"
    with config_file.open("w") as f:
        yaml.dump(valid_config_dict, f)
    return config_file


@pytest.fixture
def malformed_config_file(tmp_path: Path) -> Path:
    """Create a malformed YAML config file."""
    config_file = tmp_path / "malformed_config.yaml"
    with config_file.open("w") as f:
        f.write("invalid: yaml: content: [")
    return config_file


@pytest.fixture
def sample_metadata() -> dict:
    """Sample metadata for testing extraction functions."""
    return {
        "level": 1,
        "parent_id": "parent_123",
        "doc_idx": 0,
        "parent_idx": 0,
        "child_idx": 0,
        "source_id": "source_456",
        "extra_field": "extra_value",
    }


@pytest.fixture
def sample_extra_metadata() -> dict:
    """Sample extra metadata for testing merge functionality."""
    return {
        "custom_field": "custom_value",
        "priority": 1,
        "tags": ["test", "sample"],
    }
