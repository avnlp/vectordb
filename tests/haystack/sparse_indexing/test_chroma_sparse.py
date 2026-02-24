"""Tests for Chroma sparse indexing pipeline.

Note: Chroma OSS does not support sparse vectors. Only Chroma Cloud supports
sparse vectors. These tests verify the NotImplementedError is raised correctly.
"""

import pytest

from vectordb.haystack.sparse_indexing.indexing.chroma import (
    ChromaSparseIndexingPipeline,
)
from vectordb.haystack.sparse_indexing.search.chroma import (
    ChromaSparseSearchPipeline,
)


class TestChromaSparseIndexing:
    """Unit tests for Chroma sparse indexing pipeline."""

    @pytest.fixture
    def chroma_config(self) -> dict:
        """Create Chroma-specific test config."""
        return {
            "dataloader": {"name": "triviaqa", "limit": 10},
            "sparse": {
                "model": "prithivida/Splade_PP_en_v1",
            },
            "chroma": {
                "collection_name": "test_sparse_collection",
                "persist_directory": "./test_data",
            },
            "indexing": {"batch_size": 100},
            "query": {"top_k": 5},
        }

    def test_indexing_init_loads_config(
        self,
        chroma_config: dict,
    ) -> None:
        """Test indexing pipeline initialization loads config correctly."""
        pipeline = ChromaSparseIndexingPipeline(chroma_config)
        assert pipeline.config == chroma_config

    def test_indexing_run_raises_not_implemented(
        self,
        chroma_config: dict,
    ) -> None:
        """Test indexing run raises NotImplementedError."""
        pipeline = ChromaSparseIndexingPipeline(chroma_config)

        with pytest.raises(NotImplementedError) as exc_info:
            pipeline.run()

        assert "Chroma sparse indexing is not supported" in str(exc_info.value)


class TestChromaSparseSearch:
    """Unit tests for Chroma sparse search pipeline."""

    @pytest.fixture
    def chroma_config(self) -> dict:
        """Create Chroma-specific test config."""
        return {
            "sparse": {
                "model": "prithivida/Splade_PP_en_v1",
            },
            "chroma": {
                "collection_name": "test_sparse_collection",
                "persist_directory": "./test_data",
            },
            "query": {"top_k": 5},
        }

    def test_search_init_loads_config(
        self,
        chroma_config: dict,
    ) -> None:
        """Test search pipeline initialization loads config correctly."""
        pipeline = ChromaSparseSearchPipeline(chroma_config)
        assert pipeline.config == chroma_config

    def test_search_run_raises_not_implemented(
        self,
        chroma_config: dict,
    ) -> None:
        """Test search run raises NotImplementedError."""
        pipeline = ChromaSparseSearchPipeline(chroma_config)

        with pytest.raises(NotImplementedError) as exc_info:
            pipeline.search("test query", top_k=5)

        assert "Chroma sparse search is not supported" in str(exc_info.value)
