"""Integration tests for FactScore loader."""

import pytest

from vectordb.dataloaders.catalog import DataloaderCatalog


INTEGRATION_LIMIT = 3


@pytest.mark.integration
@pytest.mark.enable_socket
class TestFactScoreLoaderIntegration:
    """Smoke test for FactScore loader integration."""

    def test_load(self) -> None:
        """Test loading FactScore dataset through DataloaderCatalog.

        Verifies that the FactScore loader can be created via the catalog,
        loads records successfully, and returns properly structured data
        with text content and metadata containing questions.
        """
        loader = DataloaderCatalog.create("factscore", limit=INTEGRATION_LIMIT)
        dataset = loader.load()

        records = dataset.records()
        assert records
        assert len(records) <= INTEGRATION_LIMIT
        assert records[0].text
        assert "question" in records[0].metadata
