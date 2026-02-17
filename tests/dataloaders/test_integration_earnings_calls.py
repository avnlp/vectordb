"""Integration tests for earnings calls loader."""

import pytest

from vectordb.dataloaders.catalog import DataloaderCatalog


INTEGRATION_LIMIT = 3


@pytest.mark.integration
@pytest.mark.enable_socket
class TestEarningsCallsLoaderIntegration:
    """Smoke test for earnings calls loader integration."""

    def test_load(self) -> None:
        """Test loading earnings calls data through the dataloader catalog.

        Verifies that the earnings calls loader can be created and used to load
        records with valid text and metadata.
        """
        loader = DataloaderCatalog.create("earnings_calls", limit=INTEGRATION_LIMIT)
        dataset = loader.load()

        records = dataset.records()
        assert records
        assert len(records) <= INTEGRATION_LIMIT
        assert records[0].text
        assert "question" in records[0].metadata
