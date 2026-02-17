"""Integration tests for ARC loader."""

import pytest

from vectordb.dataloaders.catalog import DataloaderCatalog


INTEGRATION_LIMIT = 3


@pytest.mark.integration
@pytest.mark.enable_socket
class TestARCLoaderIntegration:
    """Smoke test for ARC loader integration."""

    def test_load(self) -> None:
        """Test that ARC data can be loaded and returns valid records.

        Verifies the dataloader creates ARC loader, loads dataset with limited records,
        and returns records with expected text and metadata fields.
        """
        loader = DataloaderCatalog.create("arc", limit=INTEGRATION_LIMIT)
        dataset = loader.load()

        records = dataset.records()
        assert records
        assert len(records) <= INTEGRATION_LIMIT
        assert records[0].text
        assert "question" in records[0].metadata
