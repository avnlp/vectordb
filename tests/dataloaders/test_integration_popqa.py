"""Integration tests for PopQA loader."""

import pytest

from vectordb.dataloaders.catalog import DataloaderCatalog


INTEGRATION_LIMIT = 3


@pytest.mark.integration
@pytest.mark.enable_socket
class TestPopQALoaderIntegration:
    """Smoke test for PopQA loader integration."""

    def test_load(self) -> None:
        """Test loading PopQA dataset with valid records and metadata.

        Verifies that the PopQA loader creates a dataset with the expected
        structure: records contain text content and metadata includes question fields.
        """
        loader = DataloaderCatalog.create("popqa", limit=INTEGRATION_LIMIT)
        dataset = loader.load()

        records = dataset.records()
        assert records
        assert len(records) <= INTEGRATION_LIMIT
        assert records[0].text
        assert "question" in records[0].metadata
