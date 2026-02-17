"""Integration tests for TriviaQA loader."""

import pytest

from vectordb.dataloaders.catalog import DataloaderCatalog


INTEGRATION_LIMIT = 3


@pytest.mark.integration
@pytest.mark.enable_socket
class TestTriviaQALoaderIntegration:
    """Smoke test for TriviaQA loader integration."""

    def test_load(self) -> None:
        """Test that TriviaQA loader loads dataset with expected structure.

        Verifies the loader creates records with text content and question metadata.
        """
        loader = DataloaderCatalog.create("triviaqa", limit=INTEGRATION_LIMIT)
        dataset = loader.load()

        records = dataset.records()
        assert records
        assert len(records) <= INTEGRATION_LIMIT
        assert records[0].text
        assert "question" in records[0].metadata
