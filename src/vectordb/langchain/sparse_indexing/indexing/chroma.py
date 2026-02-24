"""Chroma sparse indexing pipeline stub (LangChain)."""

import logging
from typing import Any

from langchain_core.documents import Document

from .base import BaseSparseIndexingPipeline


logger = logging.getLogger(__name__)


class ChromaSparseIndexingPipeline(BaseSparseIndexingPipeline):
    """Stub pipeline for Chroma sparse indexing (LangChain).

    Chroma sparse vector indexing is not supported. This class inherits
    from BaseSparseIndexingPipeline but overrides run() to return early.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize sparse indexing stub from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
        """
        # Skip base class initialization since we don't need embeddings
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "chroma")
        self.db_config_key = "chroma"

        logger.warning(
            "Initialized Chroma sparse indexing stub. "
            "Chroma sparse vector indexing is not supported."
        )

    def _initialize_db(self) -> None:
        """No-op for Chroma (sparse not supported)."""
        pass

    def _index_documents(
        self,
        documents: list[Document],
        sparse_embeddings: list[dict[str, float]],
    ) -> int:
        """No-op for Chroma (sparse not supported).

        Returns:
            0 (no documents indexed).
        """
        return 0

    def run(self) -> dict[str, Any]:
        """Return without indexing because sparse vectors are unsupported.

        Returns:
            Dict indicating stub status and reason for no indexing.
        """
        logger.warning(
            "Skipping Chroma sparse indexing. "
            "Chroma sparse vector indexing is not supported."
        )
        return {
            "documents_indexed": 0,
            "status": "stub",
            "reason": "Chroma sparse vector indexing is not supported",
        }


# Import here to avoid circular dependency
from vectordb.langchain.utils import ConfigLoader  # noqa: E402
