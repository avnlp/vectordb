"""Chroma sparse search pipeline stub (LangChain)."""

import logging
from typing import Any

from vectordb.langchain.utils import ConfigLoader


logger = logging.getLogger(__name__)


class ChromaSparseSearchPipeline:
    """Stub pipeline for Chroma sparse search (LangChain)."""

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize sparse search stub from configuration."""
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "chroma")

        logger.warning(
            "Initialized Chroma sparse search stub. "
            "Chroma sparse vector search is not supported."
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return without searching because sparse vectors are unsupported."""
        logger.warning(
            "Skipping Chroma sparse search. "
            "Chroma sparse vector search is not supported. "
            "query=%r top_k=%d filters=%r",
            query,
            top_k,
            filters,
        )
        return {
            "documents": [],
            "query": query,
            "status": "stub",
            "reason": "Chroma sparse vector search is not supported",
        }
