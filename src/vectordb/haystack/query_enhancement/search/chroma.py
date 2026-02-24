"""Chroma search pipeline with query enhancement."""

from pathlib import Path

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.haystack.query_enhancement.search.base import (
    BaseQueryEnhancementSearchPipeline,
)


class ChromaQueryEnhancementSearchPipeline(BaseQueryEnhancementSearchPipeline):
    """Multi-query search pipeline for Chroma.

    Generates query variations using LLM (Multi-Query, HyDE, Step-Back),
    executes parallel searches, fuses results with RRF, optionally generates
    RAG answer.
    """

    def __init__(self, config_path: str | Path) -> None:
        """Initialize pipeline from configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        super().__init__(config_path, "chroma_query_enhancement_search")

    def _init_db(self) -> ChromaVectorDB:
        """Initialize Chroma VectorDB from config."""
        chroma_config = self.config.get("chroma", {})
        return ChromaVectorDB(
            collection_name=chroma_config.get("collection_name", "query_enhancement"),
            persist_directory=chroma_config.get("persist_directory"),
            config=self.config,
        )
