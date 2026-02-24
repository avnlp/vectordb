"""Qdrant search pipeline with query enhancement."""

from pathlib import Path

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.haystack.query_enhancement.search.base import (
    BaseQueryEnhancementSearchPipeline,
)


class QdrantQueryEnhancementSearchPipeline(BaseQueryEnhancementSearchPipeline):
    """Multi-query search pipeline for Qdrant.

    Generates query variations using LLM (Multi-Query, HyDE, Step-Back),
    executes parallel searches, fuses results with RRF, optionally generates
    RAG answer.
    """

    def __init__(self, config_path: str | Path) -> None:
        """Initialize pipeline from configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        super().__init__(config_path, "qdrant_query_enhancement_search")

    def _init_db(self) -> QdrantVectorDB:
        """Initialize Qdrant VectorDB from config."""
        qdrant_config = self.config.get("qdrant", {})
        return QdrantVectorDB(
            url=qdrant_config.get("url"),
            api_key=qdrant_config.get("api_key"),
            collection_name=qdrant_config.get("collection_name"),
            config=self.config,
        )
