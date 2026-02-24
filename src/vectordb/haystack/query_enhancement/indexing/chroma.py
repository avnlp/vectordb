"""Chroma indexing pipeline for query enhancement feature."""

from pathlib import Path

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.haystack.query_enhancement.indexing.base import (
    BaseQueryEnhancementIndexingPipeline,
)


class ChromaQueryEnhancementIndexingPipeline(BaseQueryEnhancementIndexingPipeline):
    """Index documents into Chroma for query enhancement retrieval.

    Loads documents from configured dataloader, embeds them,
    and upserts to Chroma collection.
    """

    def __init__(self, config_path: str | Path) -> None:
        """Initialize pipeline from configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        super().__init__(config_path, "chroma_query_enhancement_indexing")

    def _init_db(self) -> ChromaVectorDB:
        """Initialize Chroma VectorDB from config."""
        chroma_config = self.config.get("chroma", {})
        return ChromaVectorDB(
            collection_name=chroma_config.get("collection_name", "query_enhancement"),
            persist_directory=chroma_config.get("persist_directory"),
            config=self.config,
        )
