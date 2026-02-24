"""Qdrant indexing pipeline for query enhancement feature."""

from pathlib import Path

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.haystack.query_enhancement.indexing.base import (
    BaseQueryEnhancementIndexingPipeline,
)


class QdrantQueryEnhancementIndexingPipeline(BaseQueryEnhancementIndexingPipeline):
    """Index documents into Qdrant for query enhancement retrieval.

    Loads documents from configured dataloader, embeds them,
    and upserts to Qdrant collection.
    """

    def __init__(self, config_path: str | Path) -> None:
        """Initialize pipeline from configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        super().__init__(config_path, "qdrant_query_enhancement_indexing")

    def _init_db(self) -> QdrantVectorDB:
        """Initialize Qdrant VectorDB from config."""
        qdrant_config = self.config.get("qdrant", {})
        return QdrantVectorDB(
            url=qdrant_config.get("url"),
            api_key=qdrant_config.get("api_key"),
            collection_name=qdrant_config.get("collection_name"),
            config=self.config,
        )
