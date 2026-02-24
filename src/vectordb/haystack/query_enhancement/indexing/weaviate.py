"""Weaviate indexing pipeline for query enhancement feature."""

from pathlib import Path

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.haystack.query_enhancement.indexing.base import (
    BaseQueryEnhancementIndexingPipeline,
)


class WeaviateQueryEnhancementIndexingPipeline(BaseQueryEnhancementIndexingPipeline):
    """Index documents into Weaviate for query enhancement retrieval.

    Loads documents from configured dataloader, embeds them,
    and upserts to Weaviate class.
    """

    def __init__(self, config_path: str | Path) -> None:
        """Initialize pipeline from configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        super().__init__(config_path, "weaviate_query_enhancement_indexing")

    def _init_db(self) -> WeaviateVectorDB:
        """Initialize Weaviate VectorDB from config."""
        weaviate_config = self.config.get("weaviate", {})
        return WeaviateVectorDB(
            url=weaviate_config.get("url"),
            api_key=weaviate_config.get("api_key"),
            class_name=weaviate_config.get("class_name", "QueryEnhancement"),
            config=self.config,
        )
