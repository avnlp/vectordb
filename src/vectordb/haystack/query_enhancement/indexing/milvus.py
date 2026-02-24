"""Milvus indexing pipeline for query enhancement feature."""

from pathlib import Path

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.haystack.query_enhancement.indexing.base import (
    BaseQueryEnhancementIndexingPipeline,
)


class MilvusQueryEnhancementIndexingPipeline(BaseQueryEnhancementIndexingPipeline):
    """Index documents into Milvus for query enhancement retrieval.

    Loads documents from configured dataloader, embeds them,
    and upserts to Milvus collection.
    """

    def __init__(self, config_path: str | Path) -> None:
        """Initialize pipeline from configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        super().__init__(config_path, "milvus_query_enhancement_indexing")

    def _init_db(self) -> MilvusVectorDB:
        """Initialize Milvus VectorDB from config."""
        milvus_config = self.config.get("milvus", {})
        return MilvusVectorDB(
            uri=milvus_config.get("uri", "http://localhost:19530"),
            user=milvus_config.get("user"),
            password=milvus_config.get("password"),
            collection_name=milvus_config.get("collection_name"),
            config=self.config,
        )
