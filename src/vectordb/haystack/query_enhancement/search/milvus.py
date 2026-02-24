"""Milvus search pipeline with query enhancement."""

from pathlib import Path

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.haystack.query_enhancement.search.base import (
    BaseQueryEnhancementSearchPipeline,
)


class MilvusQueryEnhancementSearchPipeline(BaseQueryEnhancementSearchPipeline):
    """Multi-query search pipeline for Milvus.

    Generates query variations using LLM (Multi-Query, HyDE, Step-Back),
    executes parallel searches, fuses results with RRF, optionally generates
    RAG answer.
    """

    def __init__(self, config_path: str | Path) -> None:
        """Initialize pipeline from configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        super().__init__(config_path, "milvus_query_enhancement_search")

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
