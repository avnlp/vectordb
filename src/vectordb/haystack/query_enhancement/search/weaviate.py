"""Weaviate search pipeline with query enhancement."""

from pathlib import Path

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.haystack.query_enhancement.search.base import (
    BaseQueryEnhancementSearchPipeline,
)


class WeaviateQueryEnhancementSearchPipeline(BaseQueryEnhancementSearchPipeline):
    """Multi-query search pipeline for Weaviate.

    Generates query variations using LLM (Multi-Query, HyDE, Step-Back),
    executes parallel searches, fuses results with RRF, optionally generates
    RAG answer.
    """

    def __init__(self, config_path: str | Path) -> None:
        """Initialize pipeline from configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        super().__init__(config_path, "weaviate_query_enhancement_search")

    def _init_db(self) -> WeaviateVectorDB:
        """Initialize Weaviate VectorDB from config."""
        weaviate_config = self.config.get("weaviate", {})
        return WeaviateVectorDB(
            url=weaviate_config.get("url"),
            api_key=weaviate_config.get("api_key"),
            class_name=weaviate_config.get("class_name", "QueryEnhancement"),
            config=self.config,
        )
