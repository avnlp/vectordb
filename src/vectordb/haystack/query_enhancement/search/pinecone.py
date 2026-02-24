"""Pinecone search pipeline with query enhancement."""

from pathlib import Path

from haystack import Document

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.haystack.query_enhancement.search.base import (
    BaseQueryEnhancementSearchPipeline,
)


class PineconeQueryEnhancementSearchPipeline(BaseQueryEnhancementSearchPipeline):
    """Multi-query search pipeline for Pinecone.

    Generates query variations using LLM (Multi-Query, HyDE, Step-Back),
    executes parallel searches, fuses results with RRF, optionally generates
    RAG answer.
    """

    def __init__(self, config_path: str | Path) -> None:
        """Initialize pipeline from configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        super().__init__(config_path, "pinecone_query_enhancement_search")

    def _init_db(self) -> PineconeVectorDB:
        """Initialize Pinecone VectorDB from config."""
        pinecone_config = self.config.get("pinecone", {})
        return PineconeVectorDB(
            api_key=pinecone_config.get("api_key"),
            index_name=pinecone_config.get("index_name"),
            config=self.config,
        )

    def _search_single_query(self, query: str, top_k: int) -> list[Document]:
        """Execute search for a single query.

        Pinecone-specific override to handle namespace parameter.

        Args:
            query: Query string.
            top_k: Number of results to return.

        Returns:
            List of Document objects.
        """
        embedded = self.embedder.run(text=query)
        query_embedding = embedded["embedding"]
        namespace = self.config.get("pinecone", {}).get("namespace", "default")
        return self.db.query(query_embedding, top_k=top_k, namespace=namespace)
