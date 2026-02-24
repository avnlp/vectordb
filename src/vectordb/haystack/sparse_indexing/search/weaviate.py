"""Weaviate BM25 search pipeline for keyword/BM25-style retrieval."""

from pathlib import Path
from typing import Any

from haystack import Document

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.haystack.utils import ConfigLoader
from vectordb.utils.logging import LoggerFactory


logger = LoggerFactory(logger_name=__name__).get_logger()


class WeaviateBM25SearchPipeline:
    """Weaviate BM25 search pipeline.

    Weaviate computes BM25 internally from stored text at query time,
    so no external sparse embeddings are needed.
    """

    def __init__(self, config_or_path: dict[str, Any] | str | Path) -> None:
        """Initialize pipeline from configuration."""
        self.config = ConfigLoader.load(config_or_path)
        db_config = self.config["weaviate"]

        self.db = WeaviateVectorDB(
            url=db_config.get("url"),
            api_key=db_config.get("api_key"),
            index_name=db_config.get("index_name"),
        )

        self.top_k = self.config.get("query", {}).get("top_k", 10)

        logger.info("Initialized WeaviateBM25SearchPipeline")

    def search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """Search using BM25.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            Dict with 'query' and 'documents' keys.
        """
        top_k = top_k or self.top_k

        # 1. Query Weaviate using BM25 (computed internally from stored text)
        results = self.db.bm25_search(
            query=query,
            top_k=top_k,
        )

        # 2. Convert results to Haystack Documents
        documents = []
        for result in results:
            doc = Document(
                content=result.content,
                id=result.id,
                meta={
                    "score": result.score,
                    **result.meta,
                },
            )
            documents.append(doc)

        logger.info(f"Found {len(documents)} documents for query: {query[:50]}...")
        return {"query": query, "documents": documents}
