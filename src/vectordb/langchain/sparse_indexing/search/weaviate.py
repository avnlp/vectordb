"""Weaviate sparse search pipeline (LangChain)."""

import logging
from typing import Any

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    RAGHelper,
    SparseEmbedder,
)


logger = logging.getLogger(__name__)


class WeaviateSparseSearchPipeline:
    """Weaviate sparse search pipeline (LangChain)."""

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize search pipeline from configuration."""
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "weaviate")

        self.embedder = SparseEmbedder()

        weaviate_config = self.config["weaviate"]
        self.db = WeaviateVectorDB(
            url=weaviate_config["url"],
            api_key=weaviate_config.get("api_key"),
        )

        self.collection_name = weaviate_config.get("collection_name", "SparseSearch")
        self.llm = RAGHelper.create_llm(self.config)

        logger.info("Initialized Weaviate sparse search pipeline (LangChain)")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute sparse search."""
        # Weaviate performs BM25 search natively on the text content.
        # No query-side sparse embedding is needed.
        logger.info("Performing BM25 search for: %s", query[:50])

        documents = self.db.hybrid_search(
            query=query,
            top_k=top_k,
            alpha=0.0,  # alpha=0.0 for sparse-only (BM25) search
            filters=filters,
        )
        logger.info("Retrieved %d documents from Weaviate", len(documents))

        result = {
            "documents": documents,
            "query": query,
        }

        if self.llm is not None:
            answer = RAGHelper.generate(self.llm, query, documents)
            result["answer"] = answer
            logger.info("Generated RAG answer")

        return result
