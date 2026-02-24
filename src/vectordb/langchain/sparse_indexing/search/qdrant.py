"""Qdrant sparse search pipeline (LangChain)."""

import logging
from typing import Any

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    RAGHelper,
    SparseEmbedder,
)


logger = logging.getLogger(__name__)


class QdrantSparseSearchPipeline:
    """Qdrant sparse search pipeline (LangChain)."""

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize search pipeline from configuration."""
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "qdrant")

        self.embedder = SparseEmbedder()

        qdrant_config = self.config["qdrant"]
        self.db = QdrantVectorDB(
            url=qdrant_config.get("url", "http://localhost:6333"),
            api_key=qdrant_config.get("api_key"),
        )

        self.collection_name = qdrant_config.get("collection_name", "sparse_search")
        self.llm = RAGHelper.create_llm(self.config)

        logger.info("Initialized Qdrant sparse search pipeline (LangChain)")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute sparse search."""
        query_embedding = self.embedder.embed_query(query)
        logger.info("Embedded query with sparse embeddings: %s", query[:50])

        documents = self.db.query(
            query_embedding=None,  # No dense embeddings for sparse search
            top_k=top_k,
            filters=filters,
            collection_name=self.collection_name,
            sparse_embedding=query_embedding,
        )
        logger.info("Retrieved %d documents from Qdrant", len(documents))

        result = {
            "documents": documents,
            "query": query,
        }

        if self.llm is not None:
            answer = RAGHelper.generate(self.llm, query, documents)
            result["answer"] = answer
            logger.info("Generated RAG answer")

        return result
