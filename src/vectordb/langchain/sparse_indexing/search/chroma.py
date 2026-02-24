"""Chroma sparse search pipeline (LangChain)."""

import logging
from typing import Any

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    RAGHelper,
    SparseEmbedder,
)


logger = logging.getLogger(__name__)


class ChromaSparseSearchPipeline:
    """Chroma sparse search pipeline (LangChain)."""

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize search pipeline from configuration."""
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "chroma")

        self.embedder = SparseEmbedder()

        chroma_config = self.config["chroma"]
        self.db = ChromaVectorDB(
            path=chroma_config.get("path", "./chroma_data"),
        )

        self.collection_name = chroma_config.get("collection_name", "sparse_search")
        self.llm = RAGHelper.create_llm(self.config)

        logger.info("Initialized Chroma sparse search pipeline (LangChain)")

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
        logger.info("Retrieved %d documents from Chroma", len(documents))

        result = {
            "documents": documents,
            "query": query,
        }

        if self.llm is not None:
            answer = RAGHelper.generate(self.llm, query, documents)
            result["answer"] = answer
            logger.info("Generated RAG answer")

        return result
