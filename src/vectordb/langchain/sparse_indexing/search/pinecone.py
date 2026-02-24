"""Pinecone sparse search pipeline (LangChain)."""

import logging
from typing import Any

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    RAGHelper,
    SparseEmbedder,
)


logger = logging.getLogger(__name__)


class PineconeSparseSearchPipeline:
    """Pinecone sparse search pipeline (LangChain).

    Embeds query with sparse embedder, retrieves from Pinecone, optionally
    generates RAG answer.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize search pipeline from configuration."""
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "pinecone")

        self.embedder = SparseEmbedder()

        pinecone_config = self.config["pinecone"]
        self.db = PineconeVectorDB(
            api_key=pinecone_config["api_key"],
            index_name=pinecone_config.get("index_name"),
        )

        self.index_name = pinecone_config.get("index_name")
        self.namespace = pinecone_config.get("namespace", "")

        # Optional RAG
        self.llm = RAGHelper.create_llm(self.config)

        logger.info("Initialized Pinecone sparse search pipeline (LangChain)")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute sparse search.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            Dict with 'documents', 'query', and optional 'answer' keys.
        """
        # Embed query with sparse embedder
        query_embedding = self.embedder.embed_query(query)
        logger.info("Embedded query with sparse embeddings: %s", query[:50])

        # Search Pinecone with sparse embedding only
        documents = self.db.query_with_sparse(
            vector=[0.0] * 384,  # Placeholder dense vector
            sparse_vector=query_embedding,
            top_k=top_k,
            filter=filters,
            namespace=self.namespace,
        )
        logger.info("Retrieved %d documents from Pinecone", len(documents))

        result = {
            "documents": documents,
            "query": query,
        }

        # Optional RAG generation
        if self.llm is not None:
            answer = RAGHelper.generate(self.llm, query, documents)
            result["answer"] = answer
            logger.info("Generated RAG answer")

        return result
