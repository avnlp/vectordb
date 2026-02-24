"""Weaviate JSON search pipeline for LangChain."""

import logging
from typing import Any

from langchain_core.documents import Document

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.langchain.utils.embeddings import EmbedderHelper
from vectordb.langchain.utils.filters import DocumentFilter
from vectordb.langchain.utils.rag import RAGHelper
from vectordb.utils.logging import LoggerFactory


class WeaviateJsonSearchPipeline:
    """Searches documents in Weaviate with JSON metadata filtering.

    Attributes:
        config: Configuration dictionary.
        logger: Logger instance.
        vector_db: WeaviateVectorDB wrapper instance.
        embedder: HuggingFaceEmbeddings instance.
        llm: ChatGroq instance (optional, for RAG).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize search pipeline from configuration.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self._setup_logging()
        self._connect()
        self._init_embedder()
        self._init_llm()

    def _setup_logging(self) -> None:
        """Set up logger from config."""
        logging_config = self.config.get("logging", {})
        name = logging_config.get("name", "weaviate_json_search")
        level_str = logging_config.get("level", "INFO")
        level = getattr(logging, level_str.upper(), logging.INFO)
        factory = LoggerFactory(name, log_level=level)
        self.logger = factory.get_logger()

    def _connect(self) -> None:
        """Connect to Weaviate using VectorDB wrapper."""
        weaviate_config = self.config.get("weaviate", {})
        cluster_url = weaviate_config.get("cluster_url")
        api_key = weaviate_config.get("api_key")

        self.vector_db = WeaviateVectorDB(
            cluster_url=cluster_url,
            api_key=api_key,
        )
        self.logger.info("Connected to Weaviate")

    def _init_embedder(self) -> None:
        """Initialize text embedder."""
        self.embedder = EmbedderHelper.create_embedder(self.config)
        self.logger.info("Initialized text embedder")

    def _init_llm(self) -> None:
        """Initialize LLM for RAG (optional)."""
        self.llm = RAGHelper.create_llm(self.config)
        if self.llm:
            self.logger.info("Initialized LLM for RAG")
        else:
            self.logger.info("RAG disabled")

    def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Execute search with JSON metadata filtering.

        Args:
            query: Search query text.
            filters: Optional metadata filters as nested dict.
            top_k: Number of results to return.

        Returns:
            Dictionary with query, documents, and optional answer.
        """
        # Resolve top_k from config if not provided
        if top_k is None:
            top_k = self.config.get("search", {}).get("top_k", 10)

        # Get collection name
        collection_config = self.config.get("collection", {})
        collection_config.get("name", "json_indexed")

        # Embed query
        query_embedding = EmbedderHelper.embed_query(self.embedder, query)
        self.logger.info("Embedded query: %s", query[:50])

        # Query vector DB using hybrid search
        results: list[Document] = self.vector_db.query(
            query_string=query,
            vector=query_embedding,
            limit=top_k,
            filters=filters,
            hybrid=True,
            return_documents=True,
        )
        self.logger.info("Search returned %d results", len(results))

        # Apply JSON metadata filters if specified in config or passed as param
        config_filters = self.config.get("filters", {})
        effective_filters = filters or config_filters

        if effective_filters:
            for condition in effective_filters.get("conditions", []):
                field = condition.get("field", "")
                value = condition.get("value")
                operator = condition.get("operator", "equals")

                # Extract JSON path from field (e.g., "metadata.category" -> "category")
                json_path = (
                    field.replace("metadata.", "")
                    if field.startswith("metadata.")
                    else field
                )

                results = DocumentFilter.filter_by_metadata_json(
                    results,
                    json_path=json_path,
                    value=value,
                    operator=operator,
                )
                self.logger.info(
                    "Applied filter %s %s %s, %d results remaining",
                    json_path,
                    operator,
                    value,
                    len(results),
                )

        # Generate RAG response if LLM is enabled
        answer: str | None = None
        if self.llm and results:
            answer = RAGHelper.generate(self.llm, query, results)
            self.logger.info("Generated RAG response")

        response: dict[str, Any] = {
            "query": query,
            "documents": results,
        }
        if answer is not None:
            response["answer"] = answer

        return response
