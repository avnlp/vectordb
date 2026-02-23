"""Weaviate hybrid search pipeline (LangChain).

Implements hybrid search using Weaviate's native BM25 keyword search combined
with dense vector similarity search. Unlike Pinecone/Qdrant which use sparse
embeddings, Weaviate's hybrid search fuses BM25 scores (classical keyword
relevance) with vector similarity scores.

The hybrid search mechanism:
    1. Query text is passed directly to Weaviate for BM25 keyword matching
    2. Query is also embedded densely for semantic similarity
    3. Weaviate fuses BM25 and vector scores using alpha weighting
    4. Results are ranked by the combined score

BM25 (Best Match 25):
    A probabilistic ranking function that estimates relevance based on:
    - Term frequency in document
    - Inverse document frequency (rare terms score higher)
    - Field length normalization (shorter fields score higher for matches)

Alpha parameter (0.0 to 1.0):
    - 1.0: Pure vector/semantic search
    - 0.0: Pure BM25/keyword search
    - 0.5: Balanced hybrid (default)

Note:
    Weaviate requires the query_text parameter for BM25 component; sparse
    embeddings are not used as Weaviate handles keyword matching internally.
"""

import logging
from typing import Any

from vectordb.databases.weaviate import WeaviateVectorDB
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
    SparseEmbedder,
)


logger = logging.getLogger(__name__)


class WeaviateHybridSearchPipeline:
    """Weaviate hybrid search pipeline.

    Combines Weaviate's native BM25 keyword search with dense vector similarity
    for enhanced retrieval. Uses alpha weighting to balance lexical and
    semantic relevance.

    Attributes:
        config: Validated configuration dictionary.
        dense_embedder: Embedder for semantic vector generation.
        sparse_embedder: SparseEmbedder instance (not used for search, kept
            for API consistency).
        db: WeaviateVectorDB instance for database operations.
        collection_name: Target Weaviate collection/class name.
        alpha: Fusion weight between BM25 and vector search (0.0-1.0).
        llm: Optional language model for RAG answer generation.

    Example:
        >>> pipeline = WeaviateHybridSearchPipeline("config.yaml")
        >>> results = pipeline.search(
        ...     query="deep learning frameworks", top_k=10, filters={"category": "ai"}
        ... )
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize hybrid search pipeline from configuration.

        Args:
            config_or_path: Configuration dictionary or path to YAML config file.
                Must contain weaviate section with cluster_url (or legacy url),
                api_key, and optional collection_name, alpha settings.

        Raises:
            ValueError: If required configuration keys are missing.
            FileNotFoundError: If config_or_path is a file path that doesn't exist.

        Note:
            The sparse_embedder is initialized but not used for Weaviate search
            since Weaviate handles keyword matching via BM25 internally.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "weaviate")

        self.dense_embedder = EmbedderHelper.create_embedder(self.config)
        self.sparse_embedder = SparseEmbedder()

        weaviate_config = self.config["weaviate"]
        self.db = WeaviateVectorDB(
            cluster_url=weaviate_config.get("cluster_url")
            or weaviate_config.get("url", "http://localhost:8080"),
            api_key=weaviate_config.get("api_key") or "",
        )

        self.collection_name = weaviate_config.get("collection_name")
        self.alpha = weaviate_config.get("alpha", 0.5)
        self.llm = RAGHelper.create_llm(self.config)

        logger.info("Initialized Weaviate hybrid search pipeline (LangChain)")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute hybrid search combining BM25 keyword and vector similarity.

        Generates dense embedding for semantic search and passes query text
        for Weaviate's BM25 keyword matching. Results are fused using alpha
        weighting. Optionally generates RAG answer if LLM configured.

        Args:
            query: Search query text for both BM25 and vector search.
            top_k: Maximum number of results to return. Defaults to 10.
            filters: Optional metadata filters as dictionary for pre-filtering.
                Supports Weaviate's filter syntax.

        Returns:
            Dictionary containing:
                - documents: List of retrieved Document objects
                - query: Original query string
                - answer: Generated RAG answer (only if LLM configured)

        Raises:
            RuntimeError: If database connection fails during search.

        Note:
            BM25 component uses the raw query text, not sparse embeddings.
            This differs from Pinecone/Qdrant which accept sparse vectors.
        """
        dense_embedding = EmbedderHelper.embed_query(self.dense_embedder, query)
        logger.info("Generated dense embedding for query: %s...", query[:50])

        documents = self.db.hybrid_search(
            query_embedding=dense_embedding,
            query_text=query,
            top_k=top_k,
            filters=filters,
            collection_name=self.collection_name,
            alpha=self.alpha,
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
