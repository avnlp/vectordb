"""Qdrant MMR search pipeline.

This pipeline performs diversity-aware retrieval using Maximal Marginal Relevance.
It embeds the query, retrieves candidates from Qdrant, applies MMR reranking,
and optionally generates a RAG answer.

Qdrant-Specific Features:
    - Supports rich metadata filtering via filter parameter
    - Efficient HNSW indexing for approximate nearest neighbor search
    - Supports both local (in-memory/disk) and server deployments
    - Payload storage for document content and metadata

MMR Algorithm Steps:
    1. Embed query: Convert query text to dense vector using configured embedder
    2. Retrieve candidates: Get top_k_candidates from Qdrant
       (uses cosine similarity by default, configurable per collection)
    3. Apply MMR reranking:
       a. Calculate query-document relevance (cosine similarity)
       b. Calculate redundancy (max cosine similarity to selected docs)
       c. Score: λ×relevance - (1-λ)×redundancy
       d. Greedily select highest-scoring documents
    4. Return top_k diverse results

Similarity/Distance Calculations:
    - Query-document relevance: Cosine similarity in embedding space
    - Inter-document redundancy: Cosine similarity between document vectors
    - Both use the same embedding model for consistent similarity metrics

Lambda Parameter Guide:
    - λ = 0.7-0.9: Focus on relevance, minimal diversity
    - λ = 0.5: Balanced relevance and diversity (recommended default)
    - λ = 0.1-0.3: Focus on diversity, useful for exploration

Configuration (YAML):
    - qdrant.url: Qdrant server URL
    - qdrant.collection_name: Collection to search
    - qdrant.api_key: Optional API key for authenticated servers
    - mmr.lambda_threshold: MMR lambda parameter (default 0.5)
    - rag.enabled: Whether to generate RAG answers

Usage:
    >>> pipeline = QdrantMmrSearchPipeline("config.yaml")
    >>> results = pipeline.search("machine learning applications", top_k=10)
    >>> # results = {"documents": [...], "query": "...", "answer": "..."}
"""

import logging
from typing import Any

from vectordb.databases.qdrant import QdrantVectorDB
from vectordb.haystack.utils import (
    ConfigLoader,
    DocumentFilter,
    EmbedderFactory,
    RAGHelper,
    RerankerFactory,
)


logger = logging.getLogger(__name__)


class QdrantMmrSearchPipeline:
    """Qdrant MMR search pipeline for diversity-aware retrieval.

    Embeds query, retrieves from Qdrant, applies MMR reranking,
    and optionally generates RAG answer.

    This pipeline implements the standard MMR search pattern with
    Qdrant-specific support for flexible deployment modes.

    Attributes:
        config: Validated configuration dictionary.
        embedder: Haystack text embedder component.
        db: QdrantVectorDB instance for database operations.
        mmr_ranker: Diversity ranker for MMR reranking.
        lambda_threshold: MMR lambda parameter (0.0-1.0).
        rag_enabled: Whether RAG answer generation is enabled.
        generator: Optional LLM generator for RAG answers.

    Note:
        Qdrant supports both local (in-memory/disk) and server modes,
        making it suitable for both development and production deployments.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize search pipeline from configuration.

        Args:
            config_or_path: Config dict or path to YAML file.

        Raises:
            ValueError: If required config missing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "qdrant")

        self.embedder = EmbedderFactory.create_text_embedder(self.config)

        qdrant_config = self.config["qdrant"]
        self.db = QdrantVectorDB(
            url=qdrant_config.get("url"),
            api_key=qdrant_config.get("api_key"),
            collection_name=qdrant_config.get("collection_name"),
        )

        self.mmr_ranker = RerankerFactory.create_diversity_ranker(self.config)
        self.lambda_threshold = self.config.get("mmr", {}).get("lambda_threshold", 0.5)

        # Optional RAG generator
        self.rag_enabled = self.config.get("rag", {}).get("enabled", False)
        self.generator = (
            RAGHelper.create_generator(self.config) if self.rag_enabled else None
        )

        logger.info("Initialized Qdrant MMR search pipeline")

    def search(
        self,
        query: str,
        top_k: int = 10,
        top_k_candidates: int = 50,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute MMR search with diversity-aware retrieval.

        Args:
            query: Search query text.
            top_k: Number of results to return after MMR.
            top_k_candidates: Number of candidates to retrieve before MMR.
                Higher values (2-3x top_k) give MMR more options for diversity.
            filters: Optional metadata filters.

        Returns:
            Dict with 'documents', 'query', and optional 'answer' keys.
            Also includes 'candidates_retrieved' and 'documents_after_mmr'
            for debugging and evaluation.
        """
        # Step 1: Embed query using configured text embedder
        query_result = self.embedder.run(text=query)
        query_embedding = query_result["embedding"]

        # Step 2: Retrieve candidates from Qdrant collection
        # Filters are normalized to database-agnostic format
        filters = DocumentFilter.normalize(filters)
        candidates = self.db.query(
            query_embedding=query_embedding,
            top_k=top_k_candidates,
            filter=filters if filters else None,
        )
        logger.info("Retrieved %d candidates", len(candidates))

        if not candidates:
            return {"documents": [], "query": query, "answer": None}

        # Step 3: Apply MMR reranking for diversity
        # Uses cosine similarity for both relevance and redundancy
        ranked_result = self.mmr_ranker.run(
            query=query,
            documents=candidates,
            top_k=top_k,
        )
        documents = ranked_result["documents"]
        logger.info("MMR reranked to %d documents", len(documents))

        result: dict[str, Any] = {
            "documents": documents,
            "query": query,
            "candidates_retrieved": len(candidates),
            "documents_after_mmr": len(documents),
        }

        # Step 4: Optional RAG answer generation
        if self.rag_enabled and self.generator and documents:
            answer = RAGHelper.generate(self.generator, query, documents)
            result["answer"] = answer

        return result
