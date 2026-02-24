"""Milvus MMR search pipeline.

This pipeline performs diversity-aware retrieval using Maximal Marginal Relevance.
It embeds the query, retrieves candidates from Milvus, applies MMR reranking,
and optionally generates a RAG answer.

Milvus-Specific Features:
    - Supports metadata filtering via filter parameter
    - Efficient vector search with HNSW index
    - Scalable to billions of vectors
    - Supports both local (Lite) and server deployments

MMR Algorithm Steps:
    1. Embed query: Convert query text to dense vector using configured embedder
    2. Retrieve candidates: Get top_k_candidates from Milvus
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
    - milvus.uri: Milvus server URI
    - milvus.collection_name: Collection to search
    - mmr.lambda_threshold: MMR lambda parameter (default 0.5)
    - rag.enabled: Whether to generate RAG answers

Usage:
    >>> pipeline = MilvusMmrSearchPipeline("config.yaml")
    >>> results = pipeline.search("machine learning applications", top_k=10)
    >>> # results = {"documents": [...], "query": "...", "answer": "..."}
"""

import logging
from typing import Any

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.haystack.utils import (
    ConfigLoader,
    DocumentFilter,
    EmbedderFactory,
    RAGHelper,
    RerankerFactory,
)


logger = logging.getLogger(__name__)


class MilvusMmrSearchPipeline:
    """Milvus MMR search pipeline for diversity-aware retrieval.

    Embeds query, retrieves from Milvus, applies MMR reranking,
    and optionally generates RAG answer.

    This pipeline implements the standard MMR search pattern:
    1. Embed the query using the configured text embedder
    2. Retrieve candidates from Milvus (higher top_k for more candidates)
    3. Apply MMR reranking to balance relevance and diversity
    4. Optionally generate RAG answer if enabled

    Attributes:
        config: Validated configuration dictionary.
        embedder: Haystack text embedder component.
        db: MilvusVectorDB instance for database operations.
        mmr_ranker: Diversity ranker for MMR reranking.
        lambda_threshold: MMR lambda parameter (0.0-1.0) controlling
            relevance vs diversity trade-off.
        rag_enabled: Whether RAG answer generation is enabled.
        generator: Optional LLM generator for RAG answers.

    Note:
        top_k_candidates should typically be 2-3x the final top_k to give
        MMR enough candidates to select a diverse set from.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize search pipeline from configuration.

        Args:
            config_or_path: Config dict or path to YAML file.

        Raises:
            ValueError: If required config missing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "milvus")

        self.embedder = EmbedderFactory.create_text_embedder(self.config)

        milvus_config = self.config["milvus"]
        self.db = MilvusVectorDB(
            uri=milvus_config["uri"],
            collection_name=milvus_config.get("collection_name"),
        )

        self.mmr_ranker = RerankerFactory.create_diversity_ranker(self.config)
        self.lambda_threshold = self.config.get("mmr", {}).get("lambda_threshold", 0.5)

        # Optional RAG generator
        self.rag_enabled = self.config.get("rag", {}).get("enabled", False)
        self.generator = (
            RAGHelper.create_generator(self.config) if self.rag_enabled else None
        )

        logger.info("Initialized Milvus MMR search pipeline")

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
        # Step 1: Embed the query for vector search
        query_result = self.embedder.run(text=query)
        query_embedding = query_result["embedding"]

        # Step 2: Retrieve candidates from Milvus
        # Normalize filters to database-agnostic format
        filters = DocumentFilter.normalize(filters)
        candidates = self.db.query(
            query_embedding=query_embedding,
            top_k=top_k_candidates,
            filter=filters if filters else None,
        )
        logger.info("Retrieved %d candidates", len(candidates))

        # Handle empty results early to avoid unnecessary processing
        if not candidates:
            return {"documents": [], "query": query, "answer": None}

        # Step 3: Apply MMR reranking for diversity
        # MMR selects documents that are both relevant to query
        # and diverse from each other using cosine similarity
        ranked_result = self.mmr_ranker.run(
            query=query,
            documents=candidates,
            top_k=top_k,
        )
        documents = ranked_result["documents"]
        logger.info("MMR reranked to %d documents", len(documents))

        # Build result with metadata for debugging and evaluation
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
