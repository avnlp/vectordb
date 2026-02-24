"""Pinecone MMR search pipeline.

This pipeline performs diversity-aware retrieval using Maximal Marginal Relevance.
It embeds the query, retrieves candidates from Pinecone, applies MMR reranking,
and optionally generates a RAG answer.

Pinecone-Specific Features:
    - Supports metadata filtering via filter parameter
    - Namespace support for logical partitioning
    - Real-time index updates (no reindexing required)
    - Serverless and pod-based deployment options

MMR Algorithm Steps:
    1. Embed query: Convert query text to dense vector
    2. Retrieve candidates: Get top_k_candidates from Pinecone
       (uses cosine/dotproduct/euclidean similarity based on index metric)
    3. Apply MMR reranking:
       a. For each candidate, calculate similarity to query (relevance)
       b. Calculate max similarity to already-selected documents (redundancy)
       c. Score: λ×relevance - (1-λ)×redundancy
       d. Greedily select highest-scoring document
    4. Return top_k diverse results

Lambda Parameter Guide:
    - λ = 0.7-0.9: Focus on relevance, minimal diversity
    - λ = 0.5: Balanced relevance and diversity (recommended default)
    - λ = 0.1-0.3: Focus on diversity, useful for exploration

Configuration (YAML):
    - pinecone.api_key: Pinecone API key
    - pinecone.index_name: Index to search
    - pinecone.namespace: Optional namespace for search scope
    - mmr.lambda_threshold: MMR lambda parameter (default 0.5)
    - rag.enabled: Whether to generate RAG answers

Usage:
    >>> pipeline = PineconeMmrSearchPipeline("config.yaml")
    >>> results = pipeline.search("machine learning applications", top_k=10)
    >>> # results = {"documents": [...], "query": "...", "answer": "..."}
"""

import logging
from typing import Any

from vectordb.databases.pinecone import PineconeVectorDB
from vectordb.haystack.utils import (
    ConfigLoader,
    DocumentFilter,
    EmbedderFactory,
    RAGHelper,
    RerankerFactory,
)


logger = logging.getLogger(__name__)


class PineconeMmrSearchPipeline:
    """Pinecone MMR search pipeline for diversity-aware retrieval.

    Embeds query, retrieves from Pinecone, applies MMR reranking,
    and optionally generates RAG answer.

    This pipeline implements the standard MMR search pattern with
    Pinecone-specific namespace support for logical partitioning.

    Attributes:
        config: Validated configuration dictionary.
        embedder: Haystack text embedder component.
        db: PineconeVectorDB instance for database operations.
        index_name: Name of the Pinecone index.
        namespace: Optional namespace for search scope.
        mmr_ranker: Diversity ranker for MMR reranking.
        lambda_threshold: MMR lambda parameter (0.0-1.0).
        rag_enabled: Whether RAG answer generation is enabled.
        generator: Optional LLM generator for RAG answers.

    Note:
        Namespaces allow logical separation of documents within a single
        index, enabling multi-tenant or multi-dataset scenarios.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize search pipeline from configuration.

        Args:
            config_or_path: Config dict or path to YAML file.

        Raises:
            ValueError: If required config missing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "pinecone")

        self.embedder = EmbedderFactory.create_text_embedder(self.config)

        pinecone_config = self.config["pinecone"]
        self.db = PineconeVectorDB(
            api_key=pinecone_config["api_key"],
            index_name=pinecone_config.get("index_name"),
        )
        self.index_name = pinecone_config.get("index_name")
        self.namespace = pinecone_config.get("namespace", "")

        self.mmr_ranker = RerankerFactory.create_diversity_ranker(self.config)
        self.lambda_threshold = self.config.get("mmr", {}).get("lambda_threshold", 0.5)

        # Optional RAG generator
        self.rag_enabled = self.config.get("rag", {}).get("enabled", False)
        self.generator = (
            RAGHelper.create_generator(self.config) if self.rag_enabled else None
        )

        logger.info("Initialized Pinecone MMR search pipeline")

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

        # Step 2: Retrieve candidates from Pinecone
        # Namespace restricts search to a logical partition within the index
        filters = DocumentFilter.normalize(filters)
        candidates = self.db.query(
            query_embedding=query_embedding,
            top_k=top_k_candidates,
            namespace=self.namespace,
            filter=filters if filters else None,
        )
        logger.info("Retrieved %d candidates", len(candidates))

        if not candidates:
            return {"documents": [], "query": query, "answer": None}

        # Step 3: Apply MMR reranking
        # MMR balances relevance (query similarity) with diversity
        # Uses cosine similarity for relevance and redundancy calculation
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
