"""Chroma search pipeline with diversity filtering and RAG.

Implements search with Maximum Margin Relevance (MMR) diversity filtering
for Chroma vector database, with optional RAG answer generation.

Pipeline Flow:
1. Embed query using SentenceTransformersTextEmbedder (initialized once)
2. Retrieve top_k_candidates from Chroma using vector similarity
3. Apply MMR diversity ranking to select diverse subset
4. Optionally generate RAG answer using diverse documents

Diversity Filtering:
Uses SentenceTransformersDiversityRanker with configurable similarity metric
(cosine or dot_product) and top_k parameter. The MMR algorithm balances
query relevance against inter-document diversity.

RAG Generation:
When enabled, formats diverse documents using dataset-specific prompts and
generates answers via OpenAIGenerator (Groq or OpenAI providers).

Configuration:
Pipeline behavior controlled via YAML config with Chroma connection params
(host, port, persistence), retrieval settings, diversity algorithm options,
and RAG settings. Supports both ephemeral and persistent Chroma modes.

Example:
    >>> from vectordb.haystack.diversity_filtering.pipelines.chroma_search import (
    ...     ChromaDiversitySearchPipeline,
    ... )
    >>> pipeline = ChromaDiversitySearchPipeline("config.yaml")
    >>> results = pipeline.search(query="machine learning applications")
    >>> print(f"Retrieved {len(results['documents'])} diverse documents")
"""

import logging
from typing import Any

from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.rankers import SentenceTransformersDiversityRanker

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.haystack.diversity_filtering.rankers import ClusteringDiversityRanker
from vectordb.haystack.diversity_filtering.utils.config_loader import (
    ConfigLoader,
)
from vectordb.haystack.diversity_filtering.utils.prompts import (
    format_documents,
    get_prompt_template,
)


logger = logging.getLogger(__name__)


class ChromaDiversitySearchPipeline:
    """Chroma search pipeline with diversity filtering and RAG support.

    This pipeline implements a search workflow with optional diversity filtering
    using Maximum Margin Relevance (MMR) and RAG answer generation. Components
    are initialized once in __init__ and reused across multiple search calls.

    Attributes:
        config: Validated configuration object.
        embedder: SentenceTransformersTextEmbedder for query embedding.
        db: ChromaVectorDB instance for document storage and retrieval.
        ranker: Optional SentenceTransformersDiversityRanker for MMR filtering.
        rag_enabled: Whether RAG answer generation is enabled.
        generator: Optional OpenAIGenerator for RAG answer synthesis.
        prompt_template: Template string for RAG prompt construction.

    Performance:
        Components are initialized once during construction, avoiding the
        overhead of reloading embedding models on each search request.
        This reduces latency from seconds to milliseconds for repeated searches.

    Example:
        >>> pipeline = ChromaDiversitySearchPipeline("config.yaml")
        >>> result1 = pipeline.search("query one")
        >>> result2 = pipeline.search("query two")  # Reuses embedder
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize search pipeline from configuration.

        Args:
            config_or_path: Config dict or path to YAML file.

        Raises:
            FileNotFoundError: If config file not found.
            ValueError: If configuration invalid.
        """
        self.config = ConfigLoader.load(config_or_path)

        # Initialize embedder once - avoids reloading on each search
        self.embedder = SentenceTransformersTextEmbedder(
            model=self.config.embedding.model,
            device=self.config.embedding.device,
        )
        self.embedder.warm_up()

        # Initialize ChromaDB connection
        self.db = ChromaVectorDB(
            host=self.config.vectordb.chroma.host,
            port=self.config.vectordb.chroma.port,
            index=self.config.index.name,
            embedding_dim=self.config.embedding.dimension,
            is_persistent=self.config.vectordb.chroma.is_persistent,
        )

        self.ranker = None
        if self.config.diversity.algorithm in (
            "maximum_margin_relevance",
            "greedy_diversity_order",
        ):
            self.ranker = SentenceTransformersDiversityRanker(
                model=self.config.embedding.model,
                top_k=self.config.diversity.top_k,
                similarity="cosine"
                if self.config.diversity.similarity_metric == "cosine"
                else "dot_product",
                strategy=self.config.diversity.algorithm,
            )
            self.ranker.warm_up()
            logger.info(
                "Initialized %s diversity ranker with %s similarity",
                self.config.diversity.algorithm,
                self.config.diversity.similarity_metric,
            )
        elif self.config.diversity.algorithm == "clustering":
            self.ranker = ClusteringDiversityRanker(
                model=self.config.embedding.model,
                top_k=self.config.diversity.top_k,
                similarity=self.config.diversity.similarity_metric,
            )
            self.ranker.warm_up()
            logger.info(
                "Initialized ClusteringDiversityRanker with %s similarity",
                self.config.diversity.similarity_metric,
            )

        # Initialize RAG components if enabled
        self.rag_enabled = self.config.rag.enabled
        self.generator = None
        self.prompt_template = None

        if self.rag_enabled:
            self.generator = OpenAIGenerator(
                api_key_env_var=f"{self.config.rag.provider.upper()}_API_KEY",
                model=self.config.rag.model,
                generation_kwargs={
                    "temperature": self.config.rag.temperature,
                    "max_tokens": self.config.rag.max_tokens,
                },
            )
            self.prompt_template = get_prompt_template(self.config.dataset.name)
            logger.info(
                "Initialized RAG generator with model %s", self.config.rag.model
            )

        logger.info("Initialized Chroma diversity search pipeline")

    def search(self, query: str) -> dict[str, Any]:
        """Execute search with diversity filtering and optional RAG.

        Args:
            query: Search query string.

        Returns:
            Dictionary with search results including diverse documents and
            optional answer.
        """
        # Embed query using pre-initialized embedder
        query_embedding = self.embedder.run(text=query)["embedding"]

        # Retrieve candidates from Chroma
        candidates = self.db.search(
            query_embedding=query_embedding,
            top_k=self.config.retrieval.top_k_candidates,
        )

        if not candidates:
            logger.warning("No candidates retrieved for query: %s", query)
            return {
                "documents": [],
                "num_diverse": 0,
                "answer": None,
                "query": query,
            }

        # Apply diversity filtering
        if self.ranker is not None:
            diverse_docs = self.ranker.run(documents=candidates, query=query)[
                "documents"
            ]
        else:
            diverse_docs = candidates[: self.config.diversity.top_k]

        result: dict[str, Any] = {
            "documents": [
                {
                    "content": doc.content,
                    "meta": doc.meta,
                    "score": getattr(doc, "score", None),
                }
                for doc in diverse_docs
            ],
            "num_diverse": len(diverse_docs),
            "query": query,
            "answer": None,
        }

        # Generate RAG answer if enabled
        if self.rag_enabled and self.generator:
            try:
                doc_content = format_documents(
                    [{"content": doc.content, "meta": doc.meta} for doc in diverse_docs]
                )

                prompt_builder = PromptBuilder(template=self.prompt_template)
                prompt = prompt_builder.run(query=query, documents=doc_content)[
                    "prompt"
                ]

                response = self.generator.run(prompt=prompt)
                result["answer"] = response.get("replies", [None])[0]

            except Exception as e:
                logger.exception("Error generating RAG answer: %s", e)
                result["answer"] = (
                    "Unable to generate an answer at this time. Please try again later."
                )

        return result
