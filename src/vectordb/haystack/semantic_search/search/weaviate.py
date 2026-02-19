"""Weaviate semantic search pipeline.

This pipeline implements semantic search using Weaviate vector database,
with support for schema-based vector search, hybrid retrieval, and
optional RAG answer generation.

Weaviate-Specific Features:
    - Schema-based vector search with GraphQL interface
    - Hybrid search combining vector and BM25 keyword ranking
    - Modular AI integrations (vectorization, generative, qna)
    - Multi-tenancy with tenant isolation
    - Query-time reranking

Search Pipeline Flow:
    1. Embed query: Convert query text to dense vector using configured embedder
    2. Over-fetch: Retrieve 2x top_k documents for diversification pool
    3. Filter: Apply metadata filters to refine results
    4. Diversify: Apply result diversification for better coverage
    5. Trim: Return top_k most relevant and diverse documents
    6. Generate: Optionally create RAG answer using retrieved context

Configuration (YAML):
    Required sections:
        - weaviate.url: Weaviate server URL
        - weaviate.api_key: Authentication credentials
        - weaviate.class_name: Target class/collection for search
        - embeddings.model: HuggingFace model for query embedding

    Optional settings:
        - rag.enabled: Enable LLM answer generation
        - rag.generator_model: Model for answer synthesis

    Example config:
        weaviate:
          url: "https://my-cluster.weaviate.cloud"
          api_key: "${WEAVIATE_API_KEY}"
          class_name: "SemanticSearch"
        embeddings:
          model: "sentence-transformers/all-MiniLM-L6-v2"
        rag:
          enabled: true
          generator_model: "gpt-4o-mini"

Usage:
    >>> from vectordb.haystack.semantic_search import WeaviateSemanticSearchPipeline
    >>> pipeline = WeaviateSemanticSearchPipeline("config.yaml")
    >>> results = pipeline.search("machine learning applications", top_k=10)
    >>> print(f"Retrieved {len(results['documents'])} documents")
    >>> if "answer" in results:
    ...     print(f"RAG Answer: {results['answer']}")

Comparison with LangChain:
    Haystack Integration (this module):
        - Pipeline-based architecture with explicit component connections
        - Native Haystack Document format throughout
        - Built-in RAG prompt templates and generators
        - Easier to customize with Haystack's component ecosystem

    LangChain Integration (vectordb.langchain):
        - Chain-based composition with LCEL
        - LangChain Document format with different metadata conventions
        - More flexible but requires more boilerplate for standard RAG
        - Better integration with LangChain's agent and tool ecosystem

    Both implementations use the same underlying WeaviateVectorDB class for
database operations, ensuring consistent behavior across frameworks.
"""

import logging
from typing import Any

from vectordb import WeaviateVectorDB
from vectordb.haystack.utils import (
    ConfigLoader,
    DiversificationHelper,
    DocumentFilter,
    EmbedderFactory,
    RAGHelper,
)


logger = logging.getLogger(__name__)


class WeaviateSemanticSearchPipeline:
    """Weaviate semantic search pipeline.

    Embeds query, retrieves from Weaviate, optionally reranks and generates RAG answer.

    This pipeline implements a complete semantic search workflow:
    1. Query embedding using configured text embedder
    2. Dense vector search in Weaviate class/collection
    3. Metadata filtering and result diversification
    4. Optional LLM-based answer generation

    Attributes:
        config: Validated configuration dictionary.
        embedder: Haystack text embedder component for query encoding.
        db: WeaviateVectorDB instance for database operations.
        class_name: Name of the Weaviate class/collection to search.
        rag_enabled: Whether RAG answer generation is enabled.
        generator: Optional LLM generator for RAG answers.

    Note:
        Weaviate combines vector search with semantic capabilities through
        GraphQL, enabling complex queries beyond simple vector similarity.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize search pipeline from configuration.

        Args:
            config_or_path: Config dict or path to YAML file.

        Raises:
            ValueError: If required config missing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "weaviate")

        self.embedder = EmbedderFactory.create_text_embedder(self.config)

        weaviate_config = self.config["weaviate"]
        self.db = WeaviateVectorDB(
            url=weaviate_config.get("url", "http://localhost:8080"),
            api_key=weaviate_config.get("api_key", ""),
        )
        self.class_name = weaviate_config["class_name"]

        # Optional RAG generator
        self.rag_enabled = self.config.get("rag", {}).get("enabled", False)
        self.generator = (
            RAGHelper.create_generator(self.config) if self.rag_enabled else None
        )

        logger.info("Initialized Weaviate search pipeline")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute semantic search.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            Dict with 'documents', 'query', and optional 'answer' keys.
        """
        # Embed query
        query_result = self.embedder.run(text=query)
        query_embedding = query_result["embedding"]

        # Search Weaviate
        filters = DocumentFilter.normalize(filters)
        documents = self.db.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,
            class_name=self.class_name,
            where_filter=filters if filters else None,
        )
        logger.info("Retrieved %d documents", len(documents))

        # Apply filters and diversification
        documents = DocumentFilter.apply(documents, filters)
        documents = DiversificationHelper.apply(documents, self.config)
        documents = documents[:top_k]

        result: dict[str, Any] = {
            "documents": documents,
            "query": query,
        }

        # Optional RAG
        if self.rag_enabled and self.generator and documents:
            prompt = RAGHelper.format_prompt(query, documents)
            gen_result = self.generator.run(prompt=prompt)
            result["answer"] = gen_result.get("replies", [""])[0]

        return result
