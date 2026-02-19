"""Chroma semantic search pipeline with optional RAG generation.

This pipeline implements semantic search using Chroma's embedded vector database,
with support for local development and small-to-medium scale deployments.

Chroma-Specific Features:
    - Local embedded vector database with SQLite persistence
    - Simple API with collection-based organization
    - Supports cosine similarity search (default metric)
    - Good for development and small-to-medium datasets
    - No external service dependencies

Search Pipeline Flow:
    1. Embed query: Convert query text to dense vector using configured embedder
    2. Over-fetch: Retrieve 2x top_k documents for diversification pool
    3. Filter: Apply metadata filters to refine results
    4. Diversify: Apply result diversification for better coverage
    5. Trim: Return top_k most relevant and diverse documents
    6. Generate: Optionally create RAG answer using retrieved context

Configuration (YAML):
    Required sections:
        - chroma.host: Chroma server host (default: localhost)
        - chroma.port: Chroma server port (default: 8000)
        - chroma.collection_name: Target collection for search
        - embeddings.model: HuggingFace model for query embedding

    Optional settings:
        - chroma.persist_directory: Local storage path for persistent mode
        - rag.enabled: Enable LLM answer generation
        - rag.generator_model: Model for answer synthesis

    Example config:
        chroma:
          host: "localhost"
          port: 8000
          collection_name: "semantic-search"
          persist_directory: "./chroma_data"
        embeddings:
          model: "sentence-transformers/all-MiniLM-L6-v2"
        rag:
          enabled: true
          generator_model: "gpt-4o-mini"

Usage:
    >>> from vectordb.haystack.semantic_search import ChromaSemanticSearchPipeline
    >>> pipeline = ChromaSemanticSearchPipeline("config.yaml")
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

    Both implementations use the same underlying ChromaVectorDB class for
database operations, ensuring consistent behavior across frameworks.

Note:
    Chroma is ideal for local development and smaller datasets.
    For production at scale, consider migrating to Pinecone, Milvus, or Qdrant.
"""

import logging
from typing import Any

from vectordb import ChromaVectorDB
from vectordb.haystack.utils import (
    ConfigLoader,
    DiversificationHelper,
    DocumentFilter,
    EmbedderFactory,
    RAGHelper,
)


logger = logging.getLogger(__name__)


class ChromaSemanticSearchPipeline:
    """Chroma semantic search pipeline with RAG support.

    This pipeline implements a semantic search workflow with optional
    RAG answer generation. It handles query embedding, vector search,
    result filtering, diversification, and LLM-based synthesis.

    Attributes:
        config: Validated configuration dictionary.
        embedder: Haystack text embedder component for query encoding.
        db: ChromaVectorDB instance for database operations.
        collection_name: Name of the Chroma collection to search.
        rag_enabled: Whether RAG answer generation is enabled.
        generator: Optional LLM generator for RAG answers.

    Note:
        Chroma stores data locally in persist_directory. For production
        deployments, consider migrating to a server-based database.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize search pipeline from configuration.

        Args:
            config_or_path: Config dict or path to YAML file.

        Raises:
            ValueError: If required config missing.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "chroma")

        self.embedder = EmbedderFactory.create_text_embedder(self.config)

        chroma_config = self.config["chroma"]
        self.db = ChromaVectorDB(
            host=chroma_config.get("host", "localhost"),
            port=chroma_config.get("port", 8000),
        )
        self.collection_name = chroma_config["collection_name"]

        # Optional RAG generator
        self.rag_enabled = self.config.get("rag", {}).get("enabled", False)
        self.generator = (
            RAGHelper.create_generator(self.config) if self.rag_enabled else None
        )

        logger.info("Initialized Chroma search pipeline")

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

        # Search Chroma
        filters = DocumentFilter.normalize(filters)
        documents = self.db.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,
            collection_name=self.collection_name,
            where=filters if filters else None,
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
