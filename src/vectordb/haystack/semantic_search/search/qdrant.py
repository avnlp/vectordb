"""Qdrant semantic search pipeline.

This pipeline implements semantic search using Qdrant vector database,
with support for high-performance dense vector retrieval, rich metadata
filtering, and optional RAG answer generation.

Qdrant-Specific Features:
    - High-performance vector search with HNSW indexing
    - Rich metadata filtering with MongoDB-style query syntax
    - Supports both local (in-memory/disk) and server deployments
    - Payload storage for document content and metadata
    - Quantization support for memory-efficient storage

Search Pipeline Flow:
    1. Embed query: Convert query text to dense vector using configured embedder
    2. Over-fetch: Retrieve 2x top_k documents for diversification pool
    3. Filter: Apply metadata filters to refine results
    4. Diversify: Apply result diversification for better coverage
    5. Trim: Return top_k most relevant and diverse documents
    6. Generate: Optionally create RAG answer using retrieved context

Configuration (YAML):
    Required sections:
        - qdrant.url: Qdrant server URL (e.g., "http://localhost:6333")
        - qdrant.collection_name: Target collection for search
        - embeddings.model: HuggingFace model for query embedding

    Optional settings:
        - qdrant.api_key: API key for authenticated servers
        - qdrant.prefer_grpc: Use gRPC protocol (default: True)
        - rag.enabled: Enable LLM answer generation
        - rag.generator_model: Model for answer synthesis

    Example config:
        qdrant:
          url: "http://localhost:6333"
          api_key: ""  # Empty for local Qdrant
          collection_name: "semantic-search"
          prefer_grpc: true
        embeddings:
          model: "sentence-transformers/all-MiniLM-L6-v2"
        rag:
          enabled: true
          generator_model: "gpt-4o-mini"

Usage:
    >>> from vectordb.haystack.semantic_search import QdrantSemanticSearchPipeline
    >>> pipeline = QdrantSemanticSearchPipeline("config.yaml")
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

    Both implementations use the same underlying QdrantVectorDB class for
database operations, ensuring consistent behavior across frameworks.
"""

import logging
from typing import Any

from vectordb import QdrantVectorDB
from vectordb.haystack.utils import (
    ConfigLoader,
    DiversificationHelper,
    DocumentFilter,
    EmbedderFactory,
    RAGHelper,
)


logger = logging.getLogger(__name__)


class QdrantSemanticSearchPipeline:
    """Qdrant semantic search pipeline.

    Embeds query, retrieves from Qdrant, optionally reranks and generates RAG answer.

    This pipeline implements a complete semantic search workflow:
    1. Query embedding using configured text embedder
    2. Dense vector search in Qdrant collection
    3. Metadata filtering and result diversification
    4. Optional LLM-based answer generation

    Attributes:
        config: Validated configuration dictionary.
        embedder: Haystack text embedder component for query encoding.
        db: QdrantVectorDB instance for database operations.
        collection_name: Name of the Qdrant collection to search.
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
            url=qdrant_config.get("url", "http://localhost:6333"),
            api_key=qdrant_config.get("api_key", ""),
        )
        self.collection_name = qdrant_config["collection_name"]

        # Optional RAG generator
        self.rag_enabled = self.config.get("rag", {}).get("enabled", False)
        self.generator = (
            RAGHelper.create_generator(self.config) if self.rag_enabled else None
        )

        logger.info("Initialized Qdrant search pipeline")

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

        # Search Qdrant
        filters = DocumentFilter.normalize(filters)
        documents = self.db.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,
            collection_name=self.collection_name,
            filters=filters if filters else None,
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
