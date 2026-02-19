"""Pinecone semantic search pipeline.

This pipeline implements semantic search using Pinecone's managed vector database,
with support for dense vector retrieval, metadata filtering, and optional RAG
answer generation.

Pinecone-Specific Features:
    - Fully managed cloud-native vector database (no index tuning needed)
    - Real-time index updates without reindexing
    - Namespace support for logical data partitioning
    - Metadata filtering with automatic flattening for nested structures
    - Serverless and pod-based deployment options

Search Pipeline Flow:
    1. Embed query: Convert query text to dense vector using configured embedder
    2. Over-fetch: Retrieve 2x top_k documents for diversification pool
    3. Filter: Apply metadata filters to refine results
    4. Diversify: Apply result diversification (MMR or similar) for better coverage
    5. Trim: Return top_k most relevant and diverse documents
    6. Generate: Optionally create RAG answer using retrieved context

Configuration (YAML):
    Required sections:
        - pinecone.api_key: Pinecone API authentication
        - pinecone.index_name: Target index for search
        - embeddings.model: HuggingFace model for query embedding

    Optional settings:
        - pinecone.namespace: Logical partition within index
        - pinecone.metric: Similarity metric (cosine, dotproduct, euclidean)
        - rag.enabled: Enable LLM answer generation
        - rag.generator_model: Model for answer synthesis

    Example config:
        pinecone:
          api_key: "${PINECONE_API_KEY}"
          index_name: "semantic-search"
          namespace: "production"
          metric: "cosine"
        embeddings:
          model: "sentence-transformers/all-MiniLM-L6-v2"
        rag:
          enabled: true
          generator_model: "gpt-4o-mini"

Usage:
    >>> from vectordb.haystack.semantic_search import PineconeSemanticSearchPipeline
    >>> pipeline = PineconeSemanticSearchPipeline("config.yaml")
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
        - Chain-based composition with LCEL (LangChain Expression Language)
        - LangChain Document format with different metadata conventions
        - More flexible but requires more boilerplate for standard RAG
        - Better integration with LangChain's agent and tool ecosystem

    Both implementations use the same underlying PineconeVectorDB class for
database operations, ensuring consistent behavior across frameworks.
"""

import logging
from typing import Any

from vectordb import PineconeVectorDB
from vectordb.haystack.utils import (
    ConfigLoader,
    DiversificationHelper,
    DocumentFilter,
    EmbedderFactory,
    RAGHelper,
)


logger = logging.getLogger(__name__)


class PineconeSemanticSearchPipeline:
    """Pinecone semantic search pipeline.

    Embeds query, retrieves from Pinecone, optionally reranks and generates RAG answer.

    This pipeline implements a complete semantic search workflow:
    1. Query embedding using configured text embedder
    2. Dense vector search in Pinecone index
    3. Metadata filtering and result diversification
    4. Optional LLM-based answer generation

    Attributes:
        config: Validated configuration dictionary.
        embedder: Haystack text embedder component for query encoding.
        db: PineconeVectorDB instance for database operations.
        index_name: Name of the Pinecone index to search.
        namespace: Optional namespace for logical data partitioning.
        rag_enabled: Whether RAG answer generation is enabled.
        generator: Optional LLM generator for RAG answers.

    Note:
        The pipeline over-fetches (2x top_k) before diversification to ensure
        the final results have both high relevance and good diversity.
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

        # Optional RAG generator
        self.rag_enabled = self.config.get("rag", {}).get("enabled", False)
        self.generator = (
            RAGHelper.create_generator(self.config) if self.rag_enabled else None
        )

        logger.info("Initialized Pinecone search pipeline")

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

        # Search Pinecone
        filters = DocumentFilter.normalize(filters)
        documents = self.db.query(
            query_embedding=query_embedding,
            top_k=top_k * 2,
            namespace=self.namespace,
            filter=filters if filters else None,
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
