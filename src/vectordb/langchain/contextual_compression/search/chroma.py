"""Chroma contextual compression search pipeline (LangChain).

This module provides a search pipeline for Chroma vector database with
contextual compression support. The pipeline retrieves documents, applies
compression to filter/summarize content, and optionally generates answers.

Contextual Compression in Search:
    The pipeline implements a two-stage retrieval approach:

    1. Over-fetching: Retrieve 2x the desired number of documents (top_k * 2)
       to ensure good coverage before compression

    2. Compression: Apply reranking or LLM extraction to select the most
       relevant content and reduce to the requested top_k count

    This approach balances recall (finding relevant documents) with precision
    (only using the most relevant content for generation).

Compression Modes:
    - Reranking (default): Uses cross-encoder models to score document
      relevance and return the top_k highest-scoring documents.
      Fast, preserves original text quality.

    - LLM Extraction: Uses an LLM to extract relevant passages from
      retrieved documents. More aggressive compression, adds latency.

Chroma-Specific Features:
    - Collection-based retrieval: Searches within a named collection
    - Metadata filtering: Supports filtering by document metadata
    - No namespace support: Unlike Pinecone, all docs in one collection

Configuration Schema:
    Required configuration sections:
    - chroma: Connection and collection settings
    - embedding: Query embedding model configuration
    - compression: Mode (reranking/llm_extraction) and model settings
    - rag (optional): LLM configuration for answer generation
"""

import logging
from typing import Any

from vectordb.databases.chroma import ChromaVectorDB
from vectordb.langchain.components import ContextCompressor
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
    RerankerHelper,
)


# Module-level logger for search operations
logger = logging.getLogger(__name__)


class ChromaContextualCompressionSearchPipeline:
    """Search pipeline for Chroma with contextual compression.

    This pipeline implements contextual compression-based retrieval for Chroma.
    It retrieves documents, compresses them using reranking or LLM extraction,
    and optionally generates answers using an LLM.

    Attributes:
        config: Loaded and validated configuration dictionary
        embedder: Initialized embedding model for query vectorization
        db: ChromaVectorDB instance for document retrieval
        collection_name: Name of the Chroma collection to search
        compressor: ContextCompressor instance for document compression
        llm: Optional LLM for answer generation (None if not configured)

    Design Decisions:
        - Over-fetch then compress: Retrieve 2x documents, compress to top_k
        - Configurable compression: Reranking (fast) or LLM extraction (thorough)
        - Optional RAG: Answer generation only if LLM configured
        - Graceful degradation: Works without RAG for retrieval-only use cases
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize the search pipeline from configuration.

        Args:
            config_or_path: Either a configuration dictionary or path to a YAML
                configuration file. Must contain 'chroma', 'embedding', and
                optionally 'compression' and 'rag' sections.

        Raises:
            ValueError: If the configuration is invalid or missing required keys.

        Example:
            >>> pipeline = ChromaContextualCompressionSearchPipeline("config.yaml")
            >>> result = pipeline.search("What is AI?", top_k=5)
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "chroma")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        chroma_config = self.config["chroma"]
        self.db = ChromaVectorDB(
            persist_dir=chroma_config.get("persist_dir"),
        )

        self.collection_name = chroma_config.get("collection_name")

        compression_config = self.config.get("compression", {})
        mode = compression_config.get("mode", "reranking")

        if mode == "reranking":
            reranker = RerankerHelper.create_reranker(self.config)
            self.compressor = ContextCompressor(mode="reranking", reranker=reranker)
        else:  # llm_extraction
            llm = RAGHelper.create_llm(self.config)
            if llm is None:
                from langchain_groq import ChatGroq

                llm = ChatGroq(model="llama-3.3-70b-versatile")
            self.compressor = ContextCompressor(mode="llm_extraction", llm=llm)

        self.llm = RAGHelper.create_llm(self.config)

        logger.info(
            "Initialized Chroma contextual compression search pipeline (LangChain)"
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute contextual compression search.

        This method performs the complete search workflow:
        1. Embed the query into a vector
        2. Retrieve 2*top_k documents from Chroma
        3. Compress documents to top_k using configured strategy
        4. Optionally generate an answer using the compressed context

        Args:
            query: The search query text.
            top_k: Number of documents to return after compression.
                Default is 10. The pipeline retrieves 2*top_k initially.
            filters: Optional metadata filters to apply during retrieval.
                Format depends on Chroma's filter syntax.

        Returns:
            Dictionary containing:
                - 'documents': List of compressed Document objects
                - 'query': The original query string
                - 'answer': Generated answer (only if LLM configured)

        Example:
            >>> result = pipeline.search("What is machine learning?", top_k=5)
            >>> print(f"Found {len(result['documents'])} documents")
            >>> if "answer" in result:
            ...     print(f"Answer: {result['answer']}")
        """
        logger.info("Starting contextual compression search")

        query_embedding = EmbedderHelper.embed_query(self.embedder, query)
        logger.info("Embedded query: %s", query[:50])

        # Over-fetch documents to enable compression filtering
        retrieved_documents = self.db.query(
            query_embedding=query_embedding,
            top_k=top_k * 2,
            filters=filters,
            collection_name=self.collection_name,
        )
        logger.info("Retrieved %d documents from Chroma", len(retrieved_documents))

        # Apply contextual compression: reranking or LLM extraction
        compressed_documents = self.compressor.compress(
            query=query,
            documents=retrieved_documents,
            top_k=top_k,
        )
        logger.info("Compressed to %d documents", len(compressed_documents))

        result = {
            "documents": compressed_documents,
            "query": query,
        }

        if self.llm is not None:
            answer = RAGHelper.generate(self.llm, query, compressed_documents)
            result["answer"] = answer
            logger.info("Generated RAG answer")

        return result
