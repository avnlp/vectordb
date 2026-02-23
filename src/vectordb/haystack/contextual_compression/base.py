"""Base class for contextual compression pipelines.

Contextual compression filters retrieved documents to retain only relevant chunks,
reducing token usage for LLM context windows. This module provides the abstract
base class that orchestrates the compression workflow.

Compression Pipeline Flow:
    1. Dense retrieval: Fetch top_k*2 documents via vector similarity search.
    2. Compression: Apply reranking or LLM extraction to filter documents.
    3. Return: Top-k most relevant documents after compression.

Compression Types:
    - Reranking: Cross-encoder or API-based models score document relevance.
    - LLM Extraction: LLM extracts only query-relevant text passages.

Subclasses must implement database-specific connection and retrieval methods.
"""

from abc import ABC, abstractmethod
from typing import Any

from haystack import Document
from haystack.components.embedders import SentenceTransformersTextEmbedder

from vectordb.haystack.contextual_compression.compression_utils import CompressorFactory
from vectordb.haystack.json_indexing.common.config import load_config
from vectordb.utils.config import setup_logger


class BaseContextualCompressionPipeline(ABC):
    """Abstract base class for contextual compression pipelines.

    Provides shared initialization, embedder setup, compression orchestration,
    and search logic. Subclasses implement database-specific connection and retrieval.

    Attributes:
        config: Configuration dictionary loaded from YAML.
        logger: Logger instance for the pipeline.
        dense_embedder: Text embedder for query embeddings.
        compressor: Initialized compressor (reranker or LLM extractor).
    """

    def __init__(self, config_path: str) -> None:
        """Initialize contextual compression pipeline from configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        self.config = load_config(config_path)
        self.logger = setup_logger(self.config)
        self._init_embedders()
        self._connect()
        self._ensure_collection_ready()
        self._init_compressor()

    def _init_embedders(self) -> None:
        """Initialize dense embedder from configuration."""
        embeddings_config = self.config.get("embeddings", {})
        dense_model = embeddings_config.get("model", "Qwen/Qwen3-Embedding-0.6B")

        # Support model aliases
        model_aliases = {
            "qwen3": "Qwen/Qwen3-Embedding-0.6B",
            "minilm": "sentence-transformers/all-MiniLM-L6-v2",
        }
        dense_model = model_aliases.get(dense_model.lower(), dense_model)

        self.dense_embedder = SentenceTransformersTextEmbedder(model=dense_model)
        self.dense_embedder.warm_up()
        self.logger.info("Initialized dense embedder with model: %s", dense_model)

    @abstractmethod
    def _connect(self) -> None:
        """Establish connection to the vector database.

        Subclasses must implement database-specific connection logic.
        """

    @abstractmethod
    def _ensure_collection_ready(self) -> None:
        """Verify/prepare the database collection for retrieval.

        Subclasses must implement database-specific collection preparation.
        """

    @abstractmethod
    def _retrieve_base_results(self, query: str, top_k: int) -> list[Document]:
        """Retrieve base results from vector database.

        Args:
            query: The search query text.
            top_k: Number of results to retrieve before compression.

        Returns:
            List of retrieved Document objects.
        """

    def _init_compressor(self) -> None:
        """Initialize compressor from configuration using CompressorFactory.

        Creates either a reranker or LLM extractor based on config.
        """
        try:
            self.compressor = CompressorFactory.create_compressor(self.config)
            compression_type = self.config.get("compression", {}).get(
                "type", "reranking"
            )
            self.logger.info("Initialized compressor: %s", compression_type)
        except Exception as e:
            self.logger.error("Failed to initialize compressor: %s", str(e))
            raise

    def run(self, query: str, top_k: int = 10) -> dict[str, Any]:
        """Execute compression pipeline query.

        Retrieves documents from database and applies compression.

        Args:
            query: The search query text.
            top_k: Number of top results to return after compression.

        Returns:
            Dictionary with 'documents' key containing compressed documents.
        """
        self.logger.info("Running compression pipeline for query: %s", query)

        try:
            # Step 1: Dense retrieval
            # Retrieve more documents than needed (top_k * 2) to create a candidate pool
            # for compression algorithms. This ensures high recall before filtering.
            retrieval_config = self.config.get("retrieval", {})
            retrieval_top_k = retrieval_config.get("top_k", top_k * 2)
            base_docs = self._retrieve_base_results(query, retrieval_top_k)

            if not base_docs:
                self.logger.info("No documents retrieved")
                return {"documents": []}

            # Step 2: Compression (reranking or LLM extraction)
            # Reranking: Cross-encoder/API scores docs, returns top-k by relevance
            # LLM Extraction: LLM extracts only relevant passages per document
            # Both approaches reduce token count while preserving relevance
            compressed = self.compressor.run(
                query=query,
                documents=base_docs,
            )

            # Extract documents from compressor output
            result_docs = compressed.get("documents", [])
            final_docs = result_docs[:top_k]

            self.logger.info(
                "Compressed %d documents to %d", len(base_docs), len(final_docs)
            )
            return {"documents": final_docs}

        except Exception as e:
            self.logger.error("Error during compression pipeline: %s", str(e))
            return {"documents": []}

    def evaluate(
        self, questions: list[str], ground_truths: list[str]
    ) -> dict[str, Any]:
        """Evaluate compression quality using metrics.

        Runs the compression pipeline for each question and collects results.

        Args:
            questions: List of query questions.
            ground_truths: List of ground truth answers.

        Returns:
            Dictionary with evaluation results including:
            - questions: Number of questions evaluated
            - metrics: Dictionary with evaluation metrics
            - results: List of results for each question
        """
        self.logger.info("Evaluating pipeline on %d questions", len(questions))

        results = []
        for question, ground_truth in zip(questions, ground_truths):
            # Run the compression pipeline for each question
            result = self.run(query=question)
            results.append(
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "retrieved_documents": result.get("documents", []),
                }
            )

        self.logger.info("Evaluation completed for %d questions", len(questions))

        return {
            "questions": len(questions),
            "metrics": {},
            "results": results,
        }
