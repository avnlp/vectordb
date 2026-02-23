"""Milvus agentic RAG search pipeline (LangChain).

This module implements the search phase of agentic RAG using Milvus as the
enterprise-grade vector store backend. Milvus's distributed architecture
provides the scalability and advanced search capabilities needed for
large-scale agentic RAG deployments.

Agentic RAG with Milvus:
    The agentic pattern benefits from Milvus's advanced features:
    - Partitioning allows the agent to target specific document subsets
    - Hybrid search (dense + sparse) improves retrieval for complex queries
    - GPU acceleration reduces latency during iterative retrieval
    - Dynamic fields support flexible metadata for agent decision-making

Milvus-Specific Considerations:
    - Collection-based organization similar to Chroma but with enterprise features
    - Dynamic schema enables flexible metadata without migrations
    - Partition key routing can optimize agent queries to relevant data subsets
    - Multi-replica support ensures availability during intensive agent workloads

Architecture Notes:
    Milvus's separation of storage and compute allows independent scaling
    of the vector search layer as agentic RAG query volumes grow.
"""

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_groq import ChatGroq

from vectordb.databases.milvus import MilvusVectorDB
from vectordb.langchain.agentic_rag.base import AgenticRAGPipeline
from vectordb.langchain.components import (
    AgenticRouter,
    ContextCompressor,
)
from vectordb.langchain.utils import (
    ConfigLoader,
    EmbedderHelper,
    RAGHelper,
    RerankerHelper,
)


logger = logging.getLogger(__name__)


class MilvusAgenticRAGPipeline(AgenticRAGPipeline):
    """Milvus agentic RAG pipeline (LangChain).

    Combines agentic routing with vector search, document compression,
    answer reflection, and RAG generation using Milvus backend.

    Milvus's enterprise features make this suitable for production deployments
    requiring high availability, scalability, and advanced search capabilities.

    Attributes:
        config: Loaded and validated configuration dictionary.
        embedder: Embedding model for query vectorization.
        db: MilvusVectorDB instance for distributed document retrieval.
        collection_name: Target Milvus collection for search.
        llm: Language model for RAG generation and reflection.
        max_iterations: Maximum agent reasoning iterations.
        compression_mode: Document compression strategy.
        router: AgenticRouter for action decisions.
        compressor: ContextCompressor for document pruning.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize agentic RAG pipeline from configuration.

        Sets up the Milvus connection, embedding model, LLM, router, and
        document compressor.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain milvus, embedding, rag, and agentic sections.

        Raises:
            ValueError: If RAG LLM is not enabled.
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "milvus")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        # Initialize Milvus connection
        milvus_config = self.config["milvus"]
        self.db = MilvusVectorDB(
            uri=milvus_config.get("uri"),
            token=milvus_config.get("token"),
        )

        self.collection_name = milvus_config.get("collection_name")

        # LLM is required for both generation and agentic routing
        self.llm = RAGHelper.create_llm(self.config)
        if self.llm is None:
            msg = "RAG LLM must be enabled for agentic RAG"
            raise ValueError(msg)

        # Agentic configuration
        agentic_config = self.config.get("agentic", {})
        self.max_iterations = agentic_config.get("max_iterations", 3)
        self.compression_mode = agentic_config.get("compression_mode", "reranking")

        # Initialize router with capable model for action decisions
        router_model = agentic_config.get("router_model", "llama-3.3-70b-versatile")
        self.router = AgenticRouter(
            ChatGroq(
                model=router_model,
                api_key=self.llm.api_key,
                temperature=0.5,
            )
        )

        # Configure document compressor
        if self.compression_mode == "reranking":
            reranker = RerankerHelper.create_reranker(self.config)
            self.compressor = ContextCompressor(mode="reranking", reranker=reranker)
        else:
            self.compressor = ContextCompressor(mode="llm_extraction", llm=self.llm)

        logger.info(
            "Initialized Milvus agentic RAG pipeline (LangChain) "
            "with max_iterations=%d, compression_mode=%s",
            self.max_iterations,
            self.compression_mode,
        )

    def run(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute agentic RAG pipeline.

        Runs the iterative agent loop with Milvus as the retrieval backend.
        Milvus's dynamic schema supports flexible metadata filtering that
        the agent can leverage during its reasoning process.

        Args:
            query: User query text to process.
            top_k: Number of documents to retrieve per search (default 10).
            filters: Optional metadata filters for Milvus query.

        Returns:
            Dictionary containing:
                - final_answer: Generated answer string
                - documents: List of Document objects used in final answer
                - intermediate_steps: List of iteration records with actions/reasoning
                - reasoning: List of router reasoning strings for transparency
        """
        logger.info("Starting agentic RAG for query: %s", query[:100])

        # Track state across iterations
        current_answer: str | None = None
        retrieved_documents: list[Document] = []
        intermediate_steps: list[dict[str, Any]] = []
        iteration = 1

        # Compute query embedding once for reuse
        query_embedding = EmbedderHelper.embed_query(self.embedder, query)

        # Agentic reasoning loop
        while iteration <= self.max_iterations:
            logger.info("Iteration %d/%d", iteration, self.max_iterations)

            # Router decides next action
            decision = self.router.route(
                query=query,
                has_documents=len(retrieved_documents) > 0,
                current_answer=current_answer,
                iteration=iteration,
                max_iterations=self.max_iterations,
            )

            action = decision["action"]
            reasoning = decision["reasoning"]

            step_record = {
                "iteration": iteration,
                "action": action,
                "reasoning": reasoning,
            }

            # Execute action
            if action == "search":
                logger.info("Executing SEARCH action")

                # Retrieve from Milvus collection
                documents = self.db.query(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    filters=filters,
                    collection_name=self.collection_name,
                )
                logger.info("Retrieved %d documents from Milvus", len(documents))

                # Compress documents
                if documents:
                    compressed_docs = self.compressor.compress(
                        query=query,
                        documents=documents,
                        top_k=5,
                    )
                    retrieved_documents = compressed_docs
                    logger.info("Compressed to %d documents", len(compressed_docs))
                else:
                    retrieved_documents = documents

                step_record["documents_retrieved"] = len(retrieved_documents)

            elif action == "reflect":
                logger.info("Executing REFLECT action")

                # If no answer exists yet, generate a draft to reflect on.
                if retrieved_documents and not current_answer:
                    current_answer = RAGHelper.generate(
                        self.llm, query, retrieved_documents
                    )
                    step_record["draft_answer_generated"] = True

                if retrieved_documents and current_answer:
                    documents_context = "\n".join(
                        doc.page_content for doc in retrieved_documents[:3]
                    )
                    reflection_prompt = (
                        f"Given the following documents and answer, is the answer "
                        f"accurate, complete, and well-supported by the documents? "
                        f"Suggest improvements if needed.\n\n"
                        f"Documents:\n"
                        f"{documents_context}\n\n"
                        f"Current Answer: {current_answer}"
                    )

                    reflection = self.llm.invoke(reflection_prompt)
                    logger.info("Reflection: %s", reflection.content[:200])

                    step_record["reflection"] = reflection.content

            elif action == "generate":
                logger.info("Executing GENERATE action")

                if retrieved_documents:
                    current_answer = RAGHelper.generate(
                        self.llm, query, retrieved_documents
                    )
                else:
                    current_answer = self.llm.invoke(query).content

                logger.info("Generated answer: %s", current_answer[:200])
                step_record["answer_generated"] = True

                # Exit loop after generation
                intermediate_steps.append(step_record)
                break

            intermediate_steps.append(step_record)
            iteration += 1

        # Fallback if no generation occurred
        if current_answer is None:
            logger.warning("No answer generated, using LLM fallback")
            current_answer = self.llm.invoke(query).content

        return {
            "final_answer": current_answer,
            "documents": retrieved_documents,
            "intermediate_steps": intermediate_steps,
            "reasoning": [step.get("reasoning") for step in intermediate_steps],
        }
