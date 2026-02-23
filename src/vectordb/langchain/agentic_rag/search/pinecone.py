"""Pinecone agentic RAG search pipeline (LangChain).

This module implements the search phase of agentic RAG using Pinecone as the
managed vector store backend. Pinecone's cloud-native infrastructure provides
the reliability and performance required for production agentic RAG deployments.

Agentic RAG with Pinecone:
    The agentic pattern leverages Pinecone's low-latency search for iterative
    retrieval during agent reasoning. While the current implementation computes
    the query embedding once, future enhancements could use the agent's reflection
    to reformulate queries and retrieve with different embeddings.

Pinecone-Specific Considerations:
    - Namespace support enables multi-tenant agentic RAG where different agents
      access different document sets within the same index
    - Metadata filtering allows the agent to apply dynamic filters based on
      reasoning about query requirements
    - Serverless indexes auto-scale to handle variable agent query loads

Architecture Notes:
    Pinecone's network latency is mitigated by:
    1. Single query embedding computation reused across iterations
    2. Document compression reducing context window processing time
    3. Efficient batch retrieval minimizing API round-trips
"""

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_groq import ChatGroq

from vectordb.databases.pinecone import PineconeVectorDB
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


class PineconeAgenticRAGPipeline(AgenticRAGPipeline):
    """Pinecone agentic RAG pipeline (LangChain).

    Combines agentic routing with vector search, document compression,
    answer reflection, and RAG generation using Pinecone backend. The agent
    makes routing decisions at each step to determine whether to search,
    reflect, or generate.

    Pinecone's managed infrastructure provides production-grade reliability
    for agentic RAG systems that require consistent low-latency retrieval
    across multiple iterations.

    Attributes:
        config: Loaded and validated configuration dictionary.
        embedder: Embedding model for query vectorization.
        db: PineconeVectorDB instance for cloud document retrieval.
        index_name: Target Pinecone index for search.
        namespace: Logical namespace for document isolation.
        llm: Language model for RAG generation and reflection.
        max_iterations: Maximum agent reasoning iterations.
        compression_mode: Document compression strategy.
        router: AgenticRouter for action decisions.
        compressor: ContextCompressor for document pruning.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize agentic RAG pipeline from configuration.

        Sets up the Pinecone connection, embedding model, LLM, router, and
        document compressor. Validates that RAG LLM is enabled since it's
        required for both generation and agentic routing.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain pinecone, embedding, rag, and agentic sections.

        Raises:
            ValueError: If RAG LLM is not enabled (required for agentic routing).

        Example:
            >>> pipeline = PineconeAgenticRAGPipeline("config.yaml")
            >>> result = pipeline.run("What are the key findings?", top_k=10)
            >>> print(result["final_answer"])
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "pinecone")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        # Initialize Pinecone connection for cloud vector storage
        pinecone_config = self.config["pinecone"]
        self.db = PineconeVectorDB(
            api_key=pinecone_config["api_key"],
            index_name=pinecone_config.get("index_name"),
        )

        self.index_name = pinecone_config.get("index_name")
        self.namespace = pinecone_config.get("namespace", "")

        # LLM is required for both generation and agentic routing decisions
        self.llm = RAGHelper.create_llm(self.config)
        if self.llm is None:
            msg = "RAG LLM must be enabled for agentic RAG"
            raise ValueError(msg)

        # Agentic configuration controls iteration limits and compression strategy
        agentic_config = self.config.get("agentic", {})
        self.max_iterations = agentic_config.get("max_iterations", 3)
        self.compression_mode = agentic_config.get("compression_mode", "reranking")

        # Router uses a dedicated model for action decisions
        router_model = agentic_config.get("router_model", "llama-3.3-70b-versatile")
        self.router = AgenticRouter(
            ChatGroq(
                model=router_model,
                api_key=self.llm.api_key,
                temperature=0.5,
            )
        )

        # Compression reduces context window usage while preserving relevance
        if self.compression_mode == "reranking":
            reranker = RerankerHelper.create_reranker(self.config)
            self.compressor = ContextCompressor(mode="reranking", reranker=reranker)
        else:
            self.compressor = ContextCompressor(mode="llm_extraction", llm=self.llm)

        logger.info(
            "Initialized Pinecone agentic RAG pipeline (LangChain) "
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

        Runs the iterative agent loop with Pinecone as the retrieval backend.
        The namespace parameter enables querying specific document subsets
        within the index.

        Args:
            query: User query text to process.
            top_k: Number of documents to retrieve per search (default 10).
            filters: Optional metadata filters for Pinecone query.

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

        # Compute query embedding once for reuse across searches
        query_embedding = EmbedderHelper.embed_query(self.embedder, query)

        # Agentic reasoning loop
        while iteration <= self.max_iterations:
            logger.info("Iteration %d/%d", iteration, self.max_iterations)

            # Router decides next action based on current state
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

            # Execute the router's chosen action
            if action == "search":
                logger.info("Executing SEARCH action")

                # Retrieve documents from Pinecone namespace
                documents = self.db.query(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    filters=filters,
                    namespace=self.namespace,
                )
                logger.info("Retrieved %d documents from Pinecone", len(documents))

                # Compress documents to most relevant subset
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

                # Reflection evaluates answer quality and identifies gaps
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

                # Generate final answer using retrieved documents
                if retrieved_documents:
                    current_answer = RAGHelper.generate(
                        self.llm, query, retrieved_documents
                    )
                else:
                    # Fallback to LLM knowledge if no documents retrieved
                    current_answer = self.llm.invoke(query).content

                logger.info("Generated answer: %s", current_answer[:200])
                step_record["answer_generated"] = True

                # Exit loop after generation
                intermediate_steps.append(step_record)
                break

            intermediate_steps.append(step_record)
            iteration += 1

        # Fallback if no generation occurred within iteration limit
        if current_answer is None:
            logger.warning("No answer generated, using LLM fallback")
            current_answer = self.llm.invoke(query).content

        return {
            "final_answer": current_answer,
            "documents": retrieved_documents,
            "intermediate_steps": intermediate_steps,
            "reasoning": [step.get("reasoning") for step in intermediate_steps],
        }
