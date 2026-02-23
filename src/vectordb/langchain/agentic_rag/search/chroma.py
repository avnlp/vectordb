"""Chroma agentic RAG search pipeline (LangChain).

This module implements the search phase of agentic RAG using Chroma as the
vector store backend. The agentic pattern transforms traditional RAG from a
single retrieval-generation cycle into an iterative process where an LLM-based
agent makes dynamic decisions about search, reflection, and generation.

Agentic RAG Pattern:
    Unlike standard RAG which retrieves once and generates, agentic RAG uses
    an iterative loop where the agent can:
    1. SEARCH: Retrieve documents from Chroma based on the query
    2. REFLECT: Evaluate answer quality and identify information gaps
    3. GENERATE: Produce final answer when sufficient information exists

    This pattern is particularly valuable for complex queries requiring
    multi-hop reasoning or when initial retrieval returns insufficient context.

Chroma Integration:
    Chroma's local embedded mode provides fast, low-latency retrieval suitable
    for iterative agent workflows. The collection-based organization allows
    the agent to potentially query multiple document collections during its
    reasoning process.

Design Decisions:
    - Query embedding is computed once and reused across iterations to minimize
      embedding API calls and reduce latency
    - Document compression happens after each search to maintain context window
      limits while preserving the most relevant information
    - Reflection prompts include only top-3 documents to manage token usage
    - The agent can exit early on GENERATE or continue until max_iterations
"""

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_groq import ChatGroq

from vectordb.databases.chroma import ChromaVectorDB
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


class ChromaAgenticRAGPipeline(AgenticRAGPipeline):
    """Chroma agentic RAG pipeline (LangChain).

    Combines agentic routing with vector search, document compression,
    answer reflection, and RAG generation using Chroma backend. The agent
    makes routing decisions at each step to determine whether to search,
    reflect, or generate.

    The agentic approach addresses limitations of standard RAG by allowing
    iterative refinement. If initial retrieval is insufficient, the agent
    can reflect on what's missing and potentially reformulate the search
    (though current implementation uses the same query embedding).

    Attributes:
        config: Loaded and validated configuration dictionary.
        embedder: Embedding model for query vectorization.
        db: ChromaVectorDB instance for document retrieval.
        collection_name: Target Chroma collection for search.
        llm: Language model for RAG generation and reflection.
        max_iterations: Maximum agent reasoning iterations.
        compression_mode: Document compression strategy
            ("reranking" or "llm_extraction").
        router: AgenticRouter for action decisions.
        compressor: ContextCompressor for document pruning.
    """

    def __init__(self, config_or_path: dict[str, Any] | str) -> None:
        """Initialize agentic RAG pipeline from configuration.

        Sets up the Chroma connection, embedding model, LLM, router, and
        document compressor. Validates that RAG LLM is enabled since it's
        required for both generation and agentic routing.

        Args:
            config_or_path: Configuration dictionary or path to YAML file.
                Must contain chroma, embedding, rag, and agentic sections.

        Raises:
            ValueError: If RAG LLM is not enabled (required for agentic routing).

        Example:
            >>> pipeline = ChromaAgenticRAGPipeline("config.yaml")
            >>> result = pipeline.run("What are the key findings?", top_k=10)
            >>> print(result["final_answer"])
        """
        self.config = ConfigLoader.load(config_or_path)
        ConfigLoader.validate(self.config, "chroma")

        self.embedder = EmbedderHelper.create_embedder(self.config)

        # Initialize Chroma connection for local vector storage
        chroma_config = self.config["chroma"]
        self.db = ChromaVectorDB(
            path=chroma_config.get("path"),
            collection_name=chroma_config.get("collection_name"),
        )

        self.collection_name = chroma_config.get("collection_name")

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
        # Using a capable model (70B) ensures high-quality routing decisions
        router_model = agentic_config.get("router_model", "llama-3.3-70b-versatile")
        self.router = AgenticRouter(
            ChatGroq(
                model=router_model,
                api_key=self.llm.api_key,
                temperature=0.5,  # Moderate temperature for balanced creativity/consistency
            )
        )

        # Compression reduces context window usage while preserving relevance
        if self.compression_mode == "reranking":
            reranker = RerankerHelper.create_reranker(self.config)
            self.compressor = ContextCompressor(mode="reranking", reranker=reranker)
        else:
            self.compressor = ContextCompressor(mode="llm_extraction", llm=self.llm)

        logger.info(
            "Initialized Chroma agentic RAG pipeline (LangChain) "
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

        Runs the iterative agent loop: at each iteration, the router decides
        whether to search for documents, reflect on the current answer, or
        generate a final response. The loop continues until generation or
        max_iterations is reached.

        The agentic loop provides several advantages over standard RAG:
        - Can detect when retrieved documents are insufficient
        - Can reflect on answer quality before returning
        - Provides transparency through intermediate_steps logging

        Args:
            query: User query text to process.
            top_k: Number of documents to retrieve per search (default 10).
            filters: Optional metadata filters for Chroma query.

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
        # This optimization reduces embedding API calls in multi-turn scenarios
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

                # Retrieve documents from Chroma using pre-computed embedding
                documents = self.db.query(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    filters=filters,
                    collection_name=self.collection_name,
                )
                logger.info("Retrieved %d documents from Chroma", len(documents))

                # Compress documents to most relevant subset
                # Compression maintains quality while reducing token usage
                if documents:
                    compressed_docs = self.compressor.compress(
                        query=query,
                        documents=documents,
                        top_k=5,  # Target 5 most relevant documents
                    )
                    retrieved_documents = compressed_docs
                    logger.info("Compressed to %d documents", len(compressed_docs))
                else:
                    retrieved_documents = documents

                step_record["documents_retrieved"] = len(retrieved_documents)

            elif action == "reflect":
                logger.info("Executing REFLECT action")

                # Reflection evaluates answer quality and identifies gaps
                # Only reflect if we have both documents and a current answer
                if retrieved_documents and current_answer:
                    reflection_prompt = (
                        f"Given the following documents and answer, is the answer "
                        f"accurate, complete, and well-supported by the documents? "
                        f"Suggest improvements if needed.\n\n"
                        f"Documents:\n"
                        f"{chr(10).join([doc.page_content for doc in retrieved_documents[:3]])}\n\n"
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
