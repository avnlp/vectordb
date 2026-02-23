"""Base class for agentic RAG pipelines.

This module defines the abstract base class for all agentic RAG (Retrieval-Augmented
Generation) pipelines. Agentic RAG extends traditional RAG by introducing an
agent-based control flow that can perform multiple iterations of retrieval and
reflection before generating a final answer.

Agentic RAG Pattern:
    Traditional RAG follows a linear pipeline: retrieve -> generate. Agentic RAG
    introduces a feedback loop where the system can:

    1. Search: Retrieve documents from the vector database
    2. Reflect: Evaluate the current answer and identify gaps
    3. Generate: Produce a final answer when sufficient information exists

    The agent uses an LLM-based router to decide which action to take at each
    step, enabling dynamic adaptation to query complexity.

Benefits of Agentic RAG:
    - Multi-hop reasoning: Can retrieve documents, reflect, and retrieve again
      to answer complex questions requiring multiple pieces of information
    - Self-correction: Can identify when retrieved documents are insufficient
      or irrelevant and search for better sources
    - Quality assurance: Reflection step helps catch errors before final output
    - Iterative refinement: Answer quality improves through multiple iterations

Pipeline Architecture:
    All agentic RAG pipelines inherit from AgenticRAGPipeline and implement:
    - Vector store connection and querying
    - Document compression (reranking or LLM extraction)
    - Agentic routing for search/reflect/generate decisions
    - Answer generation with retrieved context

Usage:
    Concrete implementations are provided for each vector database:
    - PineconeAgenticRAGPipeline
    - WeaviateAgenticRAGPipeline
    - ChromaAgenticRAGPipeline
    - MilvusAgenticRAGPipeline
    - QdrantAgenticRAGPipeline
"""

import logging
from abc import ABC, abstractmethod
from typing import Any


# Module-level logger for agentic RAG operations
logger = logging.getLogger(__name__)


class AgenticRAGPipeline(ABC):
    """Abstract base class for agentic RAG pipelines.

    Agentic RAG combines multi-turn reasoning with vector search, document
    compression, answer reflection, and generation. The agent routes between
    search, reflect, and generate actions based on query and current state.

    The agentic loop works as follows:
    1. Router decides whether to search, reflect, or generate
    2. If search: Retrieve documents and compress them
    3. If reflect: Evaluate current answer and identify improvements
    4. If generate: Produce final answer using retrieved context
    5. Repeat until max iterations reached or generate action chosen

    Attributes:
        None (abstract base class - implementations define attributes)

    Design Pattern:
        Template Method pattern: Subclasses implement database-specific
        retrieval logic while inheriting the agentic control flow structure.
    """

    @abstractmethod
    def run(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute agentic RAG pipeline.

        This method implements the core agentic loop:
        1. Initialize with the user query
        2. While iterations < max_iterations:
           a. Route to search, reflect, or generate
           b. Execute the chosen action
           c. Update state (documents, answer, reasoning)
        3. Return final answer with metadata

        Args:
            query: User query text. This is the primary input that drives
                the entire agentic process.
            top_k: Number of documents to retrieve in each search iteration.
                Default is 10. Higher values provide more context but increase
                token usage.
            filters: Optional metadata filters for searches. Format depends
                on the underlying vector database.

        Returns:
            Dictionary containing:
                - final_answer: The generated answer string.
                - documents: List of Document objects used for the answer.
                - intermediate_steps: List of actions taken during agentic loop
                  (e.g., ['search', 'reflect', 'generate']).
                - reasoning: Router's reasoning for each step, providing
                  transparency into the agent's decision-making.

        Example:
            >>> pipeline = PineconeAgenticRAGPipeline("config.yaml")
            >>> result = pipeline.run("What is quantum computing?")
            >>> print(result["final_answer"])
            >>> print(f"Steps taken: {result['intermediate_steps']}")
        """
        pass
