"""Agentic router component for LangChain pipelines.

This module provides an LLM-based router that makes agentic decisions about
what action to take next in a RAG pipeline. The router implements a decision-making
pattern inspired by ReAct (Reasoning + Acting) frameworks, where the LLM evaluates
the current state and chooses the optimal next step.

Agentic RAG Pattern:
    Traditional RAG pipelines follow a linear flow: query -> retrieve -> generate.
    Agentic RAG introduces a feedback loop where the system can:
    1. Search for documents when information is insufficient
    2. Reflect on the current answer to identify gaps or errors
    3. Generate a final answer only when confidence is high

    This pattern is particularly effective for:
    - Complex multi-hop questions requiring multiple retrieval steps
    - Cases where initial retrieval returns irrelevant documents
    - Situations requiring self-correction or fact-checking

Decision States:
    - 'search': Retrieve documents from vector database. Chosen when:
        * No documents have been retrieved yet
        * Current answer lacks sufficient supporting evidence
        * Reflection identified information gaps

    - 'reflect': Evaluate and improve the current answer. Chosen when:
        * Documents exist but answer quality is uncertain
        * Potential contradictions or inconsistencies detected
        * Need to verify factual accuracy before finalizing

    - 'generate': Create final answer. Chosen when:
        * Sufficient information has been gathered
        * Answer has passed reflection checks
        * Maximum iterations reached (fallback)

Design Decisions:
    - JSON Output: The router requires structured JSON responses from the LLM
      rather than free-form text. This enables programmatic handling of decisions
      and makes the routing logic transparent and debuggable.

    - Iteration Limiting: A max_iterations parameter prevents infinite loops
      in cases where the router oscillates between search and reflect states.
      When the limit is reached, the system falls back to 'generate'.

    - State Persistence: The router is stateless by design - all state is passed
      through parameters. This allows the same router instance to handle multiple
      concurrent conversations without interference.

Integration with LangChain:
    The router integrates with LangChain's ChatGroq interface, allowing it to
    work with any LangChain-compatible LLM that supports structured outputs.
    The prompt template follows LangChain conventions for variable substitution.

Usage:
    >>> from langchain_groq import ChatGroq
    >>> from vectordb.langchain.components import AgenticRouter
    >>> llm = ChatGroq(model="llama-3.3-70b-versatile")
    >>> router = AgenticRouter(llm)
    >>> # Initial routing - should suggest 'search'
    >>> decision = router.route("What is quantum computing?", has_documents=False)
    >>> # decision = {"action": "search", "reasoning": "No documents retrieved yet"}
    >>> # After retrieval - may suggest 'reflect' or 'generate'
    >>> decision = router.route(
    ...     "What is quantum computing?",
    ...     has_documents=True,
    ...     current_answer="Quantum computing uses qubits...",
    ... )
"""

import json
import logging
from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq


# Module-level logger for routing decisions and debugging
logger = logging.getLogger(__name__)


class AgenticRouter:
    """Route queries to search, reflect, or generate actions using LLM reasoning.

    The router implements an agentic decision-making pattern where an LLM evaluates
    the current pipeline state and determines the optimal next action. This enables
    dynamic RAG pipelines that can adapt their retrieval strategy based on
    intermediate results.

    Attributes:
        ROUTING_TEMPLATE: Prompt template that structures the decision context for
            the LLM. Includes current state (query, documents, answer) and available
            actions with their use cases.
        llm: LangChain chat model instance used for routing decisions.

    Design Pattern:
        The router follows the Strategy pattern, encapsulating the routing algorithm
        and making it interchangeable. The decision logic is externalized to the LLM
        rather than hardcoded, allowing for nuanced decisions that consider the
        full context.
    """

    ROUTING_TEMPLATE = """You are a query routing agent. Given a query and optional current answer, decide what action to take next.

Current State:
- Query: {query}
- Has Retrieved Documents: {has_documents}
- Current Answer: {current_answer}
- Iteration: {iteration}/{max_iterations}

Your task is to decide ONE of the following actions:
1. 'search': Retrieve documents from vector database (choose this if you need more information)
2. 'reflect': Verify and improve the current answer (choose this to validate answer quality)
3. 'generate': Create final answer (choose this when you have enough information)

Return a JSON object with this exact format:
{{"action": "search|reflect|generate", "reasoning": "brief explanation"}}

Do NOT include any other text. Return ONLY the JSON object."""

    def __init__(self, llm: ChatGroq) -> None:
        """Initialize AgenticRouter with a LangChain LLM instance.

        Args:
            llm: ChatGroq instance for routing decisions. Must support structured
                text generation with JSON output formatting.

        Note:
            The LLM should be configured with appropriate temperature
            (recommend 0.0-0.3)
            for consistent routing decisions. Higher temperatures may cause inconsistent
            action selection.
        """
        self.llm = llm

    def route(
        self,
        query: str,
        has_documents: bool = False,
        current_answer: str | None = None,
        iteration: int = 1,
        max_iterations: int = 3,
    ) -> dict[str, Any]:
        """Route a query to the appropriate action based on current pipeline state.

        This method implements the core routing logic, invoking the LLM to make
        a decision about the next action. It handles edge cases like maximum
        iteration limits and invalid LLM responses.

        Args:
            query: The user's original query text. This is the primary input that
                drives the routing decision.
            has_documents: Indicates whether documents have already been retrieved
                in previous iterations. Affects whether 'search' is a valid option.
            current_answer: The answer generated so far, if any. Used by the LLM
                to assess whether reflection or generation is appropriate.
            iteration: Current iteration number (1-indexed). Used to track progress
                and enforce iteration limits.
            max_iterations: Maximum number of routing iterations allowed. Prevents
                infinite loops in edge cases. When reached, forces 'generate' action.

        Returns:
            A dictionary containing:
                - 'action': One of 'search', 'reflect', or 'generate'
                - 'reasoning': Human-readable explanation of the decision

        Raises:
            ValueError: If the LLM response cannot be parsed as valid JSON, or if
                required fields ('action', 'reasoning') are missing, or if the
                action is not one of the allowed values.

        Example:
            >>> router = AgenticRouter(llm)
            >>> result = router.route(
            ...     "Explain quantum entanglement", has_documents=False
            ... )
            >>> print(result)
            {'action': 'search',
             'reasoning': 'Need to retrieve relevant documents first'}
        """
        # Enforce iteration limit as a safety mechanism. This prevents infinite
        # loops when the router alternates between search and reflect without
        # making progress toward a final answer.
        if iteration >= max_iterations:
            logger.info(
                "Max iterations reached, forcing generate action at iteration %d",
                iteration,
            )
            return {
                "action": "generate",
                "reasoning": f"Reached maximum iterations ({max_iterations})",
            }

        # Prepare the answer string for the prompt template. Using a placeholder
        # when no answer exists helps the LLM understand the current state.
        answer_str = current_answer if current_answer else "No answer yet"

        # Construct the prompt using LangChain's PromptTemplate for consistent
        # variable substitution and escaping.
        prompt = PromptTemplate(
            template=self.ROUTING_TEMPLATE,
            input_variables=[
                "query",
                "has_documents",
                "current_answer",
                "iteration",
                "max_iterations",
            ],
        )
        formatted_prompt = prompt.format(
            query=query,
            has_documents=has_documents,
            current_answer=answer_str,
            iteration=iteration,
            max_iterations=max_iterations,
        )

        # Log the full prompt at DEBUG level for troubleshooting routing issues.
        # This helps identify when prompt formatting causes unexpected behavior.
        logger.debug("Routing prompt: %s", formatted_prompt)

        # Invoke the LLM to get the routing decision. The LLM processes the
        # structured context and returns a JSON-formatted decision.
        response = self.llm.invoke(formatted_prompt)
        response_text = response.content.strip()

        # Log the raw response for debugging JSON parsing issues.
        logger.debug("Router response: %s", response_text)

        # Parse the JSON response. The LLM is instructed to return only JSON,
        # but we handle parsing errors gracefully with informative messages.
        try:
            decision = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse router response as JSON: %s", response_text)
            raise ValueError(f"Invalid JSON from router: {response_text}") from e

        # Validate that the response contains all required fields.
        # This ensures downstream code can rely on the decision structure.
        if "action" not in decision or "reasoning" not in decision:
            msg = f"Router response missing required fields: {decision}"
            raise ValueError(msg)

        # Normalize and validate the action value.
        # Lowercase and strip to handle variations in LLM output formatting.
        action = decision["action"].lower().strip()
        if action not in ("search", "reflect", "generate"):
            msg = (
                f"Invalid action: {action}. Must be 'search', 'reflect', or 'generate'"
            )
            raise ValueError(msg)

        # Log the final decision at INFO level for monitoring routing behavior.
        # This helps identify patterns in how the router handles different queries.
        logger.info("Router decided: %s (reasoning: %s)", action, decision["reasoning"])

        return {
            "action": action,
            "reasoning": decision["reasoning"],
        }
