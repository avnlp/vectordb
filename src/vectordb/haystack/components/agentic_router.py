"""Agentic routing for advanced RAG pipelines.

This module provides an LLM-based router that orchestrates multi-step RAG workflows.
The router can select appropriate tools, evaluate answer quality, and iterate on
responses through self-reflection loops.

Capabilities:
    - Tool Selection: Choose between retrieval, web search, calculation, reasoning
    - Answer Evaluation: Assess relevance, completeness, and grounding
    - Self-Reflection: Iteratively improve answers based on evaluation
    - Fallback Handling: Graceful degradation when tools fail

Architecture:
    The router uses any OpenAI-compatible API (defaults to Groq) for LLM
    inference. It maintains state across iterations and can trigger refinement
    loops until quality thresholds are met.

Design Notes:
    - Temperature is set to 0 for deterministic routing decisions
    - Max tokens limited to prevent verbose outputs
    - All tool selections are logged for debugging

Usage:
    >>> from vectordb.haystack.components import AgenticRouter
    >>> router = AgenticRouter(model="llama-3.3-70b-versatile")
    >>> tool = router.select_tool("What is the capital of France?")
    >>> # tool = "retrieval"
    >>> router.self_reflect_loop(query, answer, context, max_iterations=3)

Note:
    This component is part of the Haystack integration layer and uses
    Haystack's OpenAIChatGenerator for LLM interactions, configured to
    use any OpenAI-compatible API (Groq by default).
"""

import json
import logging
import os
from typing import Any

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret


logger = logging.getLogger(__name__)


class AgenticRouter:
    """Route and orchestrate RAG with agent-like behavior.

    Supports:
    - Tool selection: Retrieval, web search, calculation
    - Self-reflection: Evaluate answer quality and iterate
    - Fallback mechanisms: Graceful degradation

    The router uses an LLM (via any OpenAI-compatible API, defaulting to Groq)
    to make routing decisions and evaluate answer quality. It maintains no
    persistent state between calls, making it suitable for stateless pipeline
    integration.

    Attributes:
        generator: Haystack OpenAIChatGenerator configured for the specified
            OpenAI-compatible API.
        available_tools: List of tool names the router can select from.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
        api_base_url: str = "https://api.groq.com/openai/v1",
    ) -> None:
        """Initialize agentic router.

        Args:
            model: LLM model name.
            api_key: API key (or set GROQ_API_KEY env var).
            api_base_url: Base URL for the OpenAI-compatible API.
        """
        resolved_api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_api_key:
            msg = "GROQ_API_KEY required. Set it as environment variable."
            raise ValueError(msg)

        try:
            # Use OpenAIChatGenerator with an OpenAI-compatible API
            # Temperature=0 ensures deterministic routing decisions
            self.generator = OpenAIChatGenerator(
                api_key=Secret.from_token(resolved_api_key),
                model=model,
                api_base_url=api_base_url,
                generation_kwargs={"temperature": 0, "max_tokens": 1024},
            )
            logger.info("Initialized AgenticRouter with model: %s", model)
        except Exception as e:
            logger.error("Failed to initialize AgenticRouter: %s", str(e))
            raise

        # Define available tools for routing decisions
        # These map to different RAG strategies in the pipeline
        self.available_tools = [
            "retrieval",
            "web_search",
            "calculation",
            "reasoning",
        ]

    def select_tool(self, query: str) -> str:
        """Select the best tool for a query.

        Args:
            query: User query.

        Returns:
            Selected tool: "retrieval", "web_search", "calculation", or "reasoning".
        """
        tools_str = ", ".join(self.available_tools)
        prompt = f"""Given this query: "{query}"

Select the BEST tool to answer it. Options: {tools_str}

- retrieval: For factual information from a knowledge base
- web_search: For current events, real-time information
- calculation: For mathematical or computational problems
- reasoning: For multi-step logic or analysis

Return ONLY the tool name."""  # nosec: B608

        try:
            # Use Haystack's ChatMessage API for LLM interaction
            messages = [ChatMessage.from_user(prompt)]
            response = self.generator.run(messages=messages)

            # Extract tool from LLM response, with fallback to retrieval
            tool = (
                response["replies"][0].text.strip().lower()
                if response.get("replies")
                else "retrieval"
            )

            # Validate against known tools; fallback prevents pipeline failures
            if tool not in self.available_tools:
                tool = "retrieval"

            logger.info("Tool selection: '%s' → %s", query[:50], tool)
            return tool
        except Exception as e:
            # Always return a valid tool on failure to maintain pipeline flow
            logger.error("Tool selection failed: %s", str(e))
            return "retrieval"

    def evaluate_answer_quality(
        self,
        query: str,
        answer: str,
        context: str = "",
    ) -> dict[str, Any]:
        """Evaluate generated answer quality.

        Args:
            query: Original query.
            answer: Generated answer.
            context: Retrieved context (for grounding check).

        Returns:
            Evaluation dict with score, issues, and suggestions.
        """
        prompt = f"""Evaluate this answer to the query.

Query: "{query}"
Answer: "{answer}"
Context: "{context}"

Assess:
1. Relevance (0-100): Does it answer the query?
2. Completeness (0-100): Is it sufficiently detailed?
3. Grounding (0-100): Is it grounded in the context?
4. Issues: List any problems (max 3)
5. Suggestions: List improvements (max 2)

Format as JSON:
{{"relevance": X, "completeness": X, "grounding": X, "issues": [...], "suggestions": [...]}}

Return ONLY the JSON."""

        try:
            messages = [ChatMessage.from_user(prompt)]
            response = self.generator.run(messages=messages)
            content = response["replies"][0].text if response.get("replies") else "{}"

            # Parse JSON response from LLM
            eval_dict = json.loads(content.strip())

            # Log quality metrics for debugging and monitoring
            logger.info(
                "Answer quality: relevance=%d, completeness=%d, grounding=%d",
                eval_dict.get("relevance", 0),
                eval_dict.get("completeness", 0),
                eval_dict.get("grounding", 0),
            )
            return eval_dict
        except Exception as e:
            # Return zero scores on failure to trigger refinement
            logger.error("Answer quality evaluation failed: %s", str(e))
            return {
                "relevance": 0,
                "completeness": 0,
                "grounding": 0,
                "issues": ["Evaluation failed"],
                "suggestions": [],
            }

    def should_refine_answer(
        self,
        eval_result: dict[str, Any],
        threshold: int = 70,
    ) -> bool:
        """Decide if answer needs refinement based on evaluation.

        Args:
            eval_result: Evaluation result from evaluate_answer_quality.
            threshold: Minimum acceptable quality score.

        Returns:
            True if answer should be refined, False otherwise.
        """
        avg_score = (
            eval_result.get("relevance", 0)
            + eval_result.get("completeness", 0)
            + eval_result.get("grounding", 0)
        ) / 3

        should_refine = avg_score < threshold
        logger.info(
            "Refinement needed: %s (avg_score=%.0f, threshold=%d)",
            should_refine,
            avg_score,
            threshold,
        )
        return should_refine

    def refine_answer(
        self,
        query: str,
        answer: str,
        eval_result: dict[str, Any],
    ) -> str:
        """Refine answer based on evaluation feedback.

        Args:
            query: Original query.
            answer: Original answer.
            eval_result: Evaluation result with issues and suggestions.

        Returns:
            Refined answer.
        """
        issues = eval_result.get("issues", [])
        suggestions = eval_result.get("suggestions", [])

        prompt = f"""Improve this answer based on the feedback.

Query: "{query}"
Original Answer: "{answer}"

Issues to fix:
{json.dumps(issues, indent=2)}

Suggestions:
{json.dumps(suggestions, indent=2)}

Provide a REVISED answer that addresses all issues and incorporates suggestions."""

        try:
            messages = [ChatMessage.from_user(prompt)]
            response = self.generator.run(messages=messages)
            refined = response["replies"][0].text if response.get("replies") else answer
            logger.info("Answer refined")
            return refined
        except Exception as e:
            logger.error("Answer refinement failed: %s", str(e))
            return answer

    def self_reflect_loop(
        self,
        query: str,
        answer: str,
        context: str = "",
        max_iterations: int = 2,
        quality_threshold: int = 75,
    ) -> str:
        """Run self-reflection loop to iteratively improve answer.

        Args:
            query: Original query.
            answer: Initial answer.
            context: Retrieved context.
            max_iterations: Maximum refinement iterations.
            quality_threshold: Target quality score.

        Returns:
            Final refined answer.
        """
        current_answer = answer

        # Iterative refinement loop with early exit on quality threshold
        for iteration in range(max_iterations):
            eval_result = self.evaluate_answer_quality(query, current_answer, context)

            # Exit early if quality is acceptable
            if not self.should_refine_answer(eval_result, quality_threshold):
                logger.info("Quality threshold reached at iteration %d", iteration)
                break

            # Refine answer based on evaluation feedback
            current_answer = self.refine_answer(query, current_answer, eval_result)
            logger.info("Refinement iteration %d completed", iteration + 1)

        return current_answer
