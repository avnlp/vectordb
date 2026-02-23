"""Tests for AgenticRouter component (LangChain).

This module tests the AgenticRouter class which implements decision-making logic
for agentic RAG pipelines. The router determines the next action in an iterative
retrieval loop: search for more documents, reflect on current answer, or generate
final answer.

Agentic RAG Loop:
    1. Router decides action based on query, documents, and current answer
    2. Pipeline executes action (search/reflect/generate)
    3. State updates with new documents or answer
    4. Loop continues until generate action or max iterations

Router Actions:
    - search: Retrieve more documents from vector database
    - reflect: Evaluate and verify current answer quality
    - generate: Produce final answer and exit loop

Decision Factors:
    - has_documents: Whether documents have been retrieved
    - current_answer: Partial or complete answer if available
    - iteration: Current loop iteration count
    - max_iterations: Maximum allowed iterations

Test Coverage:
    - Initialization with LLM
    - Route method with all three actions
    - Max iteration enforcement
    - Prompt formatting with context
    - JSON response parsing
    - Error handling for invalid responses
    - Edge cases: empty queries, long queries
    - Integration patterns for agentic workflows

All tests mock the LLM to avoid external API calls.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage


class TestAgenticRouter:
    """Unit tests for AgenticRouter class.

    Base test class providing fixtures for AgenticRouter tests.
    Sets up mock LLM and router instance for use across test subclasses.

    Fixtures:
        mock_llm: MagicMock simulating LangChain LLM interface
        router: AgenticRouter instance configured with mock LLM
    """

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create mock LLM for testing."""
        return MagicMock()

    @pytest.fixture
    def router(self, mock_llm):
        """Create AgenticRouter instance with mock LLM."""
        from vectordb.langchain.components.agentic_router import AgenticRouter

        return AgenticRouter(llm=mock_llm)


class TestAgenticRouterInitialization(TestAgenticRouter):
    """Tests for AgenticRouter initialization and configuration.

    Validates that AgenticRouter properly stores LLM reference and
    that class-level routing templates are correctly defined.

    Template Requirements:
        - Contains placeholders for query, has_documents, current_answer
        - Defines all three actions: search, reflect, generate
        - Provides clear instructions to LLM for decision-making
    """

    def test_initialization_with_valid_llm(self, mock_llm):
        """Test initialization with a valid LLM instance."""
        from vectordb.langchain.components.agentic_router import AgenticRouter

        router = AgenticRouter(llm=mock_llm)
        assert router.llm is mock_llm

    def test_initialization_with_none_llm(self):
        """Test initialization with None LLM (edge case)."""
        from vectordb.langchain.components.agentic_router import AgenticRouter

        router = AgenticRouter(llm=None)
        assert router.llm is None

    def test_routing_template_is_class_attribute(self):
        """Test that routing template is defined as class attribute."""
        from vectordb.langchain.components.agentic_router import AgenticRouter

        assert hasattr(AgenticRouter, "ROUTING_TEMPLATE")
        assert "query routing agent" in AgenticRouter.ROUTING_TEMPLATE
        assert "{query}" in AgenticRouter.ROUTING_TEMPLATE
        assert "{has_documents}" in AgenticRouter.ROUTING_TEMPLATE
        assert "{current_answer}" in AgenticRouter.ROUTING_TEMPLATE

    def test_routing_template_contains_all_actions(self):
        """Test that template defines all expected actions."""
        from vectordb.langchain.components.agentic_router import AgenticRouter

        template = AgenticRouter.ROUTING_TEMPLATE
        assert "'search'" in template
        assert "'reflect'" in template
        assert "'generate'" in template


class TestAgenticRouterRoute(TestAgenticRouter):
    """Tests for route method.

    Validates the core routing logic that determines the next action
    in the agentic RAG loop. The router uses an LLM to decide whether
    to search, reflect, or generate based on current state.

    Routing Logic:
        - search: When more information needed
        - reflect: When verifying answer quality
        - generate: When ready to produce final answer

    Max Iterations:
        - Forces generate action when iteration >= max_iterations
        - Prevents infinite loops
        - Returns reasoning about iteration limit

    Response Format:
        - JSON with "action" and "reasoning" fields
        - Action must be one of: search, reflect, generate
        - Reasoning explains the decision
    """

    def test_route_returns_search_action(self, router, mock_llm):
        """Test route returns search action when documents not retrieved."""
        mock_response = AIMessage(
            content='{"action": "search", "reasoning": "Need more information from database"}'
        )
        mock_llm.invoke.return_value = mock_response

        result = router.route(query="What is Python?", has_documents=False)

        assert result["action"] == "search"
        assert "reasoning" in result
        mock_llm.invoke.assert_called_once()

    def test_route_returns_reflect_action(self, router, mock_llm):
        """Test route returns reflect action for answer verification."""
        mock_response = AIMessage(
            content='{"action": "reflect", "reasoning": "Answer needs verification"}'
        )
        mock_llm.invoke.return_value = mock_response

        result = router.route(
            query="What is Python?",
            has_documents=True,
            current_answer="Python is a programming language",
        )

        assert result["action"] == "reflect"
        assert "reasoning" in result

    def test_route_returns_generate_action(self, router, mock_llm):
        """Test route returns generate action when ready to answer."""
        mock_response = AIMessage(
            content='{"action": "generate", "reasoning": "Sufficient information available"}'
        )
        mock_llm.invoke.return_value = mock_response

        result = router.route(
            query="What is Python?",
            has_documents=True,
            current_answer="Python is a programming language created by Guido van Rossum.",
        )

        assert result["action"] == "generate"
        assert "reasoning" in result

    def test_route_with_all_parameters(self, router, mock_llm):
        """Test route with all parameters provided."""
        mock_response = AIMessage(
            content='{"action": "search", "reasoning": "Need more context"}'
        )
        mock_llm.invoke.return_value = mock_response

        result = router.route(
            query="Explain quantum computing",
            has_documents=True,
            current_answer="Quantum computing uses qubits.",
            iteration=2,
            max_iterations=5,
        )

        assert result["action"] == "search"
        mock_llm.invoke.assert_called_once()

    def test_route_max_iterations_force_generate(self, router, mock_llm):
        """Test route forces generate when max iterations reached."""
        result = router.route(
            query="What is Python?",
            has_documents=False,
            iteration=5,
            max_iterations=5,
        )

        assert result["action"] == "generate"
        assert "maximum iterations" in result["reasoning"].lower()
        # LLM should not be called when max iterations reached
        mock_llm.invoke.assert_not_called()

    def test_route_max_iterations_exceeded(self, router, mock_llm):
        """Test route forces generate when iteration exceeds max."""
        result = router.route(
            query="Test query",
            has_documents=True,
            current_answer="Test answer",
            iteration=10,
            max_iterations=5,
        )

        assert result["action"] == "generate"
        assert result["reasoning"] == "Reached maximum iterations (5)"
        mock_llm.invoke.assert_not_called()

    def test_route_prompt_includes_query(self, router, mock_llm):
        """Test that LLM is invoked with formatted prompt containing query."""
        mock_response = AIMessage(content='{"action": "search", "reasoning": "test"}')
        mock_llm.invoke.return_value = mock_response

        router.route(query="specific test query", has_documents=False)

        call_args = mock_llm.invoke.call_args
        assert call_args is not None
        prompt_arg = call_args[0][0]
        assert "specific test query" in prompt_arg

    def test_route_prompt_includes_has_documents(self, router, mock_llm):
        """Test that prompt includes has_documents flag."""
        mock_response = AIMessage(content='{"action": "search", "reasoning": "test"}')
        mock_llm.invoke.return_value = mock_response

        router.route(query="test", has_documents=True)

        call_args = mock_llm.invoke.call_args
        prompt_arg = call_args[0][0]
        assert "True" in prompt_arg or "has_documents" in prompt_arg

    def test_route_prompt_includes_iteration_info(self, router, mock_llm):
        """Test that prompt includes iteration and max_iterations."""
        mock_response = AIMessage(content='{"action": "search", "reasoning": "test"}')
        mock_llm.invoke.return_value = mock_response

        router.route(query="test", has_documents=False, iteration=2, max_iterations=4)

        call_args = mock_llm.invoke.call_args
        prompt_arg = call_args[0][0]
        assert "2/4" in prompt_arg


class TestAgenticRouterErrorHandling(TestAgenticRouter):
    """Tests for error handling in AgenticRouter.

    Validates robust handling of invalid LLM responses and edge cases.
    The router should provide clear error messages for debugging.

    Error Cases:
        - Invalid JSON in LLM response
        - Missing required fields (action, reasoning)
        - Invalid action values
        - Empty JSON objects

    Validation:
        - Action must be: search, reflect, or generate
        - Reasoning must be non-empty string
        - JSON must be parseable

    Case Handling:
        - Actions are case-insensitive (normalized to lowercase)
        - Whitespace is stripped from action values
        - Extra fields in JSON are allowed
    """

    def test_route_invalid_json_raises_value_error(self, router, mock_llm):
        """Test route raises ValueError when LLM response is invalid JSON."""
        mock_response = AIMessage(content="This is not valid JSON")
        mock_llm.invoke.return_value = mock_response

        with pytest.raises(ValueError) as exc_info:
            router.route(query="test", has_documents=False)

        assert "Invalid JSON" in str(exc_info.value)

    def test_route_missing_action_field_raises_value_error(self, router, mock_llm):
        """Test route raises ValueError when response missing action field."""
        mock_response = AIMessage(content='{"reasoning": "test reasoning"}')
        mock_llm.invoke.return_value = mock_response

        with pytest.raises(ValueError) as exc_info:
            router.route(query="test", has_documents=False)

        assert "missing required fields" in str(exc_info.value)

    def test_route_missing_reasoning_field_raises_value_error(self, router, mock_llm):
        """Test route raises ValueError when response missing reasoning field."""
        mock_response = AIMessage(content='{"action": "search"}')
        mock_llm.invoke.return_value = mock_response

        with pytest.raises(ValueError) as exc_info:
            router.route(query="test", has_documents=False)

        assert "missing required fields" in str(exc_info.value)

    def test_route_invalid_action_raises_value_error(self, router, mock_llm):
        """Test route raises ValueError for invalid action value."""
        mock_response = AIMessage(
            content='{"action": "invalid_action", "reasoning": "test"}'
        )
        mock_llm.invoke.return_value = mock_response

        with pytest.raises(ValueError) as exc_info:
            router.route(query="test", has_documents=False)

        assert "Invalid action" in str(exc_info.value)
        assert "invalid_action" in str(exc_info.value)

    def test_route_action_case_insensitive(self, router, mock_llm):
        """Test route accepts actions regardless of case."""
        mock_response = AIMessage(
            content='{"action": "SEARCH", "reasoning": "need data"}'
        )
        mock_llm.invoke.return_value = mock_response

        result = router.route(query="test", has_documents=False)

        assert result["action"] == "search"

    def test_route_action_whitespace_stripped(self, router, mock_llm):
        """Test route strips whitespace from action."""
        mock_response = AIMessage(
            content='{"action": "  search  ", "reasoning": "need data"}'
        )
        mock_llm.invoke.return_value = mock_response

        result = router.route(query="test", has_documents=False)

        assert result["action"] == "search"

    def test_route_empty_json_object_raises_error(self, router, mock_llm):
        """Test route raises error for empty JSON object."""
        mock_response = AIMessage(content="{}")
        mock_llm.invoke.return_value = mock_response

        with pytest.raises(ValueError):
            router.route(query="test", has_documents=False)

    def test_route_json_with_extra_fields_valid(self, router, mock_llm):
        """Test route accepts JSON with extra fields."""
        mock_response = AIMessage(
            content='{"action": "search", "reasoning": "test", "extra_field": "value"}'
        )
        mock_llm.invoke.return_value = mock_response

        result = router.route(query="test", has_documents=False)

        assert result["action"] == "search"
        assert result["reasoning"] == "test"


class TestAgenticRouterEdgeCases(TestAgenticRouter):
    """Tests for edge cases in AgenticRouter.

    Validates handling of boundary conditions and unusual inputs.
    The router should handle these gracefully without crashing.

    Edge Cases:
        - Empty query strings
        - No current answer available
        - No documents retrieved yet
        - First iteration (iteration=1)
        - Very long queries (1000+ words)
        - Low max_iterations values

    Design Philosophy:
        Router should make best-effort decisions even with minimal context.
    """

    def test_route_empty_query(self, router, mock_llm):
        """Test route with empty query string."""
        mock_response = AIMessage(
            content='{"action": "search", "reasoning": "empty query"}'
        )
        mock_llm.invoke.return_value = mock_response

        result = router.route(query="", has_documents=False)

        assert result["action"] == "search"
        mock_llm.invoke.assert_called_once()

    def test_route_no_current_answer(self, router, mock_llm):
        """Test route when no current answer exists."""
        mock_response = AIMessage(
            content='{"action": "search", "reasoning": "no answer yet"}'
        )
        mock_llm.invoke.return_value = mock_response

        result = router.route(query="test?", has_documents=False, current_answer=None)

        assert result["action"] == "search"

    def test_route_no_documents_retrieved(self, router, mock_llm):
        """Test route when documents have not been retrieved."""
        mock_response = AIMessage(
            content='{"action": "search", "reasoning": "need documents"}'
        )
        mock_llm.invoke.return_value = mock_response

        result = router.route(query="test", has_documents=False)

        assert result["action"] == "search"

    def test_route_first_iteration(self, router, mock_llm):
        """Test route at first iteration."""
        mock_response = AIMessage(
            content='{"action": "search", "reasoning": "start search"}'
        )
        mock_llm.invoke.return_value = mock_response

        result = router.route(query="test", has_documents=False, iteration=1)

        assert result["action"] == "search"

    def test_route_low_max_iterations(self, router, mock_llm):
        """Test route with very low max_iterations value."""
        result = router.route(
            query="test", has_documents=False, iteration=1, max_iterations=1
        )

        assert result["action"] == "generate"
        mock_llm.invoke.assert_not_called()

    def test_route_preserves_reasoning_text(self, router, mock_llm):
        """Test route preserves exact reasoning text from LLM."""
        reasoning = "This is a detailed reasoning about the query routing decision"
        mock_response = AIMessage(
            content=f'{{"action": "reflect", "reasoning": "{reasoning}"}}'
        )
        mock_llm.invoke.return_value = mock_response

        result = router.route(query="test", has_documents=True)

        assert result["reasoning"] == reasoning

    def test_route_with_very_long_query(self, router, mock_llm):
        """Test route with very long query string."""
        long_query = "test " * 1000
        mock_response = AIMessage(
            content='{"action": "search", "reasoning": "long query"}'
        )
        mock_llm.invoke.return_value = mock_response

        result = router.route(query=long_query, has_documents=False)

        assert result["action"] == "search"
        mock_llm.invoke.assert_called_once()


class TestAgenticRouterIntegrationPatterns(TestAgenticRouter):
    """Tests for integration patterns and usage scenarios.

    Validates typical agentic RAG workflows and multi-iteration patterns.
    Demonstrates how AgenticRouter drives the iterative retrieval loop.

    Workflow Patterns:
        - Search -> Reflect -> Generate (typical)
        - Multiple search iterations before generate
        - Max iterations preventing infinite loops
        - Complex answers requiring verification

    Loop Characteristics:
        - Each iteration calls router for next action
        - State updates between iterations
        - Generate action terminates the loop
        - Max iterations is safety limit

    Performance Considerations:
        - Each router call requires LLM inference
        - More iterations = more latency
        - Trade-off: thoroughness vs. speed
    """

    def test_search_reflect_generate_workflow(self, router, mock_llm):
        """Test typical agentic RAG workflow: search -> reflect -> generate."""
        # First call: need more info
        search_response = AIMessage(
            content='{"action": "search", "reasoning": "need documents"}'
        )
        # Second call: verify answer
        reflect_response = AIMessage(
            content='{"action": "reflect", "reasoning": "verify accuracy"}'
        )
        # Third call: generate final answer
        generate_response = AIMessage(
            content='{"action": "generate", "reasoning": "ready to answer"}'
        )

        mock_llm.invoke.side_effect = [
            search_response,
            reflect_response,
            generate_response,
        ]

        # Search phase
        result1 = router.route(query="test", has_documents=False)
        assert result1["action"] == "search"

        # Reflect phase (after getting some documents)
        result2 = router.route(
            query="test", has_documents=True, current_answer="partial answer"
        )
        assert result2["action"] == "reflect"

        # Generate phase
        result3 = router.route(
            query="test",
            has_documents=True,
            current_answer="verified answer",
            iteration=3,
        )
        assert result3["action"] == "generate"

    def test_multiple_search_iterations(self, router, mock_llm):
        """Test multiple search iterations before reflect/generate."""
        responses = [
            AIMessage(content='{"action": "search", "reasoning": "need more"}'),
            AIMessage(content='{"action": "search", "reasoning": "still need more"}'),
            AIMessage(content='{"action": "generate", "reasoning": "enough data"}'),
        ]
        mock_llm.invoke.side_effect = responses

        for i in range(3):
            result = router.route(
                query="test",
                has_documents=(i > 0),
                current_answer="partial" if i > 0 else None,
                iteration=i + 1,
                max_iterations=10,
            )
            if i < 2:
                assert result["action"] == "search"
            else:
                assert result["action"] == "generate"

    def test_max_iterations_prevents_infinite_loop(self, router, mock_llm):
        """Test that max iterations prevents infinite loop."""
        responses = [
            AIMessage(content='{"action": "search", "reasoning": "1"}'),
            AIMessage(content='{"action": "search", "reasoning": "2"}'),
            AIMessage(content='{"action": "search", "reasoning": "3"}'),
        ]
        mock_llm.invoke.side_effect = responses

        # After 3 iterations with max_iterations=3, should force generate
        for i in range(3):
            result = router.route(
                query="test",
                has_documents=False,
                iteration=i + 1,
                max_iterations=3,
            )
            if i < 2:
                assert result["action"] == "search"
            else:
                assert result["action"] == "generate"
                assert "maximum iterations" in result["reasoning"].lower()
                mock_llm.invoke.assert_called()

    def test_route_with_complex_current_answer(self, router, mock_llm):
        """Test route with complex/long current answer."""
        complex_answer = (
            "Based on the retrieved documents, Python is a high-level programming "
            "language created by Guido van Rossum in 1991. It emphasizes code "
            "readability with significant whitespace..."
        )
        mock_response = AIMessage(
            content='{"action": "generate", "reasoning": "complete answer"}'
        )
        mock_llm.invoke.return_value = mock_response

        result = router.route(
            query="What is Python?",
            has_documents=True,
            current_answer=complex_answer,
        )

        assert result["action"] == "generate"

    def test_all_three_actions_returned_correctly(self, router, mock_llm):
        """Test that all three actions can be returned correctly."""
        actions = ["search", "reflect", "generate"]
        for action in actions:
            mock_response = AIMessage(
                content=f'{{"action": "{action}", "reasoning": "test"}}'
            )
            mock_llm.invoke.return_value = mock_response

            result = router.route(query="test", has_documents=False)

            assert result["action"] == action

    def test_reasoning_includes_max_iterations_info(self, router, mock_llm):
        """Test that reasoning includes max iterations when forcing generate."""
        result = router.route(
            query="test",
            has_documents=False,
            iteration=5,
            max_iterations=5,
        )

        assert "5" in result["reasoning"]
        assert result["action"] == "generate"
