"""Comprehensive tests for Haystack agentic router components.

This module tests the AgenticRouter which routes queries to appropriate
tools and reasoning paths based on query characteristics.
"""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestAgenticRouter:
    """Test suite for AgenticRouter component.

    Tests cover:
    - Initialization with different configurations
    - Tool selection for various query types
    - Self-reflection and answer quality evaluation
    - Answer refinement and self-reflection loops
    - Error handling and edge cases
    """

    @pytest.fixture
    def mock_generator(self):
        """Fixture for mocked OpenAIChatGenerator."""
        with patch(
            "vectordb.haystack.components.agentic_router.OpenAIChatGenerator"
        ) as mock:
            mock_gen = MagicMock()
            mock.return_value = mock_gen
            yield mock, mock_gen

    def test_initialization_default(self, mock_generator):
        """Test AgenticRouter initialization with default parameters."""
        mock_cls, mock_gen = mock_generator

        from vectordb.haystack.components.agentic_router import AgenticRouter

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            router = AgenticRouter()

            assert router is not None
            assert router.generator is mock_gen
            assert router.available_tools == [
                "retrieval",
                "web_search",
                "calculation",
                "reasoning",
            ]
            mock_cls.assert_called_once()
            call_args = mock_cls.call_args
            assert call_args is not None
            assert call_args.kwargs["api_base_url"] == "https://api.groq.com/openai/v1"
            assert call_args.kwargs["model"] == "llama-3.3-70b-versatile"

    def test_initialization_custom_model(self, mock_generator):
        """Test AgenticRouter initialization with custom model and API key."""
        mock_cls, mock_gen = mock_generator

        from vectordb.haystack.components.agentic_router import AgenticRouter

        router = AgenticRouter(
            model="llama-3.1-8b-instant",
            api_key="test-api-key",
        )

        assert router is not None
        assert router.generator is mock_gen
        mock_cls.assert_called_once()
        call_args = mock_cls.call_args
        assert call_args is not None
        assert call_args.kwargs["model"] == "llama-3.1-8b-instant"

    def test_initialization_failure(self, mock_generator):
        """Test AgenticRouter initialization failure handling."""
        mock_cls, _ = mock_generator
        mock_cls.side_effect = Exception("API connection failed")

        from vectordb.haystack.components.agentic_router import AgenticRouter

        with (
            patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}),
            pytest.raises(Exception, match="API connection failed"),
        ):
            AgenticRouter()

    def test_select_tool_retrieval(self, mock_generator):
        """Test tool selection for retrieval queries."""
        _, mock_gen = mock_generator
        mock_response = MagicMock()
        mock_response.text = "retrieval"
        mock_gen.run.return_value = {"replies": [mock_response]}

        from vectordb.haystack.components.agentic_router import AgenticRouter

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            router = AgenticRouter()
            result = router.select_tool("What is machine learning?")

        assert result == "retrieval"
        mock_gen.run.assert_called_once()
        call_args = mock_gen.run.call_args
        assert call_args is not None
        messages = call_args.kwargs["messages"]
        assert len(messages) == 1
        assert "What is machine learning?" in messages[0].text

    def test_select_tool_web_search(self, mock_generator):
        """Test tool selection for web search queries."""
        _, mock_gen = mock_generator
        mock_response = MagicMock()
        mock_response.text = "web_search"
        mock_gen.run.return_value = {"replies": [mock_response]}

        from vectordb.haystack.components.agentic_router import AgenticRouter

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            router = AgenticRouter()
            result = router.select_tool("What is the latest news today?")

        assert result == "web_search"

    def test_select_tool_calculation(self, mock_generator):
        """Test tool selection for calculation queries."""
        _, mock_gen = mock_generator
        mock_response = MagicMock()
        mock_response.text = "calculation"
        mock_gen.run.return_value = {"replies": [mock_response]}

        from vectordb.haystack.components.agentic_router import AgenticRouter

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            router = AgenticRouter()
            result = router.select_tool("What is 2 + 2?")

        assert result == "calculation"

    def test_select_tool_reasoning(self, mock_generator):
        """Test tool selection for reasoning queries."""
        _, mock_gen = mock_generator
        mock_response = MagicMock()
        mock_response.text = "reasoning"
        mock_gen.run.return_value = {"replies": [mock_response]}

        from vectordb.haystack.components.agentic_router import AgenticRouter

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            router = AgenticRouter()
            result = router.select_tool("Analyze the pros and cons of AI")

        assert result == "reasoning"

    def test_select_tool_invalid_response_fallback(self, mock_generator):
        """Test tool selection fallback for invalid LLM response."""
        _, mock_gen = mock_generator
        mock_response = MagicMock()
        mock_response.text = "invalid_tool"
        mock_gen.run.return_value = {"replies": [mock_response]}

        from vectordb.haystack.components.agentic_router import AgenticRouter

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            router = AgenticRouter()
            result = router.select_tool("Some query")

        # Should fallback to retrieval
        assert result == "retrieval"

    def test_select_tool_exception_fallback(self, mock_generator):
        """Test tool selection fallback when LLM raises exception."""
        _, mock_gen = mock_generator
        mock_gen.run.side_effect = Exception("LLM error")

        from vectordb.haystack.components.agentic_router import AgenticRouter

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            router = AgenticRouter()
            result = router.select_tool("Some query")

        # Should fallback to retrieval
        assert result == "retrieval"

    def test_select_tool_case_insensitive(self, mock_generator):
        """Test tool selection handles case-insensitive responses."""
        _, mock_gen = mock_generator
        mock_response = MagicMock()
        mock_response.text = "  RETRIEVAL  "
        mock_gen.run.return_value = {"replies": [mock_response]}

        from vectordb.haystack.components.agentic_router import AgenticRouter

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            router = AgenticRouter()
            result = router.select_tool("What is AI?")

        assert result == "retrieval"

    def test_evaluate_answer_quality_success(self, mock_generator):
        """Test successful answer quality evaluation."""
        _, mock_gen = mock_generator
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "relevance": 85,
                "completeness": 90,
                "grounding": 80,
                "issues": ["Minor issue 1"],
                "suggestions": ["Add more detail"],
            }
        )
        mock_gen.run.return_value = {"replies": [mock_response]}

        from vectordb.haystack.components.agentic_router import AgenticRouter

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            router = AgenticRouter()
            result = router.evaluate_answer_quality(
                query="What is AI?",
                answer="AI is artificial intelligence.",
                context="AI refers to machines that can perform tasks requiring human intelligence.",
            )

        assert result["relevance"] == 85
        assert result["completeness"] == 90
        assert result["grounding"] == 80
        assert result["issues"] == ["Minor issue 1"]
        assert result["suggestions"] == ["Add more detail"]

    def test_evaluate_answer_quality_no_context(self, mock_generator):
        """Test answer quality evaluation without context."""
        _, mock_gen = mock_generator
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "relevance": 90,
                "completeness": 85,
                "grounding": 75,
                "issues": [],
                "suggestions": [],
            }
        )
        mock_gen.run.return_value = {"replies": [mock_response]}

        from vectordb.haystack.components.agentic_router import AgenticRouter

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            router = AgenticRouter()
            result = router.evaluate_answer_quality(
                query="What is AI?",
                answer="AI is artificial intelligence.",
            )

        assert result["relevance"] == 90
        assert result["grounding"] == 75

    def test_evaluate_answer_quality_json_parse_error(self, mock_generator):
        """Test answer quality evaluation with invalid JSON response."""
        _, mock_gen = mock_generator
        mock_response = MagicMock()
        mock_response.text = "Not valid JSON"
        mock_gen.run.return_value = {"replies": [mock_response]}

        from vectordb.haystack.components.agentic_router import AgenticRouter

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            router = AgenticRouter()
            result = router.evaluate_answer_quality(
                query="What is AI?",
                answer="AI is artificial intelligence.",
            )

        assert result["relevance"] == 0
        assert result["completeness"] == 0
        assert result["grounding"] == 0
        assert result["issues"] == ["Evaluation failed"]
        assert result["suggestions"] == []

    def test_evaluate_answer_quality_exception(self, mock_generator):
        """Test answer quality evaluation when LLM raises exception."""
        _, mock_gen = mock_generator
        mock_gen.run.side_effect = Exception("LLM timeout")

        from vectordb.haystack.components.agentic_router import AgenticRouter

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            router = AgenticRouter()
            result = router.evaluate_answer_quality(
                query="What is AI?",
                answer="AI is artificial intelligence.",
            )

        assert result["relevance"] == 0
        assert result["issues"] == ["Evaluation failed"]

    def test_should_refine_answer_true(self, mock_generator):
        """Test should_refine_answer returns True for low scores."""
        from vectordb.haystack.components.agentic_router import AgenticRouter

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            router = AgenticRouter()
            eval_result = {
                "relevance": 50,
                "completeness": 60,
                "grounding": 55,
            }

            result = router.should_refine_answer(eval_result, threshold=70)

        assert result is True

    def test_should_refine_answer_false(self, mock_generator):
        """Test should_refine_answer returns False for high scores."""
        from vectordb.haystack.components.agentic_router import AgenticRouter

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            router = AgenticRouter()
            eval_result = {
                "relevance": 80,
                "completeness": 85,
                "grounding": 90,
            }

            result = router.should_refine_answer(eval_result, threshold=70)

        assert result is False

    def test_should_refine_answer_exact_threshold(self, mock_generator):
        """Test should_refine_answer at exact threshold boundary."""
        from vectordb.haystack.components.agentic_router import AgenticRouter

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            router = AgenticRouter()
            eval_result = {
                "relevance": 70,
                "completeness": 70,
                "grounding": 70,
            }

            result = router.should_refine_answer(eval_result, threshold=70)

        # Average is exactly 70, so should NOT refine (needs to be < threshold)
        assert result is False

    def test_should_refine_answer_missing_keys(self, mock_generator):
        """Test should_refine_answer with missing evaluation keys."""
        from vectordb.haystack.components.agentic_router import AgenticRouter

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            router = AgenticRouter()
            eval_result = {
                "relevance": 50,
                # completeness and grounding missing
            }

            result = router.should_refine_answer(eval_result, threshold=70)

        # Average is 50/3 = 16.67, so should refine
        assert result is True

    def test_refine_answer_success(self, mock_generator):
        """Test successful answer refinement."""
        _, mock_gen = mock_generator
        mock_response = MagicMock()
        mock_response.text = "Refined answer with more detail"
        mock_gen.run.return_value = {"replies": [mock_response]}

        from vectordb.haystack.components.agentic_router import AgenticRouter

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            router = AgenticRouter()
            eval_result = {
                "issues": ["Too brief", "Missing examples"],
                "suggestions": ["Add examples", "Expand explanation"],
            }

        result = router.refine_answer(
            query="What is AI?",
            answer="AI is artificial intelligence.",
            eval_result=eval_result,
        )

        assert result == "Refined answer with more detail"
        mock_gen.run.assert_called_once()
        call_args = mock_gen.run.call_args
        assert call_args is not None
        messages = call_args.kwargs["messages"]
        prompt_text = messages[0].text
        assert "What is AI?" in prompt_text
        assert "AI is artificial intelligence" in prompt_text

    def test_refine_answer_empty_issues(self, mock_generator):
        """Test answer refinement with empty issues and suggestions."""
        _, mock_gen = mock_generator
        mock_response = MagicMock()
        mock_response.text = "Improved answer"
        mock_gen.run.return_value = {"replies": [mock_response]}

        from vectordb.haystack.components.agentic_router import AgenticRouter

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            router = AgenticRouter()
            eval_result = {
                "issues": [],
                "suggestions": [],
            }

        result = router.refine_answer(
            query="What is AI?",
            answer="AI is artificial intelligence.",
            eval_result=eval_result,
        )

        assert result == "Improved answer"

    def test_refine_answer_exception_fallback(self, mock_generator):
        """Test answer refinement fallback when LLM fails."""
        _, mock_gen = mock_generator
        mock_gen.run.side_effect = Exception("LLM error")

        from vectordb.haystack.components.agentic_router import AgenticRouter

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            router = AgenticRouter()
            original_answer = "AI is artificial intelligence."
        eval_result = {
            "issues": ["Too brief"],
            "suggestions": ["Add detail"],
        }

        result = router.refine_answer(
            query="What is AI?",
            answer=original_answer,
            eval_result=eval_result,
        )

        # Should return original answer on failure
        assert result == original_answer

    def test_self_reflect_loop_no_refinement_needed(self, mock_generator):
        """Test self-reflection loop when quality is already good."""
        _, mock_gen = mock_generator
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "relevance": 90,
                "completeness": 92,
                "grounding": 88,
                "issues": [],
                "suggestions": [],
            }
        )
        mock_gen.run.return_value = {"replies": [mock_response]}

        from vectordb.haystack.components.agentic_router import AgenticRouter

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            router = AgenticRouter()
            result = router.self_reflect_loop(
                query="What is AI?",
                answer="AI is artificial intelligence.",
                context="Some context",
                max_iterations=2,
                quality_threshold=75,
            )

        # Should return original answer since quality is above threshold
        assert result == "AI is artificial intelligence."
        # Should only call LLM once (for evaluation)
        assert mock_gen.run.call_count == 1

    def test_self_reflect_loop_with_refinement(self, mock_generator):
        """Test self-reflection loop with one refinement iteration."""
        _, mock_gen = mock_generator

        # First call: evaluation (low scores)
        # Second call: refinement
        # Third call: evaluation (high scores)
        mock_responses = [
            MagicMock(
                text=json.dumps(
                    {
                        "relevance": 50,
                        "completeness": 60,
                        "grounding": 55,
                        "issues": ["Too brief"],
                        "suggestions": ["Add detail"],
                    }
                )
            ),
            MagicMock(text="Refined answer with more detail"),
            MagicMock(
                text=json.dumps(
                    {
                        "relevance": 85,
                        "completeness": 88,
                        "grounding": 82,
                        "issues": [],
                        "suggestions": [],
                    }
                )
            ),
        ]
        mock_gen.run.side_effect = [
            {"replies": [mock_responses[0]]},
            {"replies": [mock_responses[1]]},
            {"replies": [mock_responses[2]]},
        ]

        from vectordb.haystack.components.agentic_router import AgenticRouter

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            router = AgenticRouter()
            result = router.self_reflect_loop(
                query="What is AI?",
                answer="AI is artificial intelligence.",
                max_iterations=2,
                quality_threshold=75,
            )

        assert result == "Refined answer with more detail"
        assert mock_gen.run.call_count == 3

    def test_self_reflect_loop_max_iterations(self, mock_generator):
        """Test self-reflection loop respects max_iterations."""
        _, mock_gen = mock_generator

        # Always return low scores to trigger refinement
        mock_responses = [
            MagicMock(
                text=json.dumps(
                    {
                        "relevance": 50,
                        "completeness": 60,
                        "grounding": 55,
                        "issues": ["Too brief"],
                        "suggestions": ["Add detail"],
                    }
                )
            ),
            MagicMock(text="Refined answer 1"),
            MagicMock(
                text=json.dumps(
                    {
                        "relevance": 55,
                        "completeness": 58,
                        "grounding": 52,
                        "issues": ["Still brief"],
                        "suggestions": ["More detail"],
                    }
                )
            ),
            MagicMock(text="Refined answer 2"),
        ]
        mock_gen.run.side_effect = [
            {"replies": [mock_responses[0]]},
            {"replies": [mock_responses[1]]},
            {"replies": [mock_responses[2]]},
            {"replies": [mock_responses[3]]},
        ]

        from vectordb.haystack.components.agentic_router import AgenticRouter

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            router = AgenticRouter()
            result = router.self_reflect_loop(
                query="What is AI?",
                answer="AI is artificial intelligence.",
                max_iterations=2,
                quality_threshold=75,
            )

        # Should stop after max_iterations (2)
        assert result == "Refined answer 2"
        assert mock_gen.run.call_count == 4  # 2 eval + 2 refine

    def test_self_reflect_loop_zero_iterations(self, mock_generator):
        """Test self-reflection loop with zero iterations."""
        from vectordb.haystack.components.agentic_router import AgenticRouter

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            router = AgenticRouter()
            original_answer = "AI is artificial intelligence."

        result = router.self_reflect_loop(
            query="What is AI?",
            answer=original_answer,
            max_iterations=0,
            quality_threshold=75,
        )

        # Should return original answer immediately
        assert result == original_answer

    def test_available_tools_list(self, mock_generator):
        """Test that available tools list is correct."""
        from vectordb.haystack.components.agentic_router import AgenticRouter

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            router = AgenticRouter()

        assert "retrieval" in router.available_tools
        assert "web_search" in router.available_tools
        assert "calculation" in router.available_tools
        assert "reasoning" in router.available_tools
        assert len(router.available_tools) == 4

    def test_select_tool_with_whitespace(self, mock_generator):
        """Test tool selection handles whitespace in response."""
        _, mock_gen = mock_generator
        mock_response = MagicMock()
        mock_response.text = "  retrieval  \n"
        mock_gen.run.return_value = {"replies": [mock_response]}

        from vectordb.haystack.components.agentic_router import AgenticRouter

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            router = AgenticRouter()
            result = router.select_tool("What is AI?")

        assert result == "retrieval"
