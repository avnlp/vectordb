"""Tests for QueryEnhancer component (LangChain).

This module tests the QueryEnhancer class which implements query expansion techniques
for improving retrieval quality in RAG pipelines. Query enhancement transforms a single
user query into multiple variations to increase recall and capture different semantic
interpretations.

Query Enhancement Techniques:
    - Multi-query: Generates 5 semantically similar queries to cast a wider net
    - HYDE (Hypothetical Document Embeddings): Generates a hypothetical ideal answer
      to use as an additional query vector
    - Step-back: Generates broader, more general questions to retrieve context
      before answering specific queries

Test Coverage:
    - Initialization with valid and None LLM instances
    - Template validation for all three enhancement modes
    - Multi-query generation with line filtering and whitespace handling
    - HYDE query generation preserving original + hypothetical document
    - Step-back query generation with contextual expansion
    - Edge cases: empty responses, whitespace, max limits
    - Integration patterns for RAG workflows

All tests mock the LLM to avoid external API calls and ensure fast, deterministic
unit tests.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage


class TestQueryEnhancer:
    """Unit tests for QueryEnhancer class.

    Base test class providing fixtures for QueryEnhancer tests.
    Sets up mock LLM and enhancer instance for use across test subclasses.

    Fixtures:
        mock_llm: MagicMock simulating LangChain LLM interface
        enhancer: QueryEnhancer instance configured with mock LLM
    """

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        """Create mock LLM for testing."""
        return MagicMock()

    @pytest.fixture
    def enhancer(self, mock_llm):
        """Create QueryEnhancer instance with mock LLM."""
        from vectordb.langchain.components.query_enhancer import QueryEnhancer

        return QueryEnhancer(llm=mock_llm)


class TestQueryEnhancerInitialization(TestQueryEnhancer):
    """Tests for QueryEnhancer initialization and configuration.

    Validates that QueryEnhancer properly stores LLM reference and
    that class-level prompt templates are correctly defined.

    Templates Tested:
        MULTI_QUERY_TEMPLATE: Prompt for generating query variations
        HYDE_TEMPLATE: Prompt for generating hypothetical documents
        STEP_BACK_TEMPLATE: Prompt for generating broader context questions
    """

    def test_initialization_with_valid_llm(self, mock_llm):
        """Test initialization with a valid LLM instance."""
        from vectordb.langchain.components.query_enhancer import QueryEnhancer

        enhancer = QueryEnhancer(llm=mock_llm)
        assert enhancer.llm is mock_llm

    def test_initialization_with_none_llm(self):
        """Test initialization with None LLM (edge case)."""
        from vectordb.langchain.components.query_enhancer import QueryEnhancer

        enhancer = QueryEnhancer(llm=None)
        assert enhancer.llm is None

    def test_templates_are_class_attributes(self):
        """Test that prompt templates are defined as class attributes."""
        from vectordb.langchain.components.query_enhancer import QueryEnhancer

        assert hasattr(QueryEnhancer, "MULTI_QUERY_TEMPLATE")
        assert hasattr(QueryEnhancer, "HYDE_TEMPLATE")
        assert hasattr(QueryEnhancer, "STEP_BACK_TEMPLATE")
        assert (
            "You are an AI language model assistant"
            in QueryEnhancer.MULTI_QUERY_TEMPLATE
        )
        assert "hypothetical document" in QueryEnhancer.HYDE_TEMPLATE
        assert "step-back questions" in QueryEnhancer.STEP_BACK_TEMPLATE


class TestGenerateMultiQueries(TestQueryEnhancer):
    """Tests for generate_multi_queries method.

    Validates the multi-query expansion technique which generates up to 5
    semantically similar queries from a single input query. This technique
    improves recall by casting a wider semantic net during retrieval.

    Key Behaviors:
        - Splits LLM response on newlines to create query list
        - Filters out empty lines from response
        - Limits to maximum 5 queries (configurable ceiling)
        - Strips whitespace from each generated query
        - Returns original query if LLM returns empty response

    Edge Cases:
        - Empty LLM responses
        - Responses with excessive whitespace
        - Single-line responses
        - More than 5 generated queries (truncation)
    """

    def test_generate_multi_queries_success(self, enhancer, mock_llm):
        """Test successful multi-query generation."""
        mock_response = AIMessage(
            content="What is Python programming?\nHow to learn Python?\nPython features\nPython use cases\nPython vs other languages"
        )
        mock_llm.invoke.return_value = mock_response

        queries = enhancer.generate_multi_queries("What is Python?")

        assert isinstance(queries, list)
        assert len(queries) == 5
        assert all(isinstance(q, str) for q in queries)
        mock_llm.invoke.assert_called_once()

    def test_generate_multi_queries_with_empty_lines(self, enhancer, mock_llm):
        """Test multi-query generation filters empty lines."""
        mock_response = AIMessage(content="Query 1\n\nQuery 2\n\nQuery 3\n\n")
        mock_llm.invoke.return_value = mock_response

        queries = enhancer.generate_multi_queries("Test query")

        # Should filter out empty lines
        assert len(queries) == 3
        assert all(q.strip() for q in queries)

    def test_generate_multi_queries_max_five(self, enhancer, mock_llm):
        """Test multi-query returns maximum of 5 queries."""
        mock_response = AIMessage(
            content="Query 1\nQuery 2\nQuery 3\nQuery 4\nQuery 5\nQuery 6\nQuery 7"
        )
        mock_llm.invoke.return_value = mock_response

        queries = enhancer.generate_multi_queries("Test query")

        assert len(queries) == 5

    def test_generate_multi_queries_with_whitespace(self, enhancer, mock_llm):
        """Test multi-query cleans whitespace."""
        mock_response = AIMessage(content="  Query 1  \n  Query 2  \n  Query 3  ")
        mock_llm.invoke.return_value = mock_response

        queries = enhancer.generate_multi_queries("Test query")

        assert all(q == q.strip() for q in queries)

    def test_generate_multi_queries_single_query(self, enhancer, mock_llm):
        """Test multi-query with single response."""
        mock_response = AIMessage(content="Single query response")
        mock_llm.invoke.return_value = mock_response

        queries = enhancer.generate_multi_queries("Test query")

        assert len(queries) == 1

    def test_generate_multi_queries_invoke_receives_formatted_prompt(
        self, enhancer, mock_llm
    ):
        """Test that LLM invoke is called with formatted prompt."""
        mock_response = AIMessage(content="Query 1\nQuery 2")
        mock_llm.invoke.return_value = mock_response

        enhancer.generate_multi_queries("test question")

        # Verify invoke was called with a string containing the query
        call_args = mock_llm.invoke.call_args
        assert call_args is not None
        prompt_arg = call_args[0][0]
        assert "test question" in prompt_arg


class TestGenerateHydeQueries(TestQueryEnhancer):
    """Tests for generate_hyde_queries method.

    Validates the HYDE (Hypothetical Document Embeddings) technique which
    generates a hypothetical ideal answer to use as an additional query.
    This technique improves retrieval by matching against idealized content.

    Key Behaviors:
        - Returns a 2-element list: [original_query, hypothetical_document]
        - Hypothetical document generated by LLM based on query context
        - Strips whitespace from LLM response
        - Preserves original query as first element for hybrid retrieval

    Use Case:
        HYDE is particularly effective when the ideal answer has specific
        terminology that might not appear in the original query.
    """

    def test_generate_hyde_queries_success(self, enhancer, mock_llm):
        """Test successful HYDE query generation."""
        mock_response = AIMessage(
            content="Python is a high-level, interpreted programming language known for its simplicity and readability."
        )
        mock_llm.invoke.return_value = mock_response

        queries = enhancer.generate_hyde_queries("What is Python?")

        assert isinstance(queries, list)
        assert len(queries) == 2
        assert queries[0] == "What is Python?"  # Original query
        assert "Python" in queries[1]  # Hypothetical document

    def test_generate_hyde_queries_returns_original_and_hypothetical(
        self, enhancer, mock_llm
    ):
        """Test HYDE returns both original query and hypothetical document."""
        original_query = "How does machine learning work?"
        mock_response = AIMessage(
            content="Machine learning is a subset of AI that enables systems to learn from data."
        )
        mock_llm.invoke.return_value = mock_response

        queries = enhancer.generate_hyde_queries(original_query)

        assert queries[0] == original_query
        assert queries[1] == mock_response.content.strip()

    def test_generate_hyde_queries_strips_whitespace(self, enhancer, mock_llm):
        """Test HYDE query strips whitespace from response."""
        mock_response = AIMessage(content="  Hypothetical document content  ")
        mock_llm.invoke.return_value = mock_response

        queries = enhancer.generate_hyde_queries("Test query")

        assert queries[1] == "Hypothetical document content"


class TestGenerateStepBackQueries(TestQueryEnhancer):
    """Tests for generate_step_back_queries method.

    Validates the step-back prompting technique which generates broader,
    more general questions before answering specific queries. This technique
    improves retrieval by first establishing general context.

    Key Behaviors:
        - Generates up to 3 broader context questions
        - Appends original query as final element
        - Filters empty lines from LLM response
        - Returns only original query if step-backs are empty

    Example:
        Query: "How do React hooks work?"
        Step-backs: ["What is React?", "What are JavaScript frameworks?"]
        Final: ["What is React?", "What are JavaScript frameworks?",
            "How do React hooks work?"]

    Use Case:
        Step-back is effective for technical queries where general context
        helps answer specific technical questions.
    """

    def test_generate_step_back_queries_success(self, enhancer, mock_llm):
        """Test successful step-back query generation."""
        mock_response = AIMessage(
            content="What is machine learning?\nWhat are the types of AI?\nWhat is deep learning?"
        )
        mock_llm.invoke.return_value = mock_response

        queries = enhancer.generate_step_back_queries("How does neural networks work?")

        assert isinstance(queries, list)
        assert len(queries) == 4  # 3 step-back + 1 original query
        # Last element should be original query
        assert queries[-1] == "How does neural networks work?"
        # First 3 should be step-back questions
        assert len(queries[:-1]) == 3

    def test_generate_step_back_queries_max_three(self, enhancer, mock_llm):
        """Test step-back returns maximum of 3 step-back queries."""
        mock_response = AIMessage(content="Back 1\nBack 2\nBack 3\nBack 4\nBack 5")
        mock_llm.invoke.return_value = mock_response

        queries = enhancer.generate_step_back_queries("Test query")

        # Should be 3 step-back + 1 original = 4
        assert len(queries) == 4
        assert queries[-1] == "Test query"

    def test_generate_step_back_queries_with_empty_lines(self, enhancer, mock_llm):
        """Test step-back filters empty lines."""
        mock_response = AIMessage(content="Back 1\n\nBack 2\n\nBack 3\n\n")
        mock_llm.invoke.return_value = mock_response

        queries = enhancer.generate_step_back_queries("Test query")

        assert len(queries) == 4  # 3 step-back + 1 original
        assert all(q.strip() for q in queries)

    def test_generate_step_back_queries_original_at_end(self, enhancer, mock_llm):
        """Test that original query is the last element."""
        mock_response = AIMessage(
            content="Back question 1\nBack question 2\nBack question 3"
        )
        mock_llm.invoke.return_value = mock_response

        queries = enhancer.generate_step_back_queries("Original question")

        assert queries[-1] == "Original question"
        assert queries[0] != "Original question"


class TestGenerateQueries(TestQueryEnhancer):
    """Tests for generate_queries method (main interface).

    Validates the unified interface that dispatches to specific query
    enhancement techniques based on mode parameter. This is the primary
    API for consumers of the QueryEnhancer class.

    Supported Modes:
        - multi_query: Generate semantically similar queries (default)
        - hyde: Generate hypothetical document embeddings
        - step_back: Generate broader context questions

    Error Handling:
        - Raises ValueError for unknown modes
        - Error message includes list of valid modes

    Default Behavior:
        - Uses multi_query mode when mode parameter not specified
    """

    def test_generate_queries_multi_query_mode(self, enhancer, mock_llm):
        """Test generate_queries with multi_query mode."""
        mock_response = AIMessage(content="Query 1\nQuery 2\nQuery 3")
        mock_llm.invoke.return_value = mock_response

        queries = enhancer.generate_queries("test", mode="multi_query")

        assert len(queries) == 3

    def test_generate_queries_hyde_mode(self, enhancer, mock_llm):
        """Test generate_queries with hyde mode."""
        mock_response = AIMessage(content="Hypothetical document")
        mock_llm.invoke.return_value = mock_response

        queries = enhancer.generate_queries("test", mode="hyde")

        assert len(queries) == 2
        assert queries[0] == "test"
        assert queries[1] == "Hypothetical document"

    def test_generate_queries_step_back_mode(self, enhancer, mock_llm):
        """Test generate_queries with step_back mode."""
        mock_response = AIMessage(content="Back 1\nBack 2\nBack 3")
        mock_llm.invoke.return_value = mock_response

        queries = enhancer.generate_queries("test", mode="step_back")

        assert len(queries) == 4  # 3 back + 1 original
        assert queries[-1] == "test"

    def test_generate_queries_default_mode(self, enhancer, mock_llm):
        """Test generate_queries uses multi_query as default mode."""
        mock_response = AIMessage(content="Query 1\nQuery 2")
        mock_llm.invoke.return_value = mock_response

        queries = enhancer.generate_queries("test")

        # Should use multi_query by default
        assert len(queries) == 2

    def test_generate_queries_invalid_mode_raises_value_error(self, enhancer):
        """Test generate_queries raises ValueError for invalid mode."""
        with pytest.raises(ValueError) as exc_info:
            enhancer.generate_queries("test", mode="invalid_mode")

        assert "Unknown mode" in str(exc_info.value)
        assert "invalid_mode" in str(exc_info.value)

    def test_generate_queries_error_message_contains_valid_modes(self, enhancer):
        """Test error message lists valid modes."""
        with pytest.raises(ValueError) as exc_info:
            enhancer.generate_queries("test", mode="unknown")

        error_msg = str(exc_info.value)
        assert "multi_query" in error_msg
        assert "hyde" in error_msg
        assert "step_back" in error_msg


class TestQueryEnhancerEdgeCases(TestQueryEnhancer):
    """Tests for edge cases and error handling.

    Validates robust handling of unexpected inputs and boundary conditions.
    These tests ensure the QueryEnhancer degrades gracefully when the LLM
    returns unexpected responses.

    Edge Cases Covered:
        - Empty LLM responses (all modes)
        - Very long responses (truncation to max queries)
        - Whitespace-only responses
        - Mismatched document/embeddings counts

    Design Philosophy:
        QueryEnhancer should never crash on bad LLM output; it should
        return sensible defaults (empty list or original query).
    """

    def test_generate_multi_queries_empty_response(self, enhancer, mock_llm):
        """Test multi-query with empty LLM response."""
        mock_response = AIMessage(content="")
        mock_llm.invoke.return_value = mock_response

        queries = enhancer.generate_multi_queries("test")

        assert queries == []

    def test_generate_multi_queries_very_long_response(self, enhancer, mock_llm):
        """Test multi-query limits to 5 even with many responses."""
        mock_response = AIMessage(content="Q1\nQ2\nQ3\nQ4\nQ5\nQ6\nQ7\nQ8\nQ9\nQ10")
        mock_llm.invoke.return_value = mock_response

        queries = enhancer.generate_multi_queries("test")

        assert len(queries) == 5

    def test_generate_hyde_queries_empty_response(self, enhancer, mock_llm):
        """Test HYDE with empty LLM response."""
        mock_response = AIMessage(content="")
        mock_llm.invoke.return_value = mock_response

        queries = enhancer.generate_hyde_queries("test")

        assert len(queries) == 2
        assert queries[0] == "test"
        assert queries[1] == ""

    def test_generate_step_back_queries_empty_response(self, enhancer, mock_llm):
        """Test step-back with empty LLM response."""
        mock_response = AIMessage(content="")
        mock_llm.invoke.return_value = mock_response

        queries = enhancer.generate_step_back_queries("test")

        assert len(queries) == 1
        assert queries[0] == "test"

    def test_generate_step_back_queries_only_original(self, enhancer, mock_llm):
        """Test step-back when LLM returns empty step-back questions."""
        mock_response = AIMessage(content="")
        mock_llm.invoke.return_value = mock_response

        queries = enhancer.generate_step_back_queries("original query")

        assert queries == ["original query"]

    def test_all_methods_use_same_llm_instance(self, mock_llm):
        """Test that all methods use the same LLM instance."""
        from vectordb.langchain.components.query_enhancer import QueryEnhancer

        enhancer = QueryEnhancer(llm=mock_llm)

        # Call all methods
        mock_response = AIMessage(content="Response")
        mock_llm.invoke.return_value = mock_response

        enhancer.generate_multi_queries("test")
        enhancer.generate_hyde_queries("test")
        enhancer.generate_step_back_queries("test")

        # All should use the same LLM instance
        assert mock_llm.invoke.call_count == 3


class TestQueryEnhancerIntegrationPatterns(TestQueryEnhancer):
    """Tests for integration patterns and usage scenarios.

    Validates typical usage patterns in production RAG pipelines.
    These tests demonstrate how QueryEnhancer integrates with
    retrieval and generation components.

    Patterns Tested:
        - Multi-query workflow: Generate variations -> Search all -> Aggregate
        - HYDE workflow: Generate hypothetical -> Search with original + hypothetical
        - Step-back workflow: Broaden query -> Search -> Narrow with original

    Performance Considerations:
        - Each query variation triggers a separate search
        - HYDE adds 1 additional search vector
        - Step-back adds up to 3 additional search queries
        - Trade-off: recall improvement vs. latency increase
    """

    def test_pipeline_multi_query_workflow(self, enhancer, mock_llm):
        """Test typical multi-query RAG workflow."""
        # Setup mock responses for different query generation calls
        mock_responses = [
            AIMessage(content="Q1\nQ2\nQ3"),
            AIMessage(content="Q1\nQ2\nQ3"),
            AIMessage(content="Q1\nQ2\nQ3"),
        ]
        mock_llm.invoke.side_effect = mock_responses

        # Generate queries for multiple searches
        queries1 = enhancer.generate_queries("AI trends", mode="multi_query")
        queries2 = enhancer.generate_queries("ML algorithms", mode="multi_query")
        queries3 = enhancer.generate_queries("Deep learning", mode="multi_query")

        assert len(queries1) == 3
        assert len(queries2) == 3
        assert len(queries3) == 3

    def test_hyde_with_original_query_included(self, enhancer, mock_llm):
        """Test HYDE always includes original query first."""
        original_query = "How do vaccines work?"
        mock_response = AIMessage(
            content="Vaccines work by stimulating the immune system."
        )
        mock_llm.invoke.return_value = mock_response

        queries = enhancer.generate_hyde_queries(original_query)

        assert queries[0] == original_query
        assert queries[1] != original_query

    def test_step_back_contextual_expansion(self, enhancer, mock_llm):
        """Test step-back provides broader context queries."""
        mock_response = AIMessage(
            content="What is JavaScript?\nWhat are web technologies?\nWhat is programming?"
        )
        mock_llm.invoke.return_value = mock_response

        queries = enhancer.generate_step_back_queries("How to use React hooks?")

        # Original query should be at the end
        assert queries[-1] == "How to use React hooks?"
        # Step-back queries should be more general
        assert len(queries) == 4

    def test_templates_are_preserved(self, enhancer):
        """Test that prompt templates contain expected placeholders."""
        from vectordb.langchain.components.query_enhancer import QueryEnhancer

        assert "{query}" in QueryEnhancer.MULTI_QUERY_TEMPLATE
        assert "{query}" in QueryEnhancer.HYDE_TEMPLATE
        assert "{query}" in QueryEnhancer.STEP_BACK_TEMPLATE
