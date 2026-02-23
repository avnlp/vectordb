"""Query enhancement component for LangChain pipelines.

This module provides query enhancement capabilities to improve retrieval quality
in RAG applications. By generating multiple query variations, we can overcome
vocabulary mismatch and retrieve more relevant documents from the vector store.

Query Enhancement Strategies:
    1. Multi-Query Generation: Creates alternative phrasings of the original query.
       This addresses the vocabulary mismatch problem where the user's query uses
       different terminology than the indexed documents.

    2. HyDE (Hypothetical Document Embeddings): Generates a hypothetical answer
       to the query and uses it for retrieval. This bridges the gap between
       query and document distributions by transforming the query into the
       "document space".

    3. Step-Back Prompting: Generates broader, more general questions that provide
       context for the specific query. This helps retrieve background information
       needed to answer complex questions.

When to Use Each Strategy:
    - Multi-Query: Best for simple factual queries where different phrasings
      might match different documents. Good for domain-specific terminology.

    - HyDE: Best when the query is very short or the query/document distributions
      differ significantly (e.g., questions vs. encyclopedia articles). Note that
      HyDE requires an LLM call per query, adding latency.

    - Step-Back: Best for complex questions requiring background knowledge.
      The step-back questions retrieve context that helps answer the specific query.

Integration with LangChain:
    The component uses LangChain's PromptTemplate for consistent prompt formatting
    and ChatGroq for LLM-based query generation. Generated queries can be used
    with any LangChain retriever.

Performance Considerations:
    - Multi-query and step-back add minimal latency (single LLM call)
    - HyDE adds one LLM call per query, which can be significant at scale
    - Consider caching enhanced queries for frequently-asked questions
    - Each strategy increases retrieval load (more queries = more vector searches)

Usage:
    >>> from langchain_groq import ChatGroq
    >>> from vectordb.langchain.components import QueryEnhancer
    >>> llm = ChatGroq(model="llama-3.3-70b-versatile")
    >>> enhancer = QueryEnhancer(llm)
    >>> # Multi-query generation
    >>> queries = enhancer.generate_queries("What is quantum computing?", "multi_query")
    >>> # HyDE generation
    >>> queries = enhancer.generate_queries("Explain neural networks", "hyde")
    >>> # Step-back prompting
    >>> queries = enhancer.generate_queries("What is backpropagation?", "step_back")
"""

import logging

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq


# Module-level logger for query enhancement operations
logger = logging.getLogger(__name__)


class QueryEnhancer:
    """Generate multiple query perspectives for enhanced retrieval.

    This component addresses the vocabulary mismatch problem in information
    retrieval by generating query variations that capture different ways of
    expressing the same information need. It implements three complementary
    strategies: multi-query generation, HyDE, and step-back prompting.

    Attributes:
        llm: LangChain LLM instance used for query generation.
        MULTI_QUERY_TEMPLATE: Prompt for generating alternative query phrasings.
        HYDE_TEMPLATE: Prompt for generating hypothetical document answers.
        STEP_BACK_TEMPLATE: Prompt for generating broader context questions.

    Design Pattern:
        The class follows the Strategy pattern, with each enhancement mode
        implementing a different approach to query expansion. The common
        interface (generate_queries method) allows runtime selection of
        enhancement strategy.

    Retrieval Augmentation:
        Each enhancement strategy increases recall by casting a wider net:
        - Multi-query: 5x query volume, captures terminology variations
        - HyDE: 2x query volume (original + hypothetical), bridges distribution gap
        - Step-back: 4x query volume (3 step-back + original), retrieves context

        The trade-off is increased retrieval latency and potential noise.
        Consider combining with reranking to maintain precision.
    """

    MULTI_QUERY_TEMPLATE = """You are an AI language model assistant. Your task is to generate 5 different search queries that would help answer the given question. Provide only the queries, one per line, without numbering or bullet points.

Original question: {query}

Alternative queries:"""

    HYDE_TEMPLATE = """You are an AI language model assistant. Your task is to generate a hypothetical document that would answer the given question. Write a brief, focused response (2-3 sentences) that directly answers the question.

Question: {query}

Hypothetical document:"""

    STEP_BACK_TEMPLATE = """You are an AI language model assistant. Your task is to generate 3 step-back questions that would provide broader context for answering the given question. These are more general, foundational questions. Provide only the questions, one per line, without numbering.

Original question: {query}

Step-back questions:"""

    def __init__(self, llm: ChatGroq) -> None:
        """Initialize QueryEnhancer with a LangChain LLM instance.

        Args:
            llm: ChatGroq instance for generating query variations. Should be
                configured with appropriate temperature (0.3-0.7 recommended)
                for creative but coherent query generation.

        Note:
            The LLM temperature affects query diversity:
            - Lower (0.0-0.3): More conservative, similar to original query
            - Higher (0.5-0.7): More diverse, may capture more variations
            - Too high (>0.8): May generate irrelevant or off-topic queries
        """
        self.llm = llm

    def generate_multi_queries(self, query: str) -> list[str]:
        """Generate alternative query formulations for the same information need.

        This method addresses vocabulary mismatch by creating different phrasings
        of the original query. Each variation may match different documents in
        the vector store, increasing recall.

        Args:
            query: The original user query text.

        Returns:
            List of alternative query strings (up to 5). Each query is a
            different way of asking for the same information. The original
            query is NOT included in the returned list.

        Example:
            >>> enhancer = QueryEnhancer(llm)
            >>> queries = enhancer.generate_multi_queries("What is AI?")
            >>> queries
            ['Define artificial intelligence',
             'Explain what AI means',
             'What does artificial intelligence refer to',
             ...]

        Note:
            The LLM is instructed to return one query per line without numbering.
            We parse the response by splitting on newlines and filtering empty lines.
            If the LLM returns fewer than 5 queries, we return what we have.
        """
        # Construct the prompt using the multi-query template
        prompt = PromptTemplate(
            template=self.MULTI_QUERY_TEMPLATE,
            input_variables=["query"],
        )
        formatted_prompt = prompt.format(query=query)

        # Invoke the LLM to generate alternative queries
        response = self.llm.invoke(formatted_prompt)

        # Parse the response: split by newlines, strip whitespace, filter empty lines
        queries = response.content.strip().split("\n")
        queries = [q.strip() for q in queries if q.strip()]

        # Return at most 5 queries to control retrieval cost
        return queries[:5]

    def generate_hyde_queries(self, query: str) -> list[str]:
        """Generate hypothetical document for HyDE-based retrieval.

        HyDE (Hypothetical Document Embeddings) is a technique where we generate
        a hypothetical answer to the query, then use that answer for retrieval
        instead of the original query. This bridges the distribution gap between
        questions (short, interrogative) and documents (long, declarative).

        Args:
            query: The original user query text.

        Returns:
            List containing the original query followed by the hypothetical
            document answer. The returned list always has 2 elements:
            [original_query, hypothetical_answer].

        Example:
            >>> enhancer = QueryEnhancer(llm)
            >>> queries = enhancer.generate_hyde_queries("What is photosynthesis?")
            >>> queries
            ['What is photosynthesis?',
             'Photosynthesis is the process by which plants convert light energy...']

        Reference:
            HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels
            (Gao et al., 2022) - https://arxiv.org/abs/2212.10496

        Note:
            The hypothetical document is generated by the LLM and may contain
            hallucinations. However, for retrieval purposes, even an imperfect
            hypothetical document often retrieves better than the original query
            because it shares the same distribution as the indexed documents.
        """
        # Construct the prompt using the HyDE template
        prompt = PromptTemplate(
            template=self.HYDE_TEMPLATE,
            input_variables=["query"],
        )
        formatted_prompt = prompt.format(query=query)

        # Invoke the LLM to generate a hypothetical answer
        response = self.llm.invoke(formatted_prompt)
        hyde_response = response.content.strip()

        # Return both the original query and the hypothetical document
        # The original is included to ensure we don't lose the exact user intent
        return [query, hyde_response]

    def generate_step_back_queries(self, query: str) -> list[str]:
        """Generate step-back questions for broader context retrieval.

        Step-back prompting is inspired by human problem-solving: when faced with
        a specific question, we often first recall general principles or background
        knowledge before tackling the specifics. This method generates broader
        questions that retrieve context helpful for answering the specific query.

        Args:
            query: The original user query text.

        Returns:
            List of step-back questions followed by the original query. The
            list contains up to 3 step-back questions plus the original query
            at the end. Total length is 4 elements (or fewer if LLM returns less).

        Example:
            >>> enhancer = QueryEnhancer(llm)
            >>> queries = enhancer.generate_step_back_queries(
            ...     "What is backpropagation?"
            ... )
            >>> queries
            ['What is machine learning?',
             'How do neural networks learn?',
             'What is gradient descent?',
             'What is backpropagation?']

        Reference:
            Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models
            (Zheng et al., 2023) - https://arxiv.org/abs/2310.06117

        Use Cases:
            - Technical questions requiring background knowledge
            - Questions about specific instances of general concepts
            - Complex queries where context improves answer quality
        """
        # Construct the prompt using the step-back template
        prompt = PromptTemplate(
            template=self.STEP_BACK_TEMPLATE,
            input_variables=["query"],
        )
        formatted_prompt = prompt.format(query=query)

        # Invoke the LLM to generate step-back questions
        response = self.llm.invoke(formatted_prompt)

        # Parse the response: split by newlines, strip whitespace, filter empty lines
        step_back_queries = response.content.strip().split("\n")
        step_back_queries = [q.strip() for q in step_back_queries if q.strip()]

        # Return step-back questions followed by the original query
        # We limit to 3 step-back questions to control retrieval cost
        # The original query is always included last to ensure specific retrieval
        return step_back_queries[:3] + [query]

    def generate_queries(self, query: str, mode: str = "multi_query") -> list[str]:
        """Generate enhanced queries based on the specified mode.

        This is the main entry point for query enhancement. It delegates to
        the appropriate generation method based on the mode parameter.

        Args:
            query: The original user query text.
            mode: Enhancement mode to use. Must be one of:
                - 'multi_query': Generate alternative phrasings (5 queries)
                - 'hyde': Generate hypothetical document (2 queries)
                - 'step_back': Generate broader context questions (4 queries)

        Returns:
            List of enhanced query strings. The structure depends on mode:
                - multi_query: List of alternative phrasings
                - hyde: [original_query, hypothetical_document]
                - step_back: [step_back_q1, step_back_q2, step_back_q3, original_query]

        Raises:
            ValueError: If mode is not one of the recognized values.

        Example:
            >>> enhancer = QueryEnhancer(llm)
            >>> # Multi-query mode
            >>> queries = enhancer.generate_queries("What is AI?", "multi_query")
            >>> len(queries)  # Up to 5 alternative queries
            5
            >>> # HyDE mode
            >>> queries = enhancer.generate_queries("What is AI?", "hyde")
            >>> len(queries)  # Original + hypothetical
            2
            >>> # Step-back mode
            >>> queries = enhancer.generate_queries("What is AI?", "step_back")
            >>> len(queries)  # 3 step-back + original
            4
        """
        # Delegate to the appropriate generation method based on mode
        # This structure makes it easy to add new enhancement strategies
        if mode == "multi_query":
            return self.generate_multi_queries(query)
        if mode == "hyde":
            return self.generate_hyde_queries(query)
        if mode == "step_back":
            return self.generate_step_back_queries(query)

        # Raise an error for unrecognized modes
        # This fail-fast approach catches typos and invalid configurations
        msg = f"Unknown mode: {mode}. Must be 'multi_query', 'hyde', or 'step_back'"
        raise ValueError(msg)
