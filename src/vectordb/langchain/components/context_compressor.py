"""Context compression component for LangChain pipelines.

This module provides context compression capabilities for RAG applications,
addressing the token limit constraints of LLMs when processing large retrieved
contexts. By compressing retrieved documents, we can fit more relevant information
within the context window while maintaining answer quality.

Context Compression Strategies:
    1. Reranking: Uses cross-encoder models to score document relevance and
       filter to the most pertinent documents. This is a "selective" compression
       approach that preserves full documents but reduces quantity.

    2. LLM Extraction: Uses an LLM to extract and synthesize only the relevant
       passages from documents. This is a "generative" compression approach that
       can significantly reduce token count but may lose some information.

When to Use Each Mode:
    - Reranking: Best when documents are already concise and you want to preserve
      the original text verbatim. Useful for legal/medical domains where exact
      wording matters. Also faster as it avoids LLM calls during compression.

    - LLM Extraction: Best when documents are verbose and contain significant
      irrelevant content. Useful for web-scraped content or long articles where
      only specific passages answer the query.

Integration with LangChain:
    The component integrates with LangChain's Document abstraction and uses
    HuggingFaceCrossEncoder for reranking and ChatGroq for LLM-based extraction.
    This allows seamless integration with existing LangChain retrieval chains.

Performance Considerations:
    - Reranking requires loading a cross-encoder model into memory.
      Common choices are BAAI/bge-reranker-base (lightweight)
      or BAAI/bge-reranker-large (better quality).

    - LLM extraction adds latency due to the additional LLM call. Consider caching
      extraction results for frequently-asked queries.

Usage:
    >>> from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    >>> from vectordb.langchain.components import ContextCompressor
    >>> # Reranking mode
    >>> reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    >>> compressor = ContextCompressor(mode="reranking", reranker=reranker)
    >>> compressed = compressor.compress(query, documents, top_k=3)
    >>> # LLM extraction mode
    >>> from langchain_groq import ChatGroq
    >>> llm = ChatGroq(model="llama-3.3-70b-versatile")
    >>> compressor = ContextCompressor(mode="llm_extraction", llm=llm)
    >>> compressed = compressor.compress(query, documents)
"""

import logging

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq


# Module-level logger for compression operations and debugging
logger = logging.getLogger(__name__)


class ContextCompressor:
    """Compress retrieved context using reranking or LLM-based extraction.

    This component addresses the context window limitations of LLMs by reducing
    the token count of retrieved documents before passing them to the generation
    phase. It supports two complementary strategies: reranking (selective) and
    LLM extraction (generative).

    Attributes:
        mode: The compression strategy - 'reranking' or 'llm_extraction'.
        llm: LangChain LLM instance for extraction mode.
        reranker: HuggingFace cross-encoder model for reranking mode.
        EXTRACTION_TEMPLATE: Prompt template for LLM extraction mode.

    Design Pattern:
        The class follows the Strategy pattern, allowing the compression algorithm
        to be selected at runtime. Both strategies implement the same interface
        (compress method), making them interchangeable in pipelines.

    Token Optimization:
        Compression ratios vary by content type:
        - Reranking: Reduces document count (e.g., 10 -> 3 docs), ~70% reduction
        - LLM extraction: Reduces content length per doc, ~50-80% token reduction
        Combined approaches can achieve 90%+ token savings while preserving relevance.
    """

    EXTRACTION_TEMPLATE = """Given the following documents, extract the most relevant passages that directly answer the question. Return only the extracted passages, without any additional text.

Question: {query}

Documents:
{documents}

Relevant passages:"""

    def __init__(
        self,
        mode: str = "reranking",
        llm: ChatGroq | None = None,
        reranker: HuggingFaceCrossEncoder | None = None,
    ) -> None:
        """Initialize ContextCompressor with the specified compression strategy.

        Args:
            mode: Compression mode to use. Must be either:
                - 'reranking': Use cross-encoder to score and filter documents
                - 'llm_extraction': Use LLM to extract relevant passages
            llm: ChatGroq instance for LLM extraction mode. Required when mode
                is 'llm_extraction'. Not used for reranking mode.
            reranker: HuggingFaceCrossEncoder instance for reranking mode. Required
                when mode is 'reranking'. Not used for extraction mode.

        Raises:
            ValueError: If mode is not 'reranking' or 'llm_extraction', or if
                required dependencies (llm/reranker) are not provided for the
                selected mode.

        Example:
            >>> # Reranking setup
            >>> reranker = HuggingFaceCrossEncoder("BAAI/bge-reranker-base")
            >>> compressor = ContextCompressor(mode="reranking", reranker=reranker)
            >>> # LLM extraction setup
            >>> llm = ChatGroq(model="llama-3.3-70b-versatile")
            >>> compressor = ContextCompressor(mode="llm_extraction", llm=llm)
        """
        if mode not in ("reranking", "llm_extraction"):
            msg = f"Invalid mode: {mode}. Must be 'reranking' or 'llm_extraction'"
            raise ValueError(msg)

        self.mode = mode
        self.llm = llm
        self.reranker = reranker

        # Validate that required dependencies are provided for the selected mode.
        # This fail-fast approach catches configuration errors at initialization
        # rather than at compression time.
        if mode == "llm_extraction" and llm is None:
            msg = "LLM required for 'llm_extraction' mode"
            raise ValueError(msg)

        if mode == "reranking" and reranker is None:
            msg = "Reranker required for 'reranking' mode"
            raise ValueError(msg)

    def compress_reranking(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[Document]:
        """Compress documents using cross-encoder reranking.

        This method scores each document's relevance to the query using a
        cross-encoder model, then returns only the top_k most relevant documents.
        The cross-encoder considers the query-document pair jointly, providing
        more accurate relevance scores than embedding-based similarity.

        Args:
            query: The user's query text. Used to score document relevance.
            documents: List of LangChain Document objects to compress.
            top_k: Number of top-scoring documents to return. Default is 5.
                Set higher for broader coverage, lower for more aggressive compression.

        Returns:
            List of Document objects, sorted by relevance score (highest first),
            containing at most top_k documents. Returns empty list if input
            documents is empty.

        Algorithm:
            1. Create query-document pairs for scoring
            2. Score pairs using cross-encoder (higher = more relevant)
            3. Sort documents by score in descending order
            4. Return top_k documents

        Note:
            Cross-encoders are more accurate than bi-encoders for relevance scoring
            but are slower (O(n) forward passes vs O(1) for bi-encoder).
            Consider batching for large document sets.
        """
        # Handle edge case: empty document list
        if not documents or not self.reranker:
            return documents

        # Create query-document pairs for cross-encoder scoring.
        # Each pair is [query, document_text] for the model to evaluate jointly.
        pairs = [[query, doc.page_content] for doc in documents]

        # Score all pairs using the cross-encoder.
        # Scores are typically logits or probabilities indicating relevance.
        scores = self.reranker.rank(pairs)

        # Sort documents by score in descending order (highest relevance first).
        # zip pairs documents with their scores for joint sorting.
        sorted_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Return only the top_k documents, discarding scores.
        return [doc for doc, _ in sorted_docs[:top_k]]

    def compress_llm_extraction(
        self,
        query: str,
        documents: list[Document],
    ) -> list[Document]:
        """Compress documents using LLM-based passage extraction.

        This method uses an LLM to read through all documents and extract only
        the passages that are relevant to answering the query. Unlike reranking,
        which preserves full documents, this approach can extract specific
        sentences or paragraphs, achieving higher compression ratios.

        Args:
            query: The user's query text. Guides what content to extract.
            documents: List of LangChain Document objects to compress.

        Returns:
            List containing a single Document with the extracted passages.
            The document metadata includes:
                - 'source': 'compressed' to indicate processing
                - 'original_doc_count': Number of input documents

        Trade-offs:
            - Higher compression ratio than reranking (can extract just sentences)
            - Loses document boundaries (all passages merged into one doc)
            - Adds LLM latency (one LLM call per compression)
            - May hallucinate or miss relevant content (LLM-dependent)

        Note:
            The extraction prompt instructs the LLM to return only relevant
            passages without additional commentary. Prompt engineering can
            significantly affect extraction quality.
        """
        # Handle edge case: empty document list
        if not documents or not self.llm:
            return documents

        # Format documents with clear separators and numbering for the LLM.
        # This helps the LLM distinguish between different source documents.
        doc_texts = "\n\n".join(
            [
                f"Document {i + 1}:\n{doc.page_content}"
                for i, doc in enumerate(documents)
            ]
        )

        # Construct the extraction prompt using LangChain's template system.
        prompt = PromptTemplate(
            template=self.EXTRACTION_TEMPLATE,
            input_variables=["query", "documents"],
        )
        formatted_prompt = prompt.format(query=query, documents=doc_texts)

        # Invoke the LLM to extract relevant passages.
        response = self.llm.invoke(formatted_prompt)
        extracted_text = response.content.strip()

        # Create a new Document with the extracted content.
        # Metadata preserves provenance information for debugging.
        compressed_doc = Document(
            page_content=extracted_text,
            metadata={
                "source": "compressed",
                "original_doc_count": len(documents),
            },
        )

        return [compressed_doc]

    def compress(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[Document]:
        """Compress documents using the configured compression strategy.

        This is the main entry point for context compression. It delegates to
        the appropriate compression method based on the mode set during
        initialization.

        Args:
            query: The user's query text. Used to determine relevance.
            documents: List of LangChain Document objects to compress.
            top_k: Number of documents to return (only used in reranking mode).
                Ignored for llm_extraction mode.

        Returns:
            Compressed list of documents. Structure depends on mode:
                - reranking: List of top_k Document objects, sorted by relevance
                - llm_extraction: List containing single synthesized Document

        Raises:
            ValueError: If the compression mode is not recognized. This should
                not occur if the class was initialized correctly.

        Example:
            >>> compressor = ContextCompressor(mode="reranking", reranker=reranker)
            >>> docs = [Document(page_content="..."), ...]
            >>> compressed = compressor.compress("What is AI?", docs, top_k=3)
            >>> len(compressed)  # At most 3 documents
            3
        """
        # Delegate to the appropriate compression strategy based on mode.
        # This if-else structure makes it easy to add new compression modes.
        if self.mode == "reranking":
            return self.compress_reranking(query, documents, top_k)
        if self.mode == "llm_extraction":
            return self.compress_llm_extraction(query, documents)

        # This should never happen if __init__ validation is correct,
        # but we include it for defensive programming.
        msg = f"Unknown mode: {self.mode}"
        raise ValueError(msg)
