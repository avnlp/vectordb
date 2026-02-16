"""TriviaQA dataloader for Haystack framework.

This module provides a Haystack-specific dataloader for the TriviaQA dataset,
which contains open-domain trivia questions with evidence documents. The
dataloader uses an LLM generator to summarize multiple answer aliases into
a canonical answer.

Dataset Processing:
    TriviaQA contains trivia questions with multiple answer aliases and
    web search evidence documents. The dataloader uses an LLM to consolidate
    answer aliases into a single canonical answer for evaluation consistency.

Processing Pipeline:
    1. Load dataset from HuggingFace
    2. For each question:
       a. Generate canonical answer from aliases using OpenAIChatGenerator
       b. Extract context passages (web search results) as evidence documents
    3. Format into text/metadata structure
    4. Apply recursive document splitting
    5. Return Haystack Document objects

Answer Summarization:
    Multiple answer aliases (e.g., "Paris", "City of Light", "Capital of France")
    are consolidated into a single canonical answer using the
    SUMMARIZE_ANSWERS_PROMPT template. This ensures consistent evaluation
    against a single ground truth.

Context Handling:
    Each question has multiple context passages (ctxs) from web search with
    IDs, titles, and ranks. These become separate documents with metadata
    linking them to the question and canonical answer.
"""

from typing import Any

from datasets import load_dataset
from haystack import Document
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.preprocessors import RecursiveDocumentSplitter
from haystack.core.component import Component
from haystack.dataclasses import ChatMessage

from vectordb.dataloaders.prompts import SUMMARIZE_ANSWERS_PROMPT
from vectordb.dataloaders.utils import (
    get_logger,
    haystack_docs_to_dict,
    validate_documents,
)


logger = get_logger(__name__)


class TriviaQADataloader:
    """Dataloader for TriviaQA dataset with Haystack integration.

    This dataloader loads the TriviaQA open-domain QA dataset and processes
    it into Haystack Document objects. It uses an LLM generator to consolidate
    multiple answer aliases into a canonical answer.

    The dataloader performs lazy loading with caching to avoid
    reprocessing and redundant LLM calls.

    Attributes:
        dataset: HuggingFace dataset instance
        answer_summary_generator: OpenAIChatGenerator for answer consolidation
        data: Processed QA items with canonical answers
        corpus: List of dicts with text and metadata after splitting
        text_splitter: Haystack component for document chunking
    """

    def __init__(
        self,
        answer_summary_generator: OpenAIChatGenerator,
        dataset_name: str = "trivia_qa",
        split: str = "test",
        text_splitter: Component | None = None,
    ) -> None:
        """Initialize the TriviaQA dataloader.

        Args:
            answer_summary_generator: OpenAIChatGenerator instance for
                consolidating multiple answer aliases into a canonical answer.
            dataset_name: HuggingFace dataset identifier
            split: Dataset split to load (default: "test")
            text_splitter: Optional custom text splitter component.
                Defaults to RecursiveDocumentSplitter if not provided.
        """
        self.dataset = load_dataset(dataset_name, split=split)
        self.answer_summary_generator = answer_summary_generator
        self.data: list[dict[str, Any]] | None = None
        self.text_splitter = text_splitter or RecursiveDocumentSplitter()
        self.corpus: list[dict[str, Any]] = []

    def load_data(self) -> list[dict[str, Any]]:
        """Load and process the TriviaQA dataset.

        Processes the dataset by:
        1. Generating canonical answers from aliases using LLM
        2. Extracting context passages (web search results) as evidence documents
        3. Applying text splitting for optimal chunking

        Returns:
            List of dicts with "text" and "metadata" keys after splitting.
            Each dict represents a chunk of a context passage with full
            question and canonical answer metadata.
        """
        # Lazy loading with caching to avoid redundant LLM API calls
        # Critical for cost management since each row requires an LLM call
        if self.data is None:
            logger.info("Loading and processing dataset.")
            self.data = []
            for row in self.dataset:
                question = row["question"]
                answers = row["answers"]

                # Generate canonical answer from multiple aliases using LLM
                # TriviaQA questions often have many valid answer variations
                # (e.g., "Mount Everest", "Mt. Everest", "Sagarmatha")
                prompt = SUMMARIZE_ANSWERS_PROMPT.format(
                    question=question, answers=answers
                )
                result = self.answer_summary_generator.run(
                    messages=[ChatMessage.from_user(prompt)]
                )
                # Extract text from ChatMessage response object
                summarized_answer = result["replies"][0].text

                # Extract context passages from web search results
                # ctxs contains retrieved web documents with IDs, titles, and ranks
                contexts = row["ctxs"]
                docs = [ctx["text"] for ctx in contexts]
                metadata = [
                    {"id": ctx["id"], "title": ctx["title"]} for ctx in contexts
                ]

                self.data.append(
                    {
                        "question": question,
                        "answers": answers,
                        "answer": summarized_answer,
                        "docs": docs,
                        "metadata": metadata,
                    }
                )
            logger.info(f"Processed {len(self.data)} rows.")

        # Flatten documents and metadata into standard format
        # Each context passage becomes a separate document with full QA metadata
        # This enables retrieval of specific evidence passages for each question
        formatted: list[dict[str, Any]] = []
        for row in self.data:
            for doc, meta in zip(row["docs"], row["metadata"]):
                formatted.append(
                    {
                        "text": doc,
                        "metadata": {
                            "question": row["question"],
                            "answers": row["answers"],
                            "answer": row["answer"],
                            **meta,
                        },
                    }
                )

        # Apply text splitting for chunking long documents
        # Essential for embedding models with limited context windows
        # Also improves retrieval granularity for long web documents
        validate_documents(formatted)
        hs_docs = [Document(content=d["text"], meta=d["metadata"]) for d in formatted]
        split_docs = self.text_splitter.run(hs_docs)["documents"]
        self.corpus = haystack_docs_to_dict(split_docs)

        return self.corpus

    def get_documents(self) -> list[Document]:
        """Convert corpus to Haystack Documents.

        Loads data if not already loaded, then converts the corpus
        to Haystack Document objects for indexing.

        Returns:
            List of Haystack Document objects with content and metadata.

        Raises:
            ValueError: If corpus is empty (load_data() not called).
        """
        if not self.corpus:
            raise ValueError("Corpus empty. Call load_data() first.")

        return [Document(content=d["text"], meta=d["metadata"]) for d in self.corpus]
