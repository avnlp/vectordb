"""FactScore dataloader for Haystack framework.

This module provides a Haystack-specific dataloader for the FactScore dataset,
which focuses on atomic fact verification. The dataloader uses an LLM generator
to consolidate multiple possible answers into a canonical form for consistent
evaluation.

Dataset Processing:
    FactScore contains entity-based questions with multiple possible answers
    and context passages from Wikipedia. The dataloader uses an LLM to
    consolidate answers into a single canonical form for evaluation consistency.

Processing Pipeline:
    1. Load dataset from HuggingFace
    2. For each entity/question:
       a. Generate canonical answer from multiple options using OpenAIChatGenerator
       b. Extract context passages as evidence documents
       c. Preserve fact metadata (entity, topic, atomic facts)
    3. Format into text/metadata structure
    4. Apply recursive document splitting
    5. Return Haystack Document objects

Answer Consolidation:
    Multiple possible answers are consolidated into a single canonical answer
    using the SUMMARIZE_ANSWERS_PROMPT template. This ensures consistent
    evaluation against a single ground truth.

Fact Metadata:
    The dataset includes rich metadata for fact verification:
    - entity: The subject entity being discussed
    - topic: Topic classification (often same as entity)
    - facts: High-level summary facts
    - decomposed_facts: Granular atomic facts for fine-grained verification

Context Handling:
    Each entity has multiple context passages (ctxs) with IDs and titles.
    These become separate documents with metadata linking them to the
    entity and canonical answer.
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


class FactScoreDataloader:
    """Dataloader for FactScore dataset with Haystack integration.

    This dataloader loads the FactScore fact verification dataset and
    processes it into Haystack Document objects. It uses an LLM generator
    to consolidate multiple possible answers into a canonical form.

    The dataloader performs lazy loading with caching to avoid
    reprocessing and redundant LLM calls.

    Attributes:
        dataset: HuggingFace dataset instance
        answer_summary_generator: OpenAIChatGenerator for answer consolidation
        data: Processed items with canonical answers and fact metadata
        corpus: List of dicts with text and metadata after splitting
        text_splitter: Haystack component for document chunking
    """

    def __init__(
        self,
        answer_summary_generator: OpenAIChatGenerator,
        dataset_name: str = "dskar/FActScore",
        split: str = "test",
        text_splitter: Component | None = None,
    ) -> None:
        """Initialize the FactScore dataloader.

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
        """Load and process the FactScore dataset.

        Processes the dataset by:
        1. Generating canonical answers from aliases using LLM
        2. Extracting context passages as evidence documents
        3. Applying text splitting for optimal chunking

        Returns:
            List of dicts with "text" and "metadata" keys after splitting.
            Each dict represents a chunk of a context passage with full
            question and canonical answer metadata.
        """
        # Lazy loading with caching to avoid redundant LLM calls
        # This is critical since each row requires an LLM API call
        if self.data is None:
            logger.info("Loading and processing dataset.")
            self.data = []
            for row in self.dataset:
                question = row["question"]
                answers = row["answers"]

                # Generate canonical answer from multiple aliases using LLM
                # This ensures consistent evaluation against a single ground truth
                prompt = SUMMARIZE_ANSWERS_PROMPT.format(
                    question=question, answers=answers
                )
                result = self.answer_summary_generator.run(
                    messages=[ChatMessage.from_user(prompt)]
                )
                # Extract text from the first reply (ChatMessage object)
                summarized_answer = result["replies"][0].text

                # Extract context passages from Wikipedia as evidence documents
                # ctxs contains retrieved passages with IDs and titles
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
        # Ensures documents fit within embedding model context windows
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
