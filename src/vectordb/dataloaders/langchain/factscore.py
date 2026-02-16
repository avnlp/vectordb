"""FactScore dataloader for LangChain framework.

This module provides a LangChain-specific dataloader for the FactScore dataset,
which focuses on atomic fact verification. The dataloader uses a ChatGroq LLM
to summarize multiple answer aliases into a canonical answer.

Dataset Processing:
    FactScore contains questions with multiple answer aliases and context
    passages. The dataloader uses an LLM to consolidate answer aliases into
    a single canonical answer for evaluation consistency.

Processing Pipeline:
    1. Load dataset from HuggingFace
    2. For each question:
       a. Generate canonical answer from aliases using ChatGroq
       b. Extract context passages as evidence documents
       c. Extract fact-specific metadata (entity, topic, facts, decomposed_facts)
    3. Format into text/metadata structure
    4. Apply recursive character text splitting
    5. Return LangChain Document objects

Answer Summarization:
    Multiple answer aliases are consolidated into a single canonical answer
    using the SUMMARIZE_ANSWERS_PROMPT template via ChatGroq.

Fact Metadata:
    The dataset includes rich fact metadata:
    - entity: The subject entity
    - topic: Topic classification
    - facts: High-level summary facts
    - decomposed_facts: Granular atomic facts

Framework Differences from Haystack:
    - Uses ChatGroq instead of OpenAIChatGenerator
    - Uses HumanMessage for prompt formatting
    - Uses RecursiveCharacterTextSplitter
    - Returns langchain_core.documents.Document objects
"""

from typing import Any

from datasets import load_dataset
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import TextSplitter

from vectordb.dataloaders.prompts import SUMMARIZE_ANSWERS_PROMPT
from vectordb.dataloaders.utils import (
    get_logger,
    langchain_docs_to_dict,
    validate_documents,
)


logger = get_logger(__name__)


class FactScoreDataloader:
    """Dataloader for FactScore dataset with LangChain integration.

    This dataloader loads the FactScore fact verification dataset and
    processes it into LangChain Document objects. It uses a ChatGroq LLM
    to consolidate multiple answer aliases into a canonical answer.

    The dataloader performs lazy loading with caching to avoid
    reprocessing and redundant LLM calls.

    Attributes:
        dataset: HuggingFace dataset instance
        answer_summary_generator: ChatGroq for answer consolidation
        data: Processed QA items with canonical answers
        corpus: List of dicts with text and metadata after splitting
        text_splitter: LangChain text splitter for document chunking
    """

    def __init__(
        self,
        answer_summary_generator: ChatGroq,
        dataset_name: str = "dskar/FActScore",
        split: str = "test",
        text_splitter: TextSplitter | None = None,
    ) -> None:
        """Initialize the FactScore dataloader.

        Args:
            answer_summary_generator: ChatGroq instance for consolidating
                multiple answer aliases into a canonical answer.
            dataset_name: HuggingFace dataset identifier
            split: Dataset split to load (default: "test")
            text_splitter: Optional custom text splitter.
                Defaults to RecursiveCharacterTextSplitter if not provided.
        """
        self.dataset = load_dataset(dataset_name, split=split)
        self.answer_summary_generator = answer_summary_generator
        self.data: list[dict[str, Any]] | None = None
        self.text_splitter = text_splitter or RecursiveCharacterTextSplitter()
        self.corpus: list[dict[str, Any]] = []

    def load_data(self) -> list[dict[str, Any]]:
        """Load and process the FactScore dataset.

        Processes the dataset by:
        1. Generating canonical answers from aliases using ChatGroq
        2. Extracting context passages as evidence documents
        3. Extracting fact-specific metadata (entity, topic, facts)
        4. Applying text splitting for optimal chunking

        Returns:
            List of dicts with "text" and "metadata" keys after splitting.
            Each dict represents a chunk of a context passage with full
            question, entity, topic, and canonical answer metadata.
        """
        if self.data is None:
            logger.info("Loading and processing dataset.")
            self.data = []
            for row in self.dataset:
                question = row["question"]
                answers = row["answers"]

                prompt = SUMMARIZE_ANSWERS_PROMPT.format(
                    question=question, answers=answers
                )
                response = self.answer_summary_generator.invoke(
                    [HumanMessage(content=prompt)]
                )
                summarized_answer = str(response.content)

                contexts = row["ctxs"]
                docs = [ctx["text"] for ctx in contexts]
                metadata = [
                    {"id": ctx["id"], "title": ctx["title"]} for ctx in contexts
                ]

                entity = row.get("entity", "")
                topic = row.get("topic", entity)
                fact_id = row.get("id")
                facts = row.get("facts", [])
                decomposed_facts = row.get("decomposed_facts", [])

                self.data.append(
                    {
                        "question": question,
                        "answers": answers,
                        "answer": summarized_answer,
                        "docs": docs,
                        "metadata": metadata,
                        "entity": entity,
                        "topic": topic,
                        "id": fact_id,
                        "facts": facts,
                        "decomposed_facts": decomposed_facts,
                    }
                )
            logger.info(f"Processed {len(self.data)} rows.")

        # Flatten documents and metadata into standard format
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
                            "entity": row.get("entity", ""),
                            "topic": row.get("topic", ""),
                            "id": row.get("id"),
                            "facts": row.get("facts", []),
                            "decomposed_facts": row.get("decomposed_facts", []),
                            **meta,
                        },
                    }
                )

        # Apply text splitting for chunking long documents
        validate_documents(formatted)
        lc_docs = [
            Document(page_content=d["text"], metadata=d["metadata"]) for d in formatted
        ]
        split_docs = self.text_splitter.transform_documents(lc_docs)
        self.corpus = langchain_docs_to_dict(split_docs)

        return self.corpus

    def get_documents(self) -> list[Document]:
        """Convert corpus to LangChain Documents.

        Loads data if not already loaded, then converts the corpus
        to LangChain Document objects for indexing.

        Returns:
            List of LangChain Document objects with page_content and metadata.

        Raises:
            ValueError: If corpus is empty (load_data() not called).
        """
        if not self.corpus:
            raise ValueError("Corpus empty. Call load_data() first.")

        return [
            Document(page_content=d["text"], metadata=d["metadata"])
            for d in self.corpus
        ]
