"""ARC dataloader for LangChain framework.

This module provides a LangChain-specific dataloader for the ARC (AI2 Reasoning
Challenge) dataset. It extends the base loading functionality with LangChain
Document type integration and text splitting capabilities.

Dataset Processing:
    The ARC dataset contains science questions with multiple-choice answers.
    Each question includes context passages (ctxs) that serve as evidence
    documents for retrieval-based QA.

Processing Pipeline:
    1. Load dataset from HuggingFace
    2. Format questions with choice labels (A, B, C, D)
    3. Extract correct answer from choices
    4. Process context passages as evidence documents
    5. Apply recursive character text splitting for chunking
    6. Return LangChain Document objects

Answer Extraction:
    The correct answer is determined by matching the answerKey (e.g., "A")
    with the corresponding choice label and extracting the choice text.

Context Handling:
    Each question has multiple context passages (ctxs) with IDs and titles.
    These become separate documents with metadata linking them to the
    parent question.

Framework Differences from Haystack:
    - Uses RecursiveCharacterTextSplitter instead of RecursiveDocumentSplitter
    - Returns langchain_core.documents.Document objects
    - Uses transform_documents() method for splitting
"""

from typing import Any

from datasets import load_dataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import TextSplitter

from vectordb.dataloaders.utils import (
    get_logger,
    langchain_docs_to_dict,
    validate_documents,
)


logger = get_logger(__name__)


class ARCDataloader:
    """Dataloader for ARC dataset with LangChain integration.

    This dataloader loads the ARC science QA dataset and processes it into
    LangChain Document objects suitable for indexing and retrieval. It handles
    the multiple-choice format and extracts context passages as evidence.

    The dataloader performs lazy loading - data is only loaded when load_data()
    or get_documents() is called. Results are cached to avoid reprocessing.

    Attributes:
        dataset: HuggingFace dataset instance
        data: Processed QA items with questions, answers, and context
        corpus: List of dicts with text and metadata after splitting
        text_splitter: LangChain text splitter for document chunking
    """

    def __init__(
        self,
        dataset_name: str = "ai2_arc",
        split: str = "test",
        text_splitter: TextSplitter | None = None,
    ) -> None:
        """Initialize the ARC dataloader.

        Args:
            dataset_name: HuggingFace dataset identifier (default: "ai2_arc")
            split: Dataset split to load (default: "test")
            text_splitter: Optional custom text splitter.
                Defaults to RecursiveCharacterTextSplitter if not provided.
        """
        self.dataset = load_dataset(dataset_name, split=split)
        self.data: list[dict[str, Any]] | None = None
        self.text_splitter = text_splitter or RecursiveCharacterTextSplitter()
        self.corpus: list[dict[str, Any]] = []

    def load_data(self) -> list[dict[str, Any]]:
        """Load and process the ARC dataset.

        Processes the dataset by:
        1. Formatting questions with multiple-choice options
        2. Extracting the correct answer from choices
        3. Collecting context passages as evidence documents
        4. Applying text splitting for optimal chunking

        Returns:
            List of dicts with "text" and "metadata" keys after splitting.
            Each dict represents a chunk of a context passage with full
            question metadata attached.
        """
        # Lazy loading: only process if not already cached
        if self.data is None:
            logger.info("Loading and processing dataset.")
            self.data = []
            for row in self.dataset:
                question = row["question"]
                choices = row["choices"]
                answerkey = row["answerKey"]

                formatted_question = question
                for label, text in zip(choices["label"], choices["text"]):
                    formatted_question += f"\n{label}: {text}"

                answer = next(
                    text
                    for label, text in zip(choices["label"], choices["text"])
                    if label == answerkey
                )

                contexts = row["ctxs"]
                docs = [ctx["text"] for ctx in contexts]
                metadata = [
                    {"id": ctx["id"], "title": ctx["title"]} for ctx in contexts
                ]

                self.data.append(
                    {
                        "question": formatted_question,
                        "choices": choices,
                        "answer": answer,
                        "answerKey": answerkey,
                        "docs": docs,
                        "metadata": metadata,
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
                            "choices": row["choices"],
                            "answer": row["answer"],
                            "answerKey": row["answerKey"],
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
