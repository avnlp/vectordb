"""Earnings Call dataloader for LangChain framework.

This module provides a LangChain-specific dataloader for the Earnings Calls
dataset. It handles the dual-structure of the dataset: QA pairs for queries
and a separate corpus for document indexing.

Dataset Structure:
    The Earnings Calls dataset has two components:
    1. QA Dataset: Question-answer pairs with transcript references
    2. Corpus Dataset: Full transcripts organized by company and quarter

Processing Pipeline:
    1. Load QA dataset for query extraction
    2. Load corpus dataset for document indexing
    3. Parse quarter format (YYYY-QN) into year/quarter components
    4. Format transcript segments with speaker attribution
    5. Apply recursive character text splitting
    6. Return LangChain Document objects

Temporal Metadata:
    The dataset includes temporal information (year, quarter, date) which
    is preserved in metadata for time-sensitive queries.

Speaker Attribution:
    Each transcript segment includes speaker information, which is
    preserved in document metadata for speaker-aware retrieval.

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

# Format string for quarter designation (e.g., "Q1", "Q4")
QUARTER_FORMAT = "Q{quarter}"


class EarningsCallDataloader:
    """Dataloader for Earnings Call dataset with LangChain integration.

    This dataloader handles the dual-structure of the Earnings Calls dataset,
    providing access to both QA pairs (for queries) and transcript corpus
    (for indexing). It processes financial transcripts with temporal and
    entity metadata.

    The dataloader separates QA data loading from corpus loading to support
    different use cases: extracting queries vs. building document indexes.

    Attributes:
        dataset: QA dataset with question-answer pairs
        corpus_dataset: Corpus dataset with full transcripts
        data: Processed QA items
        corpus: Processed and split transcript documents
        text_splitter: LangChain text splitter for document chunking
    """

    def __init__(
        self,
        dataset_name: str = "lamini/earnings-calls-qa",
        split: str = "train",
        text_splitter: TextSplitter | None = None,
    ) -> None:
        """Initialize the Earnings Call dataloader.

        Args:
            dataset_name: HuggingFace dataset identifier
            split: Dataset split to load (default: "train")
            text_splitter: Optional custom text splitter.
                Defaults to RecursiveCharacterTextSplitter if not provided.
        """
        self.dataset = load_dataset(dataset_name, split=split)
        self.corpus_dataset = load_dataset(dataset_name, "corpus", split=split)
        self.data: list[dict[str, Any]] | None = None
        self.corpus: list[dict[str, Any]] = []
        self.text_splitter = text_splitter or RecursiveCharacterTextSplitter()

    def load_data(self) -> list[dict[str, Any]]:
        """Load and process the Earnings Call QA dataset.

        Processes QA pairs by extracting temporal information from the
        quarter field (format: "YYYY-QN") and structuring the data for
        query extraction.

        Returns:
            List of dicts with structure:
            {
                "text": "Transcript content...",
                "metadata": {
                    "question": "Financial question",
                    "answer": "Answer from transcript",
                    "date": "YYYY-MM-DD",
                    "year": "2023",
                    "quarter": "Q4",
                    "ticker": "AAPL",
                    "year-quarter": "2023-Q4"
                }
            }
        """
        if self.data is None:
            logger.info("Loading and processing dataset.")
            self.data = []
            for row in self.dataset:
                year, quarter = row["q"].split("-")
                self.data.append(
                    {
                        "text": row["transcript"],
                        "metadata": {
                            "question": row["question"],
                            "answer": row["answer"],
                            "date": row["date"],
                            "year": year,
                            "quarter": quarter,
                            "ticker": row["ticker"],
                            "year-quarter": row["q"],
                        },
                    }
                )
            logger.info(f"Processed {len(self.data)} rows.")

        return self.data

    def get_documents(self) -> list[Document]:
        """Load corpus and convert to LangChain Documents.

        Loads the transcript corpus, formats segments with speaker
        attribution, applies text splitting, and returns LangChain
        Document objects.

        The corpus is loaded lazily and cached to avoid reprocessing.

        Returns:
            List of LangChain Document objects with transcript content
            and metadata including ticker, year, quarter, date, and speaker.
        """
        if not self.corpus:
            logger.info("Loading corpus dataset.")
            for row in self.corpus_dataset:
                ticker = row["symbol"]
                year = row["year"]
                quarter = QUARTER_FORMAT.format(quarter=row["quarter"])
                date = row["date"]
                for doc in row["transcript"]:
                    self.corpus.append(
                        {
                            "text": f"{doc['speaker']}: {doc['text']}",
                            "metadata": {
                                "ticker": ticker,
                                "year": year,
                                "quarter": quarter,
                                "date": date,
                                "speaker": doc["speaker"],
                            },
                        }
                    )
            logger.info(f"Processed {len(self.corpus)} corpus entries.")

        # Validate and split documents for optimal chunking
        validate_documents(self.corpus)
        lc_docs = [
            Document(page_content=d["text"], metadata=d["metadata"])
            for d in self.corpus
        ]
        split_docs = self.text_splitter.transform_documents(lc_docs)
        final_corpus = langchain_docs_to_dict(split_docs)

        return [
            Document(page_content=d["text"], metadata=d["metadata"])
            for d in final_corpus
        ]
