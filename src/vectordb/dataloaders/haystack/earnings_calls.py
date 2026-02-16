"""Earnings Call dataloader for Haystack framework.

This module provides a Haystack-specific dataloader for the Earnings Calls
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
    5. Apply recursive document splitting
    6. Return Haystack Document objects

Temporal Metadata:
    The dataset includes temporal information (year, quarter, date) which
    is preserved in metadata for time-sensitive queries.

Speaker Attribution:
    Each transcript segment includes speaker information, which is
    preserved in document metadata for speaker-aware retrieval.
"""

from typing import Any

from datasets import load_dataset
from haystack import Document
from haystack.components.preprocessors import RecursiveDocumentSplitter
from haystack.core.component import Component

from vectordb.dataloaders.utils import (
    get_logger,
    haystack_docs_to_dict,
    validate_documents,
)


logger = get_logger(__name__)

# Format string for quarter designation (e.g., "Q1", "Q4")
# Used to normalize quarter numbers from the dataset into standard format
QUARTER_FORMAT = "Q{quarter}"


class EarningsCallDataloader:
    """Dataloader for Earnings Call dataset with Haystack integration.

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
        text_splitter: Haystack component for document chunking
    """

    def __init__(
        self,
        dataset_name: str = "lamini/earnings-calls-qa",
        split: str = "train",
        text_splitter: Component | None = None,
    ) -> None:
        """Initialize the Earnings Call dataloader.

        Args:
            dataset_name: HuggingFace dataset identifier
            split: Dataset split to load (default: "train")
            text_splitter: Optional custom text splitter component.
                Defaults to RecursiveDocumentSplitter if not provided.
        """
        # Load both QA and corpus datasets at initialization
        # The corpus is loaded as a separate dataset configuration
        self.dataset = load_dataset(dataset_name, split=split)
        self.corpus_dataset = load_dataset(dataset_name, "corpus", split=split)
        self.data: list[dict[str, Any]] | None = None
        self.corpus: list[dict[str, Any]] = []
        self.text_splitter = text_splitter or RecursiveDocumentSplitter()

    def load_data(self) -> list[dict[str, Any]]:
        """Load and process the Earnings Call QA dataset.

        Processes QA pairs by extracting temporal information from the
        quarter field (format: "YYYY-QN") and structuring the data in
        standardized text/metadata format.

        Returns:
            List of dicts with standardized structure:
            {
                "text": "Transcript text",
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
        # Lazy loading: only process if not already cached
        if self.data is None:
            logger.info("Loading and processing dataset.")
            raw_data = []
            for row in self.dataset:
                # Parse quarter format "YYYY-QN" into separate year and quarter
                # This enables temporal filtering and aggregation queries
                year, quarter = row["q"].split("-")
                raw_data.append(
                    {
                        "question": row["question"],
                        "answer": row["answer"],
                        "date": row["date"],
                        "context": row["transcript"],
                        "year": year,
                        "quarter": quarter,
                        "ticker": row["ticker"],
                        "year-quarter": row["q"],
                    }
                )
            logger.info(f"Processed {len(raw_data)} rows.")

            # Format into standardized text/metadata structure
            # This ensures compliance with DataloaderProtocol used by factory.py
            self.data = [
                {
                    "text": row["context"],
                    "metadata": {
                        "question": row["question"],
                        "answer": row["answer"],
                        "date": row["date"],
                        "year": row["year"],
                        "quarter": row["quarter"],
                        "ticker": row["ticker"],
                        "year-quarter": row["year-quarter"],
                    },
                }
                for row in raw_data
            ]

        return self.data

    def get_documents(self) -> list[Document]:
        """Load corpus and convert to Haystack Documents.

        Loads the transcript corpus, formats segments with speaker
        attribution, applies text splitting, and returns Haystack
        Document objects.

        The corpus is loaded lazily and cached to avoid reprocessing.

        Returns:
            List of Haystack Document objects with transcript content
            and metadata including ticker, year, quarter, date, and speaker.
        """
        # Load corpus only when needed (lazy loading)
        # This separates QA extraction from document indexing workflows
        if not self.corpus:
            logger.info("Loading corpus dataset.")
            for row in self.corpus_dataset:
                ticker = row["symbol"]
                year = row["year"]
                # Normalize quarter number to "QX" format using format string
                quarter = QUARTER_FORMAT.format(quarter=row["quarter"])
                date = row["date"]
                # Each transcript contains multiple speaker segments
                # Flatten these into individual documents with speaker metadata
                for doc in row["transcript"]:
                    self.corpus.append(
                        {
                            # Prefix text with speaker for context-aware chunking
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
        # Validation ensures all documents have required fields before processing
        validate_documents(self.corpus)
        hs_docs = [Document(content=d["text"], meta=d["metadata"]) for d in self.corpus]
        split_docs = self.text_splitter.run(hs_docs)["documents"]
        final_corpus = haystack_docs_to_dict(split_docs)

        return [Document(content=d["text"], meta=d["metadata"]) for d in final_corpus]
