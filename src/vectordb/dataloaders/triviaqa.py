"""TriviaQA dataset loader for open-domain question answering.

This module provides a specialized dataloader for the TriviaQA dataset, which
consists of trivia questions paired with evidence documents from web search.

Dataset Structure:
    - question: The trivia question
    - answer: The answer (can be multiple aliases)
    - search_results: Dict containing evidence documents with:
        - search_context: Full text passages
        - title: Document titles
        - description: Snippets/descriptions
        - rank: Search result ranking

Design Notes:
    TriviaQA uses a streaming approach to handle large datasets efficiently.
    Each question may have multiple evidence documents, creating multiple
    indexable documents per question. The evidence documents are ranked
    by search relevance.

    The answer field may contain multiple aliases (e.g., "Paris",
    "City of Light", "Capital of France"), requiring normalization
    for evaluation purposes.

Data Source:
    HuggingFace dataset: trivia_qa (configs: rc, unfiltered)
    The "rc" (reading comprehension) config provides evidence documents
    for each question.

Usage:
    >>> from vectordb.dataloaders import TriviaQADataloader
    >>> loader = TriviaQADataloader(dataset_name="trivia_qa", config="rc")
    >>> data = loader.load(limit=100)
    >>> # Returns list of {"text": "...", "metadata": {...}}
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import load_dataset as hf_load_dataset


logger = logging.getLogger(__name__)


class TriviaQADataloader:
    """Loader for TriviaQA dataset.

    This dataloader handles the TriviaQA format where each question has
    multiple evidence documents from web search. Each evidence document
    becomes a separate indexable item with shared question/answer metadata.

    The loader handles the nested structure of search_results, extracting
    parallel arrays of titles, contexts, and ranks. Bounds checking ensures
    robust handling of variable-length result lists.

    Attributes:
        dataset_name: HuggingFace dataset identifier
        config: Dataset configuration ("rc" for reading comprehension)
        split: Dataset split to load
        limit: Maximum number of items to load
    """

    def __init__(
        self,
        dataset_name: str = "trivia_qa",
        config: str = "rc",
        split: str = "test",
        limit: int | None = None,
    ) -> None:
        """Initialize TriviaQA dataloader.

        Args:
            dataset_name: HuggingFace dataset identifier
            config: Dataset configuration (default: "rc")
            split: Dataset split to load (default: "test")
            limit: Maximum number of items to load (None for all)
        """
        self.dataset_name = dataset_name
        self.config = config
        self.split = split
        self.limit = limit

    def load(self) -> list[dict[str, Any]]:
        """Load TriviaQA dataset and return standardized format.

        Processes each question and its evidence documents. Each evidence
        document becomes a separate item with the question and answer
        preserved in metadata. This allows retrieval systems to find
        any relevant evidence document for a given question.

        The method uses bounds checking when accessing search result arrays
        to handle cases where different result fields have varying lengths.

        Returns:
            List of standardized dicts with structure:
            {
                "text": "Evidence document text...",
                "metadata": {
                    "question": "Trivia question",
                    "answer": ["Answer alias 1", "Answer alias 2", ...],
                    "rank": 1,  # Search result rank
                    "title": "Document title"
                }
            }
        """
        dataset = hf_load_dataset(
            self.dataset_name, self.config, split=self.split, streaming=True
        )
        result: list[dict[str, Any]] = []

        for row in dataset:
            question = row["question"]
            answer = row["answer"]

            search_results = row["search_results"]
            num_results = len(search_results.get("title", []))

            for i in range(num_results):
                # Prefer search_context over description when available
                search_context = search_results.get("search_context", [])
                description = search_results.get("description", [])
                text_content = ""
                if i < len(search_context):
                    text_content = search_context[i]
                elif i < len(description):
                    text_content = description[i]

                result.append(
                    {
                        "text": text_content,
                        "metadata": {
                            "question": question,
                            "answer": answer,
                            "rank": search_results["rank"][i]
                            if i < len(search_results.get("rank", []))
                            else None,
                            "title": search_results["title"][i]
                            if i < len(search_results.get("title", []))
                            else None,
                        },
                    }
                )
                # Check limit after each document to ensure exact limit adherence
                if self.limit and len(result) >= self.limit:
                    break
            # Check limit after each question's documents as well
            if self.limit and len(result) >= self.limit:
                break

        logger.info("Loaded %d items from TriviaQA", len(result))
        return result
