"""PopQA dataset loader for entity-centric factoid QA.

PopQA (Popular Question Answering) contains factoid questions about
popular entities, designed to test knowledge retrieval for well-known
topics. The dataset focuses on entity-property-object triples derived
from Wikidata.

Dataset Structure:
    - question: Natural language factoid question
    - possible_answers: List of acceptable answer strings
    - subj: Subject entity (e.g., "Barack Obama")
    - prop: Property/relationship (e.g., "place of birth")
    - obj: Object entity (e.g., "Honolulu")
    - content: Optional content text for indexing

Use Case:
    Evaluating RAG systems on entity-centric knowledge retrieval.
    The dataset tests whether systems can retrieve facts about
    popular entities from knowledge bases or documents.

Data Source:
    HuggingFace dataset: akariasai/PopQA
    Derived from Wikidata with natural language questions.

Entity-Relation Structure:
    Each question corresponds to a Wikidata triple (subject, property, object).
    The subject is the entity being asked about, the property is the
    relationship, and the object is the answer.
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import load_dataset as hf_load_dataset


logger = logging.getLogger(__name__)


class PopQADataloader:
    """Loader for PopQA dataset.

    This dataloader loads factoid questions about popular entities.
    Each sample contains a question, possible answers, and the underlying
    Wikidata triple (subject, property, object) that defines the fact.

    The content field provides text for indexing, defaulting to the
    question text if no separate content is available.

    Attributes:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split to load
        limit: Maximum number of items to load
    """

    def __init__(
        self,
        dataset_name: str = "akariasai/PopQA",
        split: str = "test",
        limit: int | None = None,
    ) -> None:
        """Initialize PopQA dataloader.

        Args:
            dataset_name: HuggingFace dataset identifier
            split: Dataset split to load (default: "test")
            limit: Maximum number of items to load (None for all)
        """
        self.dataset_name = dataset_name
        self.split = split
        self.limit = limit

    def load(self) -> list[dict[str, Any]]:
        """Load PopQA dataset and return standardized format.

        Processes factoid questions with their Wikidata triple structure.
        The content field (or question as fallback) becomes the indexable
        text, while entity metadata enables entity-aware retrieval.

        Returns:
            List of standardized dicts with structure:
            {
                "text": "Content or question text...",
                "metadata": {
                    "question": "Natural language question",
                    "answers": ["Answer 1", "Answer 2", ...],
                    "entity": "Subject entity (e.g., 'Barack Obama')",
                    "predicate": "Property/relationship (e.g., 'place of birth')",
                    "object": "Object entity (e.g., 'Honolulu')"
                }
            }
        """
        dataset = hf_load_dataset(self.dataset_name, split=self.split, streaming=True)
        result: list[dict[str, Any]] = []

        for row in dataset:
            question = row["question"]
            possible_answers = row["possible_answers"]
            content = row.get("content", question)

            result.append(
                {
                    "text": content,
                    "metadata": {
                        "question": question,
                        "answers": possible_answers,
                        "entity": row["subj"],
                        "predicate": row["prop"],
                        "object": row["obj"],
                    },
                }
            )
            if self.limit and len(result) >= self.limit:
                break

        logger.info("Loaded %d items from PopQA", len(result))
        return result
