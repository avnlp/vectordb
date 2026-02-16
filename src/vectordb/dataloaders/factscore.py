"""FactScore dataset loader for fact verification.

FactScore evaluates RAG systems on atomic fact verification. Each sample
contains a Wikipedia article and a set of atomic facts derived from it.
This is useful for evaluating whether RAG systems can accurately retrieve
information for fact-level verification tasks.

Dataset Structure:
    - entity: The main entity being discussed
    - wikipedia_text: Full Wikipedia article text
    - topic: Topic of the article (often same as entity)
    - facts: High-level summary facts
    - decomposed_facts: Granular atomic facts for verification
    - one_fact_prompt: Prompt for single-fact evaluation
    - factscore_prompt: Prompt for FActScore evaluation

Use Case:
    Evaluating factuality of RAG responses at the atomic level.
    The dataset tests whether retrieved passages contain information
    needed to verify specific atomic facts about entities.

Data Source:
    HuggingFace dataset: dskar/FActScore
    Derived from Wikipedia with human-annotated atomic facts.

Fact Decomposition:
    The dataset distinguishes between high-level facts and decomposed
    atomic facts. Atomic facts are minimal, verifiable statements that
    can be individually evaluated against retrieved context.
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import load_dataset as hf_load_dataset


logger = logging.getLogger(__name__)


class FactScoreDataloader:
    """Loader for FactScore dataset.

    This dataloader loads Wikipedia articles with associated atomic facts
    for fact verification evaluation. The Wikipedia text serves as the
    indexable content, while facts are preserved in metadata for
    verification tasks.

    The dataset structure supports both high-level fact summaries and
    granular atomic facts, enabling evaluation at different levels
    of fact granularity.

    Attributes:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split to load
        limit: Maximum number of items to load
    """

    def __init__(
        self,
        dataset_name: str = "dskar/FActScore",
        split: str = "test",
        limit: int | None = None,
    ) -> None:
        """Initialize FactScore dataloader.

        Args:
            dataset_name: HuggingFace dataset identifier
            split: Dataset split to load (default: "test")
            limit: Maximum number of items to load (None for all)
        """
        self.dataset_name = dataset_name
        self.split = split
        self.limit = limit

    def load(self) -> list[dict[str, Any]]:
        """Load FactScore dataset and return standardized format.

        Processes Wikipedia articles with their associated facts. The
        Wikipedia text becomes the indexable content, while facts and
        entity information are preserved in metadata.

        Returns:
            List of standardized dicts with structure:
            {
                "text": "Wikipedia article text...",
                "metadata": {
                    "entity": "Entity name",
                    "topic": "Topic/category",
                    "id": "unique_id",
                    "facts": ["High-level fact 1", ...],
                    "decomposed_facts": ["Atomic fact 1", ...],
                    "one_fact_prompt": "Prompt for single-fact eval",
                    "factscore_prompt": "Prompt for FActScore eval"
                }
            }
        """
        dataset = hf_load_dataset(self.dataset_name, split=self.split, streaming=True)
        result: list[dict[str, Any]] = []

        for row in dataset:
            entity = row["entity"]
            wikipedia_text = row["wikipedia_text"]
            topic = row.get("topic", entity)
            row_id = row.get("id")
            facts = row.get("facts", [])
            decomposed_facts = row.get("decomposed_facts", [])

            result.append(
                {
                    "text": wikipedia_text,
                    "metadata": {
                        "entity": entity,
                        "topic": topic,
                        "id": row_id,
                        "facts": facts,
                        "decomposed_facts": decomposed_facts,
                        "one_fact_prompt": row["one_fact_prompt"],
                        "factscore_prompt": row["factscore_prompt"],
                    },
                }
            )
            if self.limit and len(result) >= self.limit:
                break

        logger.info("Loaded %d items from FactScore", len(result))
        return result
