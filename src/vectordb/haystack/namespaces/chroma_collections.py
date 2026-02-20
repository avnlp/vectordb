"""Chroma collection management for Haystack."""

from typing import Any

from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
)

from vectordb.dataloaders import (
    ARCLoader,
    EarningsCallsLoader,
    FactScoreLoader,
    PopQALoader,
    TriviaQALoader,
)


def get_dataloader_map() -> dict[str, type]:
    """Return a mapping of dataloader names to their classes."""
    return {
        "triviaqa": TriviaQALoader,
        "arc": ARCLoader,
        "popqa": PopQALoader,
        "factscore": FactScoreLoader,
        "earnings_calls": EarningsCallsLoader,
    }


def generate_embeddings(
    dense_model: str,
    haystack_documents_split1: list[Any],
    haystack_documents_split2: list[Any],
) -> tuple[list[Any], list[Any]]:
    """Initialize embedder and generate embeddings for both splits.

    Args:
        dense_model: Model name for the embedder.
        haystack_documents_split1: Documents from first split.
        haystack_documents_split2: Documents from second split.

    Returns:
        Tuple of embedded documents for both splits.
    """
    embedder = SentenceTransformersDocumentEmbedder(model=dense_model)
    embedder.warm_up()

    docs_with_embeddings_split1 = embedder.run(documents=haystack_documents_split1)[
        "documents"
    ]
    docs_with_embeddings_split2 = embedder.run(documents=haystack_documents_split2)[
        "documents"
    ]

    return docs_with_embeddings_split1, docs_with_embeddings_split2
