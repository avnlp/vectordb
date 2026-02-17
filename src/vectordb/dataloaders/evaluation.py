"""Evaluation query extraction for normalized datasets."""

from __future__ import annotations

from vectordb.dataloaders.types import DatasetRecord, EvaluationQuery


class EvaluationExtractor:
    """Extract deduplicated evaluation queries from normalized records."""

    @staticmethod
    def extract(
        records: list[DatasetRecord],
        limit: int | None = None,
    ) -> list[EvaluationQuery]:
        """Extract queries and answers for evaluation.

        Iterates through dataset records and deduplicates queries by
        normalizing whitespace and case, ensuring that evaluation metrics
        are computed on unique queries. This prevents inflated metrics when
        the same question appears multiple times in a dataset (e.g., with
        different evidence documents in TriviaQA).

        Args:
            records: Normalized dataset records.
            limit: Optional limit applied after deduplication.

        Returns:
            List of evaluation queries deduplicated by query text.
        """
        seen_queries: set[str] = set()
        results: list[EvaluationQuery] = []

        for record in records:
            metadata = record.metadata
            # Attempt to extract query from either "question" or "entity" field
            # depending on the dataset format (e.g., PopQA uses entity).
            raw_query = metadata.get("question") or metadata.get("entity")
            if not raw_query:
                continue

            # Normalize query by collapsing whitespace and converting to lowercase
            # to enforce deterministic deduplication. Different question phrasings
            # with identical meaning will be deduplicated as a single query.
            normalized_query = " ".join(str(raw_query).split()).casefold()
            if not normalized_query or normalized_query in seen_queries:
                continue

            seen_queries.add(normalized_query)
            # Extract answers, which may be stored under either "answers" (list)
            # or "answer" (string) depending on the dataset schema.
            answers = metadata.get("answers")
            if answers is None:
                answers = metadata.get("answer")

            # Normalize answers to a consistent list format, filtering empty values.
            normalized_answers: list[str] = []
            if isinstance(answers, list):
                normalized_answers = [
                    value.strip()
                    for value in answers
                    if isinstance(value, str) and value.strip()
                ]
            elif isinstance(answers, str) and answers.strip():
                normalized_answers = [answers.strip()]

            # Extract document IDs when available for relevance evaluation.
            # Some datasets (e.g., TriviaQA) track which documents are relevant.
            relevant_doc_ids: list[str] = []
            if "id" in metadata and metadata["id"] is not None:
                relevant_doc_ids.append(str(metadata["id"]))

            # Retain dataset-specific metadata while removing the fields
            # that have been extracted and normalized into query/answers.
            filtered_metadata = {
                key: value
                for key, value in metadata.items()
                if key not in {"question", "answers", "answer", "entity"}
            }

            results.append(
                EvaluationQuery(
                    query=str(raw_query),
                    answers=normalized_answers,
                    relevant_doc_ids=relevant_doc_ids,
                    metadata=filtered_metadata,
                )
            )

            if limit is not None and len(results) >= limit:
                break

        return results
