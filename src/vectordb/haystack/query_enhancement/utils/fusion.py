"""Result fusion utilities for multi-query retrieval."""

import hashlib

from haystack import Document


def stable_doc_id(doc: Document) -> str:
    """Generate a stable document ID.

    Uses SHA1 hash of normalized content for stability across processes.
    Python's hash() is randomized per process, so we use SHA1.

    Args:
        doc: Haystack Document.

    Returns:
        Stable identifier string.
    """
    if doc.meta and doc.meta.get("doc_id"):
        return str(doc.meta["doc_id"])
    if doc.id:
        return doc.id

    content = (doc.content or "").lower()
    # Normalize all whitespace (including internal) by splitting and joining
    content = " ".join(content.split())
    return hashlib.sha1(content.encode(), usedforsecurity=False).hexdigest()


def rrf_fusion_many(
    ranked_lists: list[list[Document]],
    k: int = 60,
    top_k: int | None = None,
) -> list[Document]:
    """Reciprocal Rank Fusion for N ranked lists.

    RRF score = sum(1 / (k + rank)) across all lists where document appears.

    Args:
        ranked_lists: List of document lists, each ordered by relevance.
        k: RRF constant (default 60 per original paper).
        top_k: Number of results to return (default: all unique documents).

    Returns:
        Fused and reranked documents.
    """
    if not ranked_lists:
        return []

    # Calculate RRF scores
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for result_list in ranked_lists:
        for rank, doc in enumerate(result_list, start=1):
            doc_id = stable_doc_id(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

            # Keep first occurrence of document
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    fused_docs = [doc_map[doc_id] for doc_id in sorted_ids]

    if top_k is None:
        top_k = len(doc_map)

    return fused_docs[:top_k]


def deduplicate_by_content(docs: list[Document]) -> list[Document]:
    """Deduplicate documents by content.

    Args:
        docs: List of documents to deduplicate.

    Returns:
        Deduplicated document list.
    """
    seen_ids: set[str] = set()
    unique_docs: list[Document] = []

    for doc in docs:
        doc_id = stable_doc_id(doc)
        if doc_id not in seen_ids:
            unique_docs.append(doc)
            seen_ids.add(doc_id)

    return unique_docs
