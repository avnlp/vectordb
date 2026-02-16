"""Haystack-specific dataloader tests.

This package contains tests for Haystack integration with dataloaders.
It verifies that dataset loaders work correctly with Haystack's
Document, Pipeline, and component abstractions.

Supported datasets:
    - ARC (AI2 Reasoning Challenge)
    - TriviaQA
    - PopQA
    - FactScore
    - Earnings Calls

Each dataset loader is tested for:
    - Correct conversion to Haystack Document format
    - Metadata preservation in Document.meta
    - Integration with Haystack pipelines and retrievers
    - Text splitting with RecursiveDocumentSplitter
"""
