"""LangChain-specific dataloader tests.

This package contains tests for LangChain integration with dataloaders.
It verifies that dataset loaders work correctly with LangChain's
Document and chain abstractions.

Supported datasets:
    - ARC (AI2 Reasoning Challenge)
    - TriviaQA
    - PopQA
    - FactScore
    - Earnings Calls

Each dataset loader is tested for:
    - Correct conversion to LangChain Document format
    - Metadata preservation
    - Integration with LangChain chains and retrievers
"""
