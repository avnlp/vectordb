"""Tests for Haystack utility classes and helpers.

This package contains tests for shared utility classes and helper functions
used across Haystack pipeline implementations. These utilities provide
common functionality for embeddings, configuration, and data processing.

Utility categories:
    - Embedding helpers: Document and query embedding utilities
    - Configuration loaders: YAML/JSON config parsing and validation
    - Filter builders: Database-agnostic filter construction
    - RAG helpers: Generator setup and prompt formatting
    - Fusion utilities: Score combination and result merging

Tested utilities:
    - EmbedderFactory: Create document and text embedders
    - ConfigLoader: Load and validate pipeline configurations
    - FilterBuilder: Construct database-specific filters
    - RAGHelper: Setup generators and format prompts
    - DataLoaderHelper: Load documents from various sources

Each utility is tested for:
    - Correct initialization with various parameters
    - Error handling for invalid inputs
    - Integration with pipeline components
    - Edge cases and special configurations
"""
