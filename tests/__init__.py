"""Test suite for the vectordb library.

This package contains comprehensive tests for:
- Database integrations (Pinecone, Weaviate, Chroma, Milvus, Qdrant)
- Haystack pipeline components and integrations
- LangChain integrations and utilities
- Dataloaders for benchmark datasets (ARC, TriviaQA, PopQA, FactScore)
- Advanced RAG features (MMR, contextual compression, diversity filtering, etc.)

The test suite is organized into the following modules:
- tests/databases: Tests for shared vector database wrappers
- tests/dataloaders: Tests for dataset loading utilities
- tests/haystack: Tests for Haystack pipeline components
- tests/langchain: Tests for LangChain integrations
- tests/utils: Tests for utility functions and helpers

Tests are marked with @pytest.mark.integration_test for database integration
tests that require running database instances.
"""
