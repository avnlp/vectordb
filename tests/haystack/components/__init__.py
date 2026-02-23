"""Tests for Haystack pipeline components.

This package contains comprehensive tests for all Haystack-specific components
used in RAG (Retrieval-Augmented Generation) pipelines. These components extend
Haystack's functionality with advanced features for vector database operations.

Components tested:
    - AgenticRouter: Routes queries to appropriate tools and reasoning paths
    - ContextCompressor: Reduces token usage by compressing retrieved context
    - DeepEvalEvaluator: Provides RAG evaluation metrics (recall, precision, etc.)
    - QueryEnhancer: Handles prompt engineering and query transformation
    - ResultMerger: Combines results from multiple retrieval sources

Each component is tested for:
    - Initialization with various configurations
    - Core functionality and edge cases
    - Error handling and fallbacks
    - Integration with Haystack pipeline framework
"""
