"""Tests for cost-optimized RAG example implementations.

This package contains validation tests for example code demonstrating
cost-optimized RAG patterns. Examples serve as reference implementations
showing best practices for reducing API costs and latency.

Example Categories Tested:
    - Basic Usage Patterns:
        * Single database configuration and initialization
        * Simple document indexing workflows
        * Query execution with result retrieval
        * YAML-based configuration loading

    - Advanced Patterns:
        * Hybrid search combining dense and sparse retrieval
        * Multi-database comparison and selection
        * Batch processing for throughput optimization
        * Local vs API embedding strategies

    - Cost Optimization Strategies:
        * Query routing based on complexity classification
        * Tiered retrieval using cheaper models for simple queries
        * Response caching for frequently asked questions
        * Model selection balancing cost and quality

Validation Approach:
    Tests verify that example code is syntactically correct and follows
    established patterns. Examples are validated against the actual
    implementation to ensure they remain accurate as the library evolves.

Quality Checks:
    - Configuration validation ensures YAML examples are well-formed
    - Import tests verify all referenced modules exist
    - Pattern tests confirm examples follow recommended practices
    - Documentation sync ensures comments match implementation

Test Scope:
    These tests focus on example correctness rather than full integration
    testing. They catch broken references, outdated APIs, and documentation
    drift. Database-specific tests in sibling packages provide integration
    coverage.

Note:
    Examples are illustrative patterns, not executable scripts. Tests
    validate the patterns themselves without requiring live services.
"""
