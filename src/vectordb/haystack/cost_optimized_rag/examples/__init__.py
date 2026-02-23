"""Examples for cost-optimized RAG pipelines.

Reference implementations demonstrating cost-optimized patterns for
different vector databases and use cases. Each example emphasizes
cost-conscious design decisions.

Example Categories:

    Basic Usage:
        - Single database indexing
        - Simple search with RAG
        - Configuration examples

    Advanced Patterns:
        - Hybrid search (sparse + dense)
        - Multi-database comparison
        - Batch processing

    Cost Optimization:
        - Local embedding strategies
        - Batch size tuning
        - Query optimization

Usage:
    Examples demonstrate patterns but are not executable scripts.
    Copy patterns into your application code.

    Typical workflow:
        1. Configure YAML with database settings
        2. Initialize indexer pipeline
        3. Run indexing
        4. Initialize searcher pipeline
        5. Execute queries

Cost Patterns Demonstrated:
    - Local embeddings (vs API)
    - Batch processing (vs single)
    - Efficient templates (minimal tokens)
    - Result filtering (reduce data transfer)

See individual database modules for specific examples:
    - indexing/: Database-specific indexing
    - search/: Database-specific search
"""
