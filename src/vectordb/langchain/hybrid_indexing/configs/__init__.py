"""Configuration schemas and utilities for hybrid indexing pipelines.

This module defines configuration structures for hybrid search setups across
different vector databases. Configurations typically specify:

- Database connection parameters (API keys, URLs, collection names)
- Embedding model settings (dense embedder configuration)
- Hybrid search parameters (alpha weighting between dense/sparse)
- Data loading options (source, preprocessing, limits)
- Optional LLM configuration for RAG pipelines

Configuration files are typically YAML or Python dictionaries that get
validated against database-specific schemas.

Example:
    pinecone:
      api_key: "${PINECONE_API_KEY}"
      index_name: "hybrid-index"
      namespace: "default"
      dimension: 384
      alpha: 0.5

    embedder:
      type: "sentence-transformers"
      model: "all-MiniLM-L6-v2"

    dataloader:
      type: "text"
      source: "data/documents/"
      limit: 1000
"""
