"""Configuration files for Weaviate sparse indexing pipelines.

Provides YAML configuration templates for sparse vector indexing with Weaviate database.
Configurations specify sparse embedding models, index settings, and database
connection parameters optimized for lexical and hybrid search scenarios.

Directory Structure:
    configs/weaviate/
    ├── triviaqa.yaml                     # TriviaQA dataset configs
    ├── arc.yaml                          # ARC dataset configs
    ├── popqa.yaml                        # PopQA dataset configs
    ├── factscore.yaml                    # FactScore dataset configs
    └── earnings_calls.yaml               # EarningsCall dataset configs

Configuration Sections:
    - embeddings: Sparse model (BM25/SPLADE), tokenizer, batch_size
    - sparse_index: Index type, term frequency settings, bm25 config
    - retrieval: top_k, fusion_type (rankedFusion), alpha for hybrid
    - database: Weaviate connection (host, port, api_key, class_name, schema)
    - dataset: Type, split, limit for indexing, preprocessing steps

Total Configurations: 15 files per database
    (5 databases × 5 datasets × 3 sparse/hybrid types)

Sparse Config Examples:
    BM25: {type: "bm25", k1: 1.2, b: 0.75, tokenizer: "whitespace"}
    SPLADE: {type: "splade", model: "naver/splade-cocondenser-ensembledistil"}
    Hybrid: {type: "hybrid", sparse_weight: 0.3, dense_weight: 0.7,
        dense_model: "sentence-transformers/all-MiniLM-L6-v2"}
"""
