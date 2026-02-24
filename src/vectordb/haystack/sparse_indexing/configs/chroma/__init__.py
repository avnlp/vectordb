"""Configuration files for Chroma sparse indexing pipelines.

Provides YAML configuration templates for sparse vector indexing with Chroma database.
Configurations specify sparse embedding models, index settings, and database
connection parameters optimized for lexical and hybrid search scenarios.

Directory Structure:
    configs/chroma/
    ├── triviaqa/
    │   ├── sparse_bm25.yaml              # BM25 sparse indexing config
    │   ├── sparse_splade.yaml            # SPLADE sparse indexing config
    │   └── hybrid.yaml                   # Hybrid sparse+dense config
    ├── arc/                              # ARC dataset configs
    ├── popqa/                            # PopQA dataset configs
    ├── factscore/                        # FactScore dataset configs
    └── earnings_calls/                   # EarningsCall dataset configs

Configuration Sections:
    - embeddings: Sparse model (BM25/SPLADE), tokenizer, batch_size
    - sparse_index: Index type, term frequency settings, normalization
    - retrieval: top_k, filter conditions, lexical weighting
    - database: Chroma connection (host, port, collection_name, embedding_function)
    - dataset: Type, split, limit for indexing, preprocessing steps

Total Configurations: 15 files per database
    (5 databases × 5 datasets × 3 sparse/hybrid types)

Sparse Config Examples:
    BM25: {type: "bm25", k1: 1.2, b: 0.75, tokenizer: "whitespace"}
    SPLADE: {type: "splade", model: "naver/splade-cocondenser-ensembledistil"}
    Hybrid: {type: "hybrid", sparse_weight: 0.3, dense_weight: 0.7,
        dense_model: "sentence-transformers/all-MiniLM-L6-v2"}
"""
