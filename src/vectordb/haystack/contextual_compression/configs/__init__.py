"""Configuration files for contextual compression pipelines.

Provides YAML configuration templates for all supported database and compression
combinations. Configurations specify embedding models, compression strategies,
and database connection parameters.

Directory Structure:
    configs/
    ├── milvus/
    │   ├── triviaqa/
    │   │   ├── reranking.yaml         # Cross-encoder reranking config
    │   │   └── llm_extraction.yaml    # GPT-4o-mini extraction config
    │   ├── arc/                       # ARC dataset configs
    │   ├── popqa/                     # PopQA dataset configs
    │   ├── factscore/                 # FactScore dataset configs
    │   └── earnings_calls/            # EarningsCall dataset configs
    ├── pinecone/... [same structure]
    ├── qdrant/... [same structure]
    ├── chroma/... [same structure]
    └── weaviate/... [same structure]

Configuration Sections:
    - embeddings: Model, dimension, batch_size
    - compression: Type (reranking/llm_extraction), algorithm-specific params
    - retrieval: top_k, candidate_multiplier
    - database: Connection parameters (host, port, api_key, etc.)
    - dataset: Type, split, limit for indexing

Total Configurations: 50 files
    (5 databases × 5 datasets × 2 compression types)

Compression Config Examples:
    Reranking: {type: "reranking", reranker: {type: "cross_encoder",
        model: "BAAI/bge-reranker-v2-m3"}}
    LLM Extraction: {type: "llm_extraction", llm: {model: "gpt-4o-mini",
        api_key: "${OPENAI_API_KEY}"}}
"""
