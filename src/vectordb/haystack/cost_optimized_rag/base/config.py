"""Configuration management for cost-optimized RAG pipelines.

Loads YAML configs with pydantic validation and environment variable resolution.
Supports database-specific optimizations through typed configuration sections.

Configuration Philosophy:
    - Environment variables for secrets (API keys, passwords) prevent credential
      leakage in version control
    - Sensible defaults minimize configuration overhead for common use cases
    - Per-database sections enable vendor-specific optimizations
    - Batch sizes tuned for cost-efficient embedding API usage

Cost-Relevant Settings:
    - embeddings.batch_size: Larger batches reduce per-request overhead but
      increase memory usage. Default: 32 (balance between throughput and RAM)
    - embeddings.cache_embeddings: Enable to avoid re-embedding unchanged docs
    - indexing.quantization: Reduces storage costs by 2-4x with minimal recall loss
    - search.top_k: Lower values reduce LLM token costs (default: 10 vs typical 50)
    - reranker.use_crossencoder: Disable for 10x speedup when perfect ranking
      isn't critical
"""

from __future__ import annotations

import os
import re
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class VectorConfig(BaseModel):
    """Vector database configuration."""

    size: int = 1024
    distance: str = "Cosine"


class EmbeddingConfig(BaseModel):
    """Embedding model configuration.

    Controls text embedding generation with cost optimization options.

    Attributes:
        model: HuggingFace model identifier or local path.
            Default: "Qwen/Qwen3-Embedding-0.6B" (1024-dim, fast inference)
            Alternatives: "sentence-transformers/all-MiniLM-L6-v2"
                (384-dim, faster, lower quality)
        batch_size: Documents to embed per batch.
            Higher values amortize API overhead but increase memory.
            Default: 32 (optimal for most embedding APIs)
        backend: Inference backend.
            "transformers": Standard PyTorch (best compatibility)
            "onnx": ONNX Runtime (2-3x faster on CPU, lower compute cost)
        cache_embeddings: Persist embeddings to avoid recomputation.
            Reduces costs for static document collections by 100%
            after initial indexing.
    """

    model: str = "Qwen/Qwen3-Embedding-0.6B"
    batch_size: int = 32
    backend: str = "transformers"
    cache_embeddings: bool = False


class ChunkingConfig(BaseModel):
    """Text chunking configuration."""

    chunk_size: int = 512
    overlap: int = 50


class PartitionConfig(BaseModel):
    """Partitioning configuration."""

    enabled: bool = False
    partition_key: str = "dataset_id"
    values: list[str] = Field(default_factory=list)


class QuantizationConfig(BaseModel):
    """Quantization configuration per database."""

    enabled: bool = False
    method: str = "scalar"  # "scalar", "binary", "pq", "rq", etc.
    compression_ratio: float = 4.0


class RerankerConfig(BaseModel):
    """Reranking configuration."""

    use_crossencoder: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5
    cohere_enabled: bool = False
    cohere_api_key: str = ""


class SearchConfig(BaseModel):
    """Search configuration with cost-aware defaults.

    Attributes:
        top_k: Documents to retrieve per query.
            Directly impacts LLM token costs in RAG pipelines.
            Default: 10 (vs industry standard 50-100) for cost efficiency
        hybrid_enabled: Combine sparse (BM25) and dense retrieval.
            Adds ~20% latency but improves recall by 5-15% without
            increasing per-query token costs.
        reranking_enabled: Apply cross-encoder reranking.
            Improves precision@K by 10-20% but adds 50-200ms latency.
            Cost-neutral for token-based LLM APIs.
        metadata_filtering_enabled: Pre-filter by metadata before vector search.
            Reduces search space and costs for partitioned data.
    """

    top_k: int = 10
    hybrid_enabled: bool = True
    reranking_enabled: bool = False
    metadata_filtering_enabled: bool = False


class IndexingConfig(BaseModel):
    """Indexing configuration."""

    partitions: PartitionConfig = Field(default_factory=PartitionConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    vector_config: VectorConfig = Field(default_factory=VectorConfig)
    payload_indexes: list[dict[str, Any]] = Field(default_factory=list)


class DataloaderConfig(BaseModel):
    """Dataloader configuration."""

    type: str
    dataset_name: str = ""
    split: str = "test"
    limit: int | None = None


class GeneratorConfig(BaseModel):
    """LLM generator configuration for RAG."""

    enabled: bool = True
    provider: str = "groq"  # "groq" or "openai"
    model: str = "llama-3.3-70b-versatile"
    api_key: str = ""
    api_base_url: str = "https://api.groq.com/openai/v1"
    temperature: float = 0.7
    max_tokens: int = 2048


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    name: str = "cost_optimized_rag"


class DatabaseConfig(BaseModel):
    """Base database configuration."""

    host: str = "localhost"
    port: int = 6333
    api_key: str = ""


class QdrantConfig(DatabaseConfig):
    """Qdrant-specific configuration."""

    https: bool = False
    port: int = 6333


class MilvusConfig(DatabaseConfig):
    """Milvus-specific configuration."""

    port: int = 19530


class WeaviateConfig(DatabaseConfig):
    """Weaviate-specific configuration."""

    port: int = 8080


class PineconeConfig(BaseModel):
    """Pinecone-specific configuration."""

    api_key: str
    environment: str = "us-west4-gcp"


class ChromaConfig(BaseModel):
    """Chroma-specific configuration."""

    path: str = "./chroma_data"


class CollectionConfig(BaseModel):
    """Collection/index configuration."""

    name: str
    description: str = ""


class RAGConfig(BaseModel):
    """Complete RAG pipeline configuration."""

    dataloader: DataloaderConfig
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    collection: CollectionConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    qdrant: QdrantConfig | None = None
    milvus: MilvusConfig | None = None
    weaviate: WeaviateConfig | None = None
    pinecone: PineconeConfig | None = None
    chroma: ChromaConfig | None = None
    generator: GeneratorConfig = Field(default_factory=GeneratorConfig)

    @field_validator("dataloader", mode="before")
    @classmethod
    def validate_dataloader(cls, v: Any) -> Any:
        """Validate dataloader configuration."""
        if isinstance(v, dict):
            return DataloaderConfig(**v)
        return v


def _resolve_env_vars(value: Any) -> Any:
    """Resolve environment variables in configuration values.

    Supports both simple ${VAR} and ${VAR:-default} syntax.

    Args:
        value: The value to resolve, can be a string, dict, or list.

    Returns:
        The resolved value with environment variables expanded.
    """
    if isinstance(value, str):
        # Match ${VAR} or ${VAR:-default}
        pattern = r"\$\{([^}:]+)(?::-([^}]*))?\}"
        match = re.match(pattern, value)
        if match:
            env_var = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(env_var, default)
        return value
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def load_config(config_path: str) -> RAGConfig:
    """Load and validate RAG configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Validated RAGConfig instance.

    Raises:
        FileNotFoundError: If config file not found.
        ValueError: If config validation fails.
    """
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        raise ValueError(f"Empty configuration file: {config_path}")

    resolved_config = _resolve_env_vars(raw_config)
    return RAGConfig(**resolved_config)
