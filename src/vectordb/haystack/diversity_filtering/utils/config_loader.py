"""YAML configuration loader with validation for diversity filtering pipelines.

Manages configuration for indexing and search pipelines across all 5 vector
databases with Pydantic-based validation and environment variable substitution.

Configuration Sections:
- dataset: Source data specification (TriviaQA, ARC, PopQA, FactScore, Earnings)
- embedding: Model settings for document/query encoding (model name, dimension,
  batch size, device)
- index: Vector collection name and storage parameters
- retrieval: Initial candidate pool size before diversity filtering
- diversity: Core diversity filtering settings:
  * algorithm: MMR, greedy_diversity_order, or clustering
  * top_k: Final number of diverse results to return
  * mmr_lambda: Trade-off parameter (0.0 = max diversity, 1.0 = max relevance)
  * similarity_metric: cosine or dot_product for distance calculations
- rag: Optional LLM generation settings (provider, model, temperature)
- vectordb: Database-specific connection parameters for Qdrant, Pinecone,
  Weaviate, Chroma, or Milvus

Environment variable substitution uses ${VAR_NAME} syntax for secure
secret management (API keys, URLs) without hardcoding credentials.
"""

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError


class DatasetConfig(BaseModel):
    """Configuration for dataset loading via DatasetRegistry."""

    name: Literal["triviaqa", "arc", "popqa", "factscore", "earnings_calls"] = Field(
        description="Dataset name"
    )
    split: str = Field(
        default="test", description="Dataset split (train, validation, test)"
    )
    max_documents: int | None = Field(
        default=None, description="Max documents to load (None = all)"
    )


class EmbeddingConfig(BaseModel):
    """Configuration for embedding model."""

    model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name or path",
    )
    dimension: int = Field(default=384, description="Embedding dimension size")
    batch_size: int = Field(default=32, description="Batch size for embedding")
    device: str | None = Field(
        default=None, description="Device to use (cuda, cpu, mps). None = auto-detect"
    )


class IndexConfig(BaseModel):
    """Configuration for vector index/collection."""

    name: str = Field(description="Index/collection name")
    recreate: bool = Field(
        default=False,
        description="Whether to recreate the index on each run (True = destructive, False = incremental/upsert)",
    )


class RetrievalConfig(BaseModel):
    """Configuration for document retrieval."""

    top_k_candidates: int = Field(
        default=100,
        description="Number of initial candidates to retrieve before diversity filtering",
    )


class DiversityConfig(BaseModel):
    """Configuration for diversity filtering algorithm."""

    algorithm: Literal[
        "maximum_margin_relevance", "greedy_diversity_order", "clustering"
    ] = Field(
        default="maximum_margin_relevance",
        description="Diversity filtering algorithm",
    )
    top_k: int = Field(default=10, description="Final number of diverse results", ge=1)
    mmr_lambda: float = Field(
        default=0.5,
        description="MMR trade-off parameter (0=diversity, 1=relevance)",
        ge=0.0,
        le=1.0,
    )
    similarity_metric: Literal["cosine", "dot_product"] = Field(
        default="cosine", description="Similarity metric for diversity calculation"
    )


class RAGConfig(BaseModel):
    """Configuration for Retrieval-Augmented Generation."""

    enabled: bool = Field(default=False, description="Enable RAG generation")
    provider: Literal["groq", "openai"] = Field(
        default="groq", description="LLM provider"
    )
    model: str = Field(default="llama-3.3-70b-versatile", description="Model name")
    temperature: float = Field(
        default=0.7,
        description="Model temperature",
        ge=0.0,
        le=2.0,
    )
    max_tokens: int = Field(default=2048, description="Max tokens in response")


class QdrantConfig(BaseModel):
    """Configuration for Qdrant vector database."""

    url: str = Field(description="Qdrant server URL")
    api_key: str | None = Field(default=None, description="API key (optional)")
    timeout: int = Field(default=30, description="Request timeout in seconds")


class PineconeConfig(BaseModel):
    """Configuration for Pinecone vector database."""

    api_key: str = Field(description="Pinecone API key")
    environment: str = Field(default="production", description="Pinecone environment")
    index_name: str = Field(description="Pinecone index name")


class WeaviateConfig(BaseModel):
    """Configuration for Weaviate vector database."""

    url: str = Field(description="Weaviate server URL")
    api_key: str | None = Field(default=None, description="API key (optional)")


class ChromaConfig(BaseModel):
    """Configuration for Chroma vector database."""

    host: str = Field(default="localhost", description="Chroma server host")
    port: int = Field(default=8000, description="Chroma server port")
    is_persistent: bool = Field(default=False, description="Use persistent storage")


class MilvusConfig(BaseModel):
    """Configuration for Milvus vector database."""

    host: str = Field(default="localhost", description="Milvus server host")
    port: int = Field(default=19530, description="Milvus server port")
    db_name: str = Field(default="default", description="Database name")


class VectorDBConfig(BaseModel):
    """Configuration for vector database backend."""

    type: Literal["qdrant", "pinecone", "weaviate", "chroma", "milvus"] = Field(
        description="Vector database type"
    )
    qdrant: QdrantConfig | None = Field(default=None, description="Qdrant config")
    pinecone: PineconeConfig | None = Field(default=None, description="Pinecone config")
    weaviate: WeaviateConfig | None = Field(default=None, description="Weaviate config")
    chroma: ChromaConfig | None = Field(default=None, description="Chroma config")
    milvus: MilvusConfig | None = Field(default=None, description="Milvus config")


class DiversityFilteringConfig(BaseModel):
    """Complete configuration for diversity filtering pipeline.

    Supports indexing and search pipelines across all 5 vector databases.
    """

    dataset: DatasetConfig
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    index: IndexConfig
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    diversity: DiversityConfig = Field(default_factory=DiversityConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    vectordb: VectorDBConfig


class ConfigLoader:
    """Load and validate diversity filtering configuration from YAML files.

    Supports environment variable substitution via ${VAR_NAME} syntax.

    Example:
        loader = ConfigLoader()
        config = loader.load("configs/qdrant/triviaqa/search.yaml")
    """

    @staticmethod
    def _resolve_env_vars(obj: Any) -> Any:
        """Recursively resolve environment variables in config object.

        Args:
            obj: Configuration object (dict, list, str, or other).

        Returns:
            Object with environment variables resolved.

        Raises:
            ValueError: If a required environment variable is missing.

        Example:
            {"key": "${HOME}/path"} -> {"key": "/home/user/path"}
        """
        if isinstance(obj, dict):
            return {k: ConfigLoader._resolve_env_vars(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [ConfigLoader._resolve_env_vars(item) for item in obj]

        if isinstance(obj, str):
            if obj.startswith("${") and obj.endswith("}"):
                env_var = obj[2:-1]
                value = os.getenv(env_var)
                if value is None:
                    msg = f"Required environment variable '{env_var}' is not set"
                    raise ValueError(msg)
                return value
            return obj

        return obj

    @staticmethod
    def load(config_path: str | Path) -> DiversityFilteringConfig:
        """Load configuration from YAML file with validation.

        Args:
            config_path: Path to YAML configuration file.

        Returns:
            Validated DiversityFilteringConfig object.

        Raises:
            FileNotFoundError: If config file not found.
            ValidationError: If config validation fails.
            yaml.YAMLError: If YAML parsing fails.
        """
        config_path = Path(config_path)

        if not config_path.exists():
            msg = f"Configuration file not found: {config_path}"
            raise FileNotFoundError(msg)

        try:
            with open(config_path) as f:
                raw_config = yaml.safe_load(f)

            if not isinstance(raw_config, dict):
                msg = f"Expected YAML to contain dict, got {type(raw_config)}"
                raise ValueError(msg)

            # Resolve environment variables
            resolved_config = ConfigLoader._resolve_env_vars(raw_config)

            return DiversityFilteringConfig(**resolved_config)

        except yaml.YAMLError as e:
            msg = f"Failed to parse YAML file {config_path}: {e}"
            raise yaml.YAMLError(msg) from e
        except ValidationError as e:
            msg = f"Configuration validation failed: {e}"
            raise ValidationError(msg) from e

    @staticmethod
    def load_dict(data: dict[str, Any]) -> DiversityFilteringConfig:
        """Load configuration from dictionary with validation.

        Args:
            data: Configuration dictionary.

        Returns:
            Validated DiversityFilteringConfig object.

        Raises:
            ValidationError: If config validation fails.
        """
        resolved_config = ConfigLoader._resolve_env_vars(data)
        return DiversityFilteringConfig(**resolved_config)
