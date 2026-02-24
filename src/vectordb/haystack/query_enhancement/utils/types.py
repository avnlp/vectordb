"""TypedDicts for query enhancement configuration validation."""

from typing import Any, Dict, Literal, TypedDict, Union


class DataLoaderConfig(TypedDict):
    """Configuration for dataloader."""

    type: str
    params: Dict[str, Any]


class EmbeddingConfig(TypedDict):
    """Configuration for embeddings."""

    model: str
    params: Dict[str, Any]


class LLMConfig(TypedDict):
    """Configuration for LLM generator."""

    model: str
    api_key: Union[str, None]


class QueryEnhancementConfig(TypedDict):
    """Configuration for query enhancement."""

    type: Literal["multi_query", "hyde", "step_back"]
    num_queries: int
    num_hyde_docs: int
    llm: LLMConfig
    fusion_method: Literal["rrf", "weighted"]
    rrf_k: int
    top_k: int


class VectorDBConfig(TypedDict):
    """Configuration for vector database."""

    api_key: str
    index_name: str
    namespace: str


class QueryEnhancementPipelineConfig(TypedDict):
    """Complete configuration for query enhancement pipeline."""

    dataloader: DataLoaderConfig
    embeddings: EmbeddingConfig
    query_enhancement: QueryEnhancementConfig
    pinecone: VectorDBConfig
    qdrant: VectorDBConfig
    milvus: VectorDBConfig
    chroma: VectorDBConfig
    weaviate: VectorDBConfig
    logging: Dict[str, Any]
    rag: Dict[str, Any]
