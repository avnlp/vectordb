"""TypedDicts for query enhancement configuration validation."""

from typing import Any, Dict, Literal, NotRequired, TypedDict, Union


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


# Database-specific configuration TypedDicts


class PineconeConfig(TypedDict):
    """Configuration for Pinecone vector database.

    Attributes:
        api_key: Pinecone API key for authentication.
        index_name: Name of the Pinecone index to use.
        namespace: Namespace within the index (Pinecone supports partitioning).
    """

    api_key: str
    index_name: str
    namespace: str


class QdrantConfig(TypedDict):
    """Configuration for Qdrant vector database.

    Attributes:
        url: Qdrant server URL (e.g., http://localhost:6333).
        collection_name: Name of the Qdrant collection.
        api_key: Optional API key for authentication.
        grpc: Whether to use gRPC for communication (default: False).
    """

    url: str
    collection_name: str
    api_key: NotRequired[str]
    grpc: NotRequired[bool]


class ChromaConfig(TypedDict):
    """Configuration for Chroma vector database.

    Attributes:
        path: Local filesystem path for persistent storage (for local Chroma).
        host: Host address for remote Chroma server.
        port: Port number for remote Chroma server.
        collection_name: Name of the Chroma collection.

    Note:
        Either use `path` for local Chroma or `host`/`port` for remote Chroma.
    """

    path: NotRequired[str]
    host: NotRequired[str]
    port: NotRequired[int]
    collection_name: str


class MilvusConfig(TypedDict):
    """Configuration for Milvus vector database.

    Attributes:
        uri: Milvus server URI (e.g., http://localhost:19530).
        token: Optional authentication token for Milvus.
        collection_name: Name of the Milvus collection.
        database: Optional database name within Milvus (default: "default").
    """

    uri: str
    token: NotRequired[str]
    collection_name: str
    database: NotRequired[str]


class WeaviateConfig(TypedDict):
    """Configuration for Weaviate vector database.

    Attributes:
        url: Weaviate cluster URL.
        api_key: Optional API key for authentication.
        collection_name: Name of the Weaviate collection.
        grpc_port: Optional gRPC port for optimized queries (default: 50051).
    """

    url: str
    api_key: NotRequired[str]
    collection_name: str
    grpc_port: NotRequired[int]


class QueryEnhancementPipelineConfig(TypedDict):
    """Complete configuration for query enhancement pipeline.

    This TypedDict defines the structure for configuring a query enhancement
    pipeline. Only the database configuration for the vector database you're
    using needs to be specified - all database configs are optional.

    Attributes:
        dataloader: Configuration for data loading.
        embeddings: Configuration for embedding model.
        query_enhancement: Configuration for query enhancement strategy.
        logging: Logging configuration (level, format, etc.).
        rag: RAG-specific configuration options.
        pinecone: Optional Pinecone database configuration.
        qdrant: Optional Qdrant database configuration.
        milvus: Optional Milvus database configuration.
        chroma: Optional Chroma database configuration.
        weaviate: Optional Weaviate database configuration.

    Example:
        >>> config: QueryEnhancementPipelineConfig = {
        ...     "dataloader": {"type": "triviaqa", "params": {}},
        ...     "embeddings": {"model": "all-MiniLM-L6-v2", "params": {}},
        ...     "query_enhancement": {...},
        ...     "logging": {"level": "INFO"},
        ...     "rag": {"enabled": True},
        ...     # Only specify the database you're using:
        ...     "qdrant": {
        ...         "url": "http://localhost:6333",
        ...         "collection_name": "my_coll",
        ...     },
        ... }
    """

    dataloader: DataLoaderConfig
    embeddings: EmbeddingConfig
    query_enhancement: QueryEnhancementConfig
    logging: Dict[str, Any]
    rag: Dict[str, Any]
    # Database configurations are optional - only specify the one(s) you use
    pinecone: NotRequired[PineconeConfig]
    qdrant: NotRequired[QdrantConfig]
    milvus: NotRequired[MilvusConfig]
    chroma: NotRequired[ChromaConfig]
    weaviate: NotRequired[WeaviateConfig]
