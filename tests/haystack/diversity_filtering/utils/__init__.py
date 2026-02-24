"""Tests for diversity filtering utility modules.

This package contains tests for supporting utilities used by diversity-aware
retrieval pipelines. Utilities handle configuration management, prompt
templates, and cross-cutting concerns for all vector database backends.

ConfigLoader Tests:
    - Configuration model validation using Pydantic
    - Environment variable substitution (${VAR_NAME} syntax)
    - YAML file parsing and error handling
    - Database-specific settings (Qdrant, Pinecone, Weaviate, Chroma, Milvus)

Configuration Models Tested:
    - DatasetConfig: Dataset selection (TriviaQA, ARC, PopQA, FactScore, Earnings)
    - EmbeddingConfig: Model selection, dimensions, batch size, device
    - IndexConfig: Index naming and metadata
    - RetrievalConfig: Candidate count before diversity filtering
    - DiversityConfig: Algorithm selection (MMR, clustering), lambda trade-off
    - VectorDBConfig: Backend-specific connection parameters
    - RAGConfig: LLM provider settings for response generation

Prompt Utilities Tests:
    - Template loading for dataset-specific RAG prompts
    - Document formatting for LLM context windows
    - Template variable substitution and rendering

Environment Variable Resolution:
    - Simple substitution: ${API_KEY} -> actual value
    - Nested resolution: Variables in nested dicts and lists
    - Missing variable handling: Empty string or error based on context
    - Security: No hardcoded credentials in test configurations

Validation Coverage:
    - Required field enforcement
    - Type coercion and validation
    - Enum validation for database types and algorithms
    - Range validation for numeric parameters (temperature, top_k)

Test Data:
    - Minimal valid configurations for each backend
    - Full configurations with all optional fields
    - Invalid configurations to test error handling
    - Environment variable templates for substitution testing

Error Handling:
    - Invalid YAML syntax detection
    - Missing required field validation
    - Type mismatch errors with helpful messages
    - Database connection parameter validation
"""
