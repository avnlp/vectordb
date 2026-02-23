"""Agentic RAG search pipelines for all vector databases.

This module provides search pipelines that implement agentic RAG for all
supported vector databases. Each pipeline implements the agentic loop:
retrieval, reflection, and generation with intelligent routing between actions.

Agentic RAG Search Architecture:
    All search pipelines follow a consistent agentic loop pattern:

    1. Initialize: Load configuration and initialize components (router,
       compressor, LLM, vector store connection)

    2. Agentic Loop: While iterations < max_iterations:
       a. Route: AgenticRouter decides search, reflect, or generate
       b. Execute: Perform the chosen action
          - Search: Retrieve and compress documents
          - Reflect: Evaluate answer quality and identify gaps
          - Generate: Create final answer using context
       c. Update: Track intermediate steps and reasoning

    3. Return: Provide final answer with metadata

Pipeline Consistency:
    All pipelines share identical interfaces:
    - __init__(config_or_path): Initialize from dict or YAML file path
    - run(query, top_k, filters) -> dict: Execute agentic RAG

    Results dict contains:
        - final_answer: Generated answer string
        - documents: List of Document objects used
        - intermediate_steps: Actions taken (e.g., ['search', 'reflect', 'generate'])
        - reasoning: Router's reasoning for each decision

Supported Databases:
    - ChromaAgenticRAGPipeline
    - MilvusAgenticRAGPipeline
    - PineconeAgenticRAGPipeline
    - QdrantAgenticRAGPipeline
    - WeaviateAgenticRAGPipeline

Usage:
    >>> from vectordb.langchain.agentic_rag.search import PineconeAgenticRAGPipeline
    >>> pipeline = PineconeAgenticRAGPipeline("config.yaml")
    >>> result = pipeline.run("What is quantum computing?")
    >>> print(result["final_answer"])
    >>> print(f"Steps: {result['intermediate_steps']}")
"""

from vectordb.langchain.agentic_rag.search.chroma import ChromaAgenticRAGPipeline
from vectordb.langchain.agentic_rag.search.milvus import MilvusAgenticRAGPipeline
from vectordb.langchain.agentic_rag.search.pinecone import PineconeAgenticRAGPipeline
from vectordb.langchain.agentic_rag.search.qdrant import QdrantAgenticRAGPipeline
from vectordb.langchain.agentic_rag.search.weaviate import WeaviateAgenticRAGPipeline


__all__ = [
    "PineconeAgenticRAGPipeline",
    "WeaviateAgenticRAGPipeline",
    "ChromaAgenticRAGPipeline",
    "MilvusAgenticRAGPipeline",
    "QdrantAgenticRAGPipeline",
]
