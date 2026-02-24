"""Query enhancement pipelines for multi-query, HyDE, and step-back strategies.

Query enhancement improves retrieval quality by transforming or expanding the
original user query before vector search. This addresses common failure modes
in naive RAG: vocabulary mismatch, ambiguous queries, and insufficient context.

Enhancement Strategies:

Multi-Query Expansion:
    Generates multiple reformulations of the original query using an LLM.
    Each variant captures different aspects or phrasings, increasing the
    chance of matching relevant documents. Results are fused using RRF.

    Example: "What is photosynthesis?" expands to:
    - "How do plants convert sunlight to energy?"
    - "Explain the process of photosynthesis in biology"
    - "What chemical reactions occur during photosynthesis?"

HyDE (Hypothetical Document Embeddings):
    Instead of embedding the query directly, generates a hypothetical answer
    document using an LLM, then embeds that document. The embedding of a
    plausible answer often aligns better with actual answer documents than
    the question embedding alone.

    Process: Query → LLM generates hypothetical answer → Embed answer → Search

Step-Back Prompting:
    For specific or narrow queries, generates a more general "step-back"
    question that provides broader context. Retrieves documents for both
    the original and step-back queries, combining results.

    Example: "What temperature does water boil at on Mount Everest?"
    Step-back: "How does altitude affect boiling point?"

Pipeline Components:
    - Indexing pipelines: Store documents with embeddings for later retrieval
    - Search pipelines: Apply query enhancement before vector search

Supported Databases:
    Milvus, Pinecone, Qdrant, Weaviate, Chroma
"""
