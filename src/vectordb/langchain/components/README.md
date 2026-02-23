# LangChain Components

This module provides reusable pipeline components for advanced retrieval and RAG features within the LangChain integration. Each component encapsulates a specific stage of the retrieval pipeline - query expansion, post-retrieval compression, or agentic routing - and can be composed with any LangChain-compatible vector store and language model.

Compared to the Haystack components module, which includes five components (with additional support for evaluation and result merging as separate components), the LangChain components module provides three focused components. This reflects the imperative nature of LangChain pipelines, where operations like result merging and evaluation are handled directly through the utilities module rather than as discrete pipeline nodes.

## Overview

- Query expansion with three strategies: multi-query generation, hypothetical document embeddings, and step-back prompting
- Post-retrieval context compression using cross-encoder reranking or LLM-based passage extraction
- LLM-based agentic routing that dynamically selects between search, reflect, and generate actions
- All components work with any LangChain-compatible language model and vector store
- Stateless design allows concurrent use across multiple conversations

## How It Works

The query expansion component addresses vocabulary mismatch between user queries and indexed documents by generating multiple query variations. In multi-query mode, it produces up to five alternative phrasings of the original question. In hypothetical document mode (based on the HyDE technique), it generates a hypothetical answer and uses it alongside the original query for retrieval, bridging the distribution gap between short questions and longer documents. In step-back mode, it generates three broader context questions that retrieve background knowledge helpful for answering the specific query, then appends the original query.

The context compression component reduces the token count of retrieved documents before they are passed to the generation phase. In reranking mode, it uses a cross-encoder model to score each document against the query and retains only the top-scoring documents, preserving their original text. In LLM extraction mode, it passes all retrieved documents to a language model with instructions to extract only the passages relevant to the query, producing a single compressed document. The two modes offer different trade-offs: reranking preserves exact wording at a coarser granularity, while LLM extraction achieves higher compression ratios but may alter the source text.

The agentic routing component implements a decision-making pattern inspired by reasoning-and-acting frameworks. Given the current pipeline state -- including the query, whether documents have been retrieved, and any answer generated so far -- it invokes a language model to choose the next action. The three possible actions are searching for additional documents, reflecting on the current answer to identify gaps or errors, and generating a final answer. An iteration limit prevents infinite loops, and the router falls back to generating an answer when the limit is reached. The router is stateless, with all state passed through method parameters.

## Directory Structure

```
components/
    __init__.py              # Package exports for all components
    query_enhancer.py        # Query expansion (multi-query, HyDE, step-back prompting)
    context_compressor.py    # Context compression (cross-encoder reranking, LLM extraction)
    agentic_router.py        # Agentic routing (search, reflect, generate decisions)
```

## Related Modules

- `src/vectordb/haystack/components/` - Haystack components (includes evaluator and result merger in addition to query enhancement, compression, and routing)
- `src/vectordb/langchain/utils/` - Shared utility helpers consumed by these components
- `src/vectordb/langchain/agentic_rag/` - Feature module that uses the agentic router in end-to-end pipelines
- `src/vectordb/langchain/query_enhancement/` - Feature module that uses the query expansion component
- `src/vectordb/langchain/contextual_compression/` - Feature module that uses the context compression component
