# Haystack Pipeline Components

Reusable pipeline components shared across all Haystack feature modules. Each component encapsulates a distinct retrieval or generation capability and can be composed into larger pipelines alongside any supported vector database.

## Components

### Query Expansion

Generates alternative formulations of the original query to improve retrieval recall. Supports three strategies:

- **Multi-query** - Rephrases the original question into several semantically similar variations, each targeting a slightly different angle of the same intent.
- **Hypothetical document embedding** - Generates a synthetic answer passage and uses it as the search query, aligning the query representation with the document embedding space.
- **Step-back prompting** - Produces broader, more abstract questions that capture the general context behind the original query, helping retrieve background information.

An LLM generates all variations, and the resulting queries are dispatched to the retriever in parallel.

### Result Fusion

Merges ranked results from multiple retrieval queries into a single unified list. Supports reciprocal rank fusion, which combines rankings based on position rather than raw scores, and weighted fusion, which applies configurable score weights to each source. Deduplicates documents that appear in more than one result set before producing the final ranking.

### Context Compression

Reduces retrieved document content to only the most relevant passages, lowering noise and token consumption before answer generation. Operates in two modes:

- **Cross-encoder reranking** - Scores each document against the query using a cross-encoder model and filters out low-relevance results.
- **LLM-based extraction** - Summarizes the relevant portions of each document using a language model, producing concise passages that directly address the query.

### Evaluation

Measures the quality of RAG pipeline outputs using three metrics: contextual precision (relevance of retrieved documents), contextual recall (coverage of expected information), and faithfulness (whether the generated answer is grounded in the retrieved context). Integrates with external evaluation frameworks to compute these scores.

### Agentic Routing

Routes queries through a multi-step reasoning loop where a language model decides the next action at each step. The available actions are: searching the database for additional information, reflecting on the quality of current results, or generating a final answer. This enables iterative refinement -- if the initial retrieval is insufficient, the system can reformulate the query and search again. The maximum number of iterations is configurable to balance quality against latency.

## Directory Structure

```
components/
    __init__.py
    agentic_router.py
    context_compressor.py
    evaluators.py
    query_enhancer.py
    result_merger.py
```

## Related Modules

- [`haystack/utils/`](../utils/) - Shared utility helpers for configuration loading, embedding creation, filtering, and result processing used by these components.
