# Components (Haystack)

The components directory contains reusable, self-contained Haystack pipeline building blocks that implement specific retrieval and generation sub-tasks. Feature pipelines compose these components rather than reimplementing the same logic.

## Components

### `AgenticRouter` (`agentic_router.py`)

An LLM-based decision-making component for agentic RAG pipelines. Supports:

- **Tool selection**: Given a query, selects the appropriate processing path (`"retrieval"`, `"web_search"`, `"calculation"`, or `"reasoning"`). Temperature is set to 0 for deterministic routing.
- **Answer quality evaluation**: Sends the query, draft answer, and retrieved context to the LLM and receives a JSON-structured assessment with `relevance`, `completeness`, and `grounding` scores (0–100) plus issues and suggestions.
- **Refinement decision**: Computes whether the average quality score falls below a threshold and decides whether another refinement iteration is needed.
- **Answer refinement**: Given issues and suggestions from evaluation, sends a targeted revision request to the LLM.
- **Self-reflection loop** (`self_reflect_loop`): Orchestrates the full evaluate-refine cycle for up to `max_iterations` rounds with an early-exit when the quality threshold is met.

Uses Haystack's `OpenAIChatGenerator` configured to call the Groq API (or any OpenAI-compatible endpoint). Requires the `GROQ_API_KEY` environment variable.

```python
from vectordb.haystack.components import AgenticRouter

router = AgenticRouter(model="llama-3.3-70b-versatile")
tool = router.select_tool("What is quantum entanglement?")  # → "retrieval"

quality = router.evaluate_answer_quality(query, answer, context)
# quality = {"relevance": 85, "completeness": 70, "grounding": 90, "issues": [...]}

final_answer = router.self_reflect_loop(query, draft_answer, context, max_iterations=2)
```

---

### `ContextCompressor` (`context_compressor.py`)

Reduces retrieved context to query-relevant fragments before generation. Supports three strategies via a unified `compress(context, query, compression_type)` interface:

- **`"abstractive"`**: LLM generates a focused summary of the context relevant to the query. Best for reducing verbose context to concise answers.
- **`"extractive"`**: LLM selects the N most relevant sentences from the original text and returns them verbatim. Preserves exact source wording.
- **`"relevance_filter"`**: LLM evaluates each paragraph in the context and drops those below a configurable relevance threshold. Best for filtering obviously irrelevant paragraphs while keeping all others intact.

All methods fall back to returning the original context unchanged on LLM failure, ensuring pipeline continuity. Compression ratios are logged for monitoring.

```python
from vectordb.haystack.components import ContextCompressor

compressor = ContextCompressor(model="llama-3.3-70b-versatile")
compressed = compressor.compress(context, query, compression_type="extractive", num_sentences=5)
```

---

### `QueryEnhancer` (`query_enhancer.py`)

Generates improved retrieval queries from the user's original input. Supports three strategies via a unified `enhance_query(query, enhancement_type)` interface:

- **`"multi_query"`**: Generates N alternative phrasings of the original query (default N=3). All variants are searched and results are merged. Temperature is 0.7 to encourage diversity.
- **`"hyde"`**: Generates M hypothetical documents that would answer the query (default M=3). Hypothetical documents are embedded and used for retrieval — their vocabulary distribution better matches the indexed corpus than a short query does.
- **`"step_back"`**: Generates a single broader, more abstract version of the query. Returns `[original_query, step_back_query]` for joint retrieval.

Always includes the original query in the returned list to ensure specific intent is not lost. Falls back to `[original_query]` on LLM failure.

```python
from vectordb.haystack.components import QueryEnhancer

enhancer = QueryEnhancer(model="llama-3.3-70b-versatile")
queries = enhancer.enhance_query("What causes inflation?", enhancement_type="multi_query", num_queries=3)
# queries = ["What causes inflation?", "What drives rising prices?", "Factors behind monetary inflation"]
```

---

### `ResultMerger` (`result_merger.py`)

Fuses results from multiple retrieval sources (dense + sparse) into a single ranked list with automatic deduplication. Provides two strategies:

- **RRF (Reciprocal Rank Fusion)**: `fuse_rrf(dense_results, sparse_results, top_k, k=60)`. Combines rankings using `1 / (k + rank)` without requiring score normalization. Robust when the two retrieval sources have very different score scales.
- **Weighted fusion**: `fuse_weighted(dense_results, sparse_results, top_k, dense_weight=0.7, sparse_weight=0.3)`. Weights inverse-rank scores by explicit weights. Use when you have domain knowledge about which retriever is more reliable.
- **Unified interface**: `fuse(dense_results, sparse_results, top_k, strategy="rrf", **kwargs)`.

Documents are identified by `doc.id` (with fallback to the first 50 characters of content) for deduplication.

---

### `Evaluators` (`evaluators.py`)

Shared evaluation helpers for measuring pipeline output quality. Used by agentic and cost-optimization pipelines to assess answer quality and retrieval metrics within the pipeline loop.

## LLM Configuration

All LLM-based components (`AgenticRouter`, `ContextCompressor`, `QueryEnhancer`) use the Groq API via Haystack's `OpenAIChatGenerator` with an OpenAI-compatible API endpoint:

```python
generator = OpenAIChatGenerator(
    api_key=Secret.from_token(api_key),
    model="llama-3.3-70b-versatile",
    api_base_url="https://api.groq.com/openai/v1",
    generation_kwargs={"temperature": 0, "max_tokens": 1024},
)
```

Set the `GROQ_API_KEY` environment variable or pass `api_key` directly to the component constructor.

## When to Use Components Directly

- When building a custom pipeline that does not fit the existing feature module templates.
- When experimenting with one pipeline stage at a time (for example, testing different compression strategies with a fixed retriever).
- When combining components from different feature modules into a novel pipeline configuration.

## Common Pitfalls

- **Over-composing before baseline validation**: Build and validate the simplest pipeline first. Add components incrementally and measure the impact of each addition.
- **Inconsistent interfaces between custom stages**: If you extend these components, maintain the same input/output conventions (Haystack `Document` objects, standard config dicts) so components remain interchangeable.
- **No tracing at component boundaries**: Each component logs at `INFO` level. Set `LOG_LEVEL=DEBUG` to see detailed prompt and response content for debugging routing and compression decisions.
