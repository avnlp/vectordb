# Components (LangChain)

The components directory contains reusable, self-contained LangChain building blocks that implement specific retrieval and generation sub-tasks. Feature pipelines compose these components rather than reimplementing the same logic.

## Components

### `AgenticRouter` (`agentic_router.py`)

An LLM-based decision-making component for agentic RAG pipelines. Uses `ChatGroq` and LangChain's `PromptTemplate` for structured decision-making.

The router implements a three-state state machine:

- **`"search"`**: Retrieve more documents from the vector store. Selected when no documents have been retrieved yet or when reflection identified information gaps.
- **`"reflect"`**: Evaluate and improve the current answer. Selected when documents exist but the answer's quality, completeness, or grounding is uncertain.
- **`"generate"`**: Produce the final answer. Selected when sufficient information has been gathered or when `max_iterations` is reached (forced fallback).

The `ROUTING_TEMPLATE` prompt presents the current state (query, `has_documents`, `current_answer`, iteration number, max iterations) and instructs the LLM to return exactly one JSON object: `{"action": "...", "reasoning": "..."}`.

**Error handling**: The `route()` method raises `ValueError` for invalid JSON responses, missing required fields (`action`, `reasoning`), or unrecognized action values. These errors are informative and help debug prompt formatting issues quickly.

**Iteration limiting**: When `iteration >= max_iterations`, the router short-circuits and returns `"generate"` without calling the LLM, preventing runaway loops.

```python
from langchain_groq import ChatGroq
from vectordb.langchain.components import AgenticRouter

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
router = AgenticRouter(llm)

decision = router.route("What is quantum computing?", has_documents=False)
# {"action": "search", "reasoning": "No documents retrieved yet"}

decision = router.route(
    "What is quantum computing?",
    has_documents=True,
    current_answer="Quantum computing uses qubits...",
    iteration=2,
    max_iterations=3,
)
# {"action": "generate", "reasoning": "Sufficient information available"}
```

---

### `ContextCompressor` (`context_compressor.py`)

LLM-based context compression for reducing retrieved documents to query-relevant fragments. Uses `ChatGroq` for compression inference.

Supports three compression strategies via a unified interface:

- **Abstractive** (`"abstractive"`): The LLM generates a focused summary of the retrieved context relevant to the query.
- **Extractive** (`"extractive"`): The LLM selects and returns verbatim the most relevant sentences from the context.
- **Relevance filtering** (`"relevance_filter"`): The LLM evaluates each paragraph for relevance and returns only paragraphs above the threshold.

All methods return the original context unchanged on LLM failure, ensuring pipeline continuity.

```python
from langchain_groq import ChatGroq
from vectordb.langchain.components import ContextCompressor

llm = ChatGroq(model="llama-3.3-70b-versatile")
compressor = ContextCompressor(llm)
compressed = compressor.compress(context, query, compression_type="extractive", num_sentences=5)
```

---

### `QueryEnhancer` (`query_enhancer.py`)

Generates improved retrieval queries from the user's original input using `ChatGroq` and `PromptTemplate`.

Supports three query generation strategies:

- **`"multi_query"`** (`generate_multi_queries`): Generates 5 alternative phrasings. Returns a list of up to 5 query strings. The original query is NOT included (it is the caller's responsibility to add it if needed).
- **`"hyde"`** (`generate_hyde_queries`): Generates a hypothetical 2–3 sentence document answer. Returns `[original_query, hypothetical_answer]`.
- **`"step_back"`** (`generate_step_back_queries`): Generates 3 broader context questions. Returns `[step_back_1, step_back_2, step_back_3, original_query]`.

Each strategy uses a distinct prompt template (`MULTI_QUERY_TEMPLATE`, `HYDE_TEMPLATE`, `STEP_BACK_TEMPLATE`) with specific formatting instructions to produce clean, structured outputs.

```python
from langchain_groq import ChatGroq
from vectordb.langchain.components import QueryEnhancer

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
enhancer = QueryEnhancer(llm)

queries = enhancer.generate_queries("What is photosynthesis?", mode="step_back")
# ["How do plants convert energy?", "What is the role of chlorophyll?",
#  "What are plant metabolic processes?", "What is photosynthesis?"]
```

## LLM Configuration

All LangChain components use `ChatGroq` from `langchain-groq`. Recommended settings:

```python
from langchain_groq import ChatGroq

# For routing (deterministic)
routing_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)

# For query generation (diverse)
generation_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
```

Set the `GROQ_API_KEY` environment variable or pass `api_key` directly to `ChatGroq`.

## When to Use Components Directly

- When building a custom pipeline that does not match the existing feature module templates.
- When experimenting with one pipeline stage (for example, testing different compression strategies with a fixed retriever).
- When combining components from different feature modules into a custom pipeline.

## Common Pitfalls

- **Over-composing before baseline validation**: Build and validate the simplest pipeline first, then add components.
- **Inconsistent LLM temperature**: Use low temperature (0.0) for routing decisions and higher (0.3–0.7) for creative tasks like query generation.
- **Not logging routing decisions**: All components log at `INFO` level. Set `LOG_LEVEL=DEBUG` to see full prompts and responses for debugging routing and compression behavior.
