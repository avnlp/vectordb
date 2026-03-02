# Agentic RAG (Haystack)

Agentic RAG introduces iterative decision-making into the retrieval pipeline. Instead of a single retrieve-then-generate pass, the system can evaluate its own answer, decide whether more retrieval is needed, reformulate the query, retrieve again, and refine the answer — repeating until quality is acceptable or the iteration limit is reached.

## How It Works

The pipeline uses `AgenticRouter` (from `components/agentic_router.py`) to orchestrate the loop. The router is an LLM-based decision-maker that selects actions based on the current pipeline state.

### The Decision Loop

```
Start
  │
  ▼
[1] Select tool: retrieval / web_search / calculation / reasoning
  │
  ▼
[2] Retrieve documents from vector store
  │
  ▼
[3] Generate draft answer
  │
  ▼
[4] Evaluate answer quality (relevance, completeness, grounding)
  │
  ├── quality >= threshold → [6] Return final answer
  │
  └── quality < threshold → [5] Refine answer
                              │
                              └── iteration < max_iterations → [2] Retrieve again
                                  iteration >= max_iterations → [6] Return final answer
```

### Tool Selection

The router can route the query to different processing paths before retrieval:

- **retrieval**: Standard vector database search (most common).
- **web_search**: Placeholder for live web search when document knowledge may be stale.
- **calculation**: Mathematical or computational operations.
- **reasoning**: Multi-step logical inference.

### Self-Reflection

After each retrieval and generation step, `AgenticRouter.evaluate_answer_quality()` sends the query, answer, and retrieved context to the LLM and receives a JSON-structured quality assessment with `relevance`, `completeness`, and `grounding` scores (0–100) plus identified issues and improvement suggestions.

`should_refine_answer()` computes the average of the three scores and returns `True` if it falls below `quality_threshold` (default 75).

`refine_answer()` sends the original answer, the quality issues, and the suggestions to the LLM for a targeted revision.

The loop continues for up to `max_iterations` refinement cycles.

Each backend has a dedicated agentic RAG class (`chroma_agentic_rag.py`, `milvus_agentic_rag.py`, etc.) with backend-specific retrieval integration. A shared `base.py` contains the common loop logic.

## When to Use It

- Complex multi-hop questions where the answer requires combining information from multiple retrieval passes.
- High-stakes applications (medical, legal, financial) where answer completeness and grounding must be verified before delivery.
- Workflows where one-shot retrieval and generation consistently underperforms on hard questions.

## When Not to Use It

- Strict low-latency endpoints where each LLM call (routing, evaluation, refinement) adds hundreds of milliseconds.
- Simple factual lookups where single-pass retrieval consistently finds the answer. Agentic overhead is wasted on easy queries.

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Potentially the highest on complex tasks requiring multiple evidence sources |
| Latency | Highest of all features — scales with number of iterations × (LLM call latency) |
| Cost | Highest — routing, evaluation, and refinement each consume LLM tokens |

## Configuration

```yaml
agentic_rag:
  max_iterations: 3             # Hard cap on refinement loop depth
  quality_threshold: 75         # Average score below which answer is refined (0–100)
  model: "llama-3.3-70b-versatile"

search:
  top_k: 10

llm:
  api_key: "${GROQ_API_KEY}"
  api_base_url: "https://api.groq.com/openai/v1"
  temperature: 0.0              # Deterministic routing; 0.7 for answer generation
```

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `max_iterations` | Caps loop cost and latency; start with 2 and increase if quality does not converge |
| `quality_threshold` | Too high = always refines (wastes tokens); too low = accepts poor answers. 70–80 is typical. |
| Routing prompts | The LLM routing prompt quality determines how well the system selects appropriate tools |

## Common Pitfalls

- **No hard loop cap**: Without `max_iterations`, the loop can run indefinitely on queries where quality thresholds are never met. Always set a finite limit.
- **Weak routing prompts**: If the tool selection prompt is ambiguous, the router may oscillate between `"search"` and `"reflect"` without making progress. Be specific about when each action is appropriate.
- **Missing observability**: Agentic pipelines are opaque by nature. Log every routing decision, quality score, and refinement iteration so you can diagnose failures. The `AgenticRouter` logs these at `INFO` level using Python's logging module.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA. Each dataset has per-backend YAML configs under `configs/`.

## Next Steps

- Use `query_enhancement/` for lighter-weight recall improvements without the full agentic loop overhead.
- Use `reranking/` or `semantic_search/` for faster single-pass pipelines that are sufficient for simpler queries.
