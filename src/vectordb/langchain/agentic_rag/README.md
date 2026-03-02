# Agentic RAG (LangChain)

Agentic RAG introduces iterative decision-making into the retrieval pipeline. Instead of a single retrieve-then-generate pass, the system evaluates its candidate answer and decides whether to retrieve more evidence, reflect on the answer, or finalize — repeating until quality is acceptable or the iteration limit is reached.

## How It Works

The pipeline uses `AgenticRouter` (from `components/agentic_router.py`) as the central decision-maker. The router is a `ChatGroq` LLM that returns structured JSON decisions at each step.

### The Decision Loop

```
Start
  │
  ├─ iteration >= max_iterations → [Generate] Return final answer
  │
  ▼
[Route] Query router: action ∈ {search, reflect, generate}
  │
  ├── "search"   → Retrieve from vector store → Generate draft → Loop
  │
  ├── "reflect"  → Evaluate draft quality → Improve if needed → Loop
  │
  └── "generate" → Format and return final answer
```

### Router Decisions

The `AgenticRouter.route()` method receives:
- `query`: the original user question.
- `has_documents`: whether documents have already been retrieved.
- `current_answer`: the current draft answer (if any).
- `iteration`: current iteration number (1-indexed).
- `max_iterations`: the hard loop limit.

It returns `{"action": "search|reflect|generate", "reasoning": "..."}`.

When `iteration >= max_iterations`, the router automatically returns `"generate"` regardless of quality — this is the safety mechanism that prevents infinite loops.

### State Machine

- **`"search"`**: Called when no documents have been retrieved yet, or when reflection identified information gaps requiring additional retrieval.
- **`"reflect"`**: Called when documents exist but answer confidence is uncertain. The router sends the query, current answer, and context back to the LLM for gap identification and correction.
- **`"generate"`**: Called when sufficient information has been gathered and the answer is ready for delivery.

## When to Use It

- Complex multi-hop questions where the answer requires combining information from multiple retrieval passes.
- High-stakes applications where answer completeness and grounding must be verified before delivery.
- Workflows where single-pass retrieval and generation consistently underperforms on hard questions.

## When Not to Use It

- Strict low-latency endpoints where each LLM call (routing + reflection) adds hundreds of milliseconds.
- Simple factual lookups where one-shot retrieval consistently finds the answer.

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Potentially the highest on complex tasks requiring multiple evidence sources |
| Latency | Highest of all features — each iteration adds two or more LLM calls |
| Cost | Highest — routing, retrieval, and reflection each consume LLM tokens |

## Configuration

```yaml
agentic_rag:
  max_iterations: 3               # Hard iteration cap
  model: "llama-3.3-70b-versatile"

search:
  top_k: 10

llm:
  api_key: "${GROQ_API_KEY}"
  temperature: 0.0                # Deterministic routing
```

## Router Prompt

The `AgenticRouter.ROUTING_TEMPLATE` is a structured prompt that includes the current query, `has_documents` flag, `current_answer`, and iteration count. The LLM must return a JSON object with `"action"` and `"reasoning"` keys. Invalid JSON or unrecognized action values raise `ValueError` for fast debugging.

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `max_iterations` | Start with 2; increase only if quality does not converge at lower values |
| LLM temperature | Use 0.0 for routing (deterministic decisions); 0.7 for answer generation (creative) |
| Routing prompt quality | The routing template directly determines how well the LLM navigates the state machine |

## Common Pitfalls

- **No hard loop cap**: Without `max_iterations`, the loop can cycle indefinitely. Always set a finite limit.
- **Ambiguous routing prompts**: If the prompt does not clearly define when to choose `"search"` vs `"reflect"` vs `"generate"`, the router oscillates without making progress.
- **Missing observability**: Log every routing decision, action, and reasoning string. The `AgenticRouter` logs at `INFO` level for decisions and `DEBUG` level for full prompts.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA.

## Next Steps

- Use `query_enhancement/` for lighter-weight recall improvements without the full agentic loop.
- Use `reranking/` or `semantic_search/` for faster single-pass pipelines that are sufficient for simpler queries.
