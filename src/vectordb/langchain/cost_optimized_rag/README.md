# Cost-Optimized RAG (LangChain)

Cost-optimized RAG provides explicit controls over compute and token spend across retrieval and generation stages, enabling predictable cost per query without unacceptable quality degradation.

## How It Works

Cost optimization is achieved through coordinated controls across multiple pipeline stages:

1. **Retrieval breadth reduction**: A smaller `candidate_pool_size` reduces the number of documents retrieved, embeddings compared, and candidates processed by any downstream reranker or compressor.
2. **Context compression**: Compressed documents sent to the generator mean fewer input tokens per query, directly reducing LLM inference cost.
3. **Model tiering**: Using a faster, cheaper model (for example, Llama 3.1 8B via Groq) for routing and evaluation decisions, and a more capable model only for final answer generation.
4. **Context budget enforcement**: A `context_budget` parameter caps the maximum tokens sent to the generator regardless of how many documents were retrieved.

`RAGHelper.create_llm(config)` creates a `ChatGroq` instance with the configured model tier. The same `RAGHelper.generate()` and `RAGHelper.format_prompt()` methods are used, but context is capped at the budget before formatting.

## When to Use It

- Production systems with explicit monthly LLM cost targets.
- High-query-volume workloads where small per-query savings compound to significant monthly reductions.
- Services with latency SLOs where both cost and response time must be controlled.

## When Not to Use It

- Research and benchmarking runs where absolute quality matters more than cost.
- Small-scale workloads where optimization effort is not economically justified.
- Early experimentation phases — establish a quality baseline before optimizing cost.

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Near-baseline quality when tuned carefully; degrades if compression or pool is too aggressive |
| Latency | Can improve (smaller pool) or worsen (compression overhead) depending on settings |
| Cost | Primary goal — 30–70% reduction achievable with careful tuning |

## Configuration

```yaml
search:
  candidate_pool_size: 15     # First-pass retrieval breadth (smaller = cheaper)
  top_k: 5

cost_optimization:
  context_budget: 2000        # Maximum tokens for generator context
  model_tiering:
    routing_model: "llama-3.1-8b-instant"
    generation_model: "llama-3.3-70b-versatile"
  compression:
    enabled: true
    type: "embedding_filter"
    relevance_threshold: 0.5

rag:
  enabled: true
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY}"
```

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `candidate_pool_size` | Primary retrieval cost lever; reducing from 50 to 15 significantly cuts embedding and ranking costs |
| `context_budget` | Directly caps generation token spend per query |
| `model_tiering` | The cheapest high-impact change: use a fast 8B model for non-critical decisions |

## Cost-Quality Evaluation

Run the same evaluation at multiple settings (`candidate_pool_size` × `context_budget`) and plot quality vs estimated cost to identify the operating point meeting your targets. The `EvaluationResult` container (from `vectordb.utils.evaluation`) can store both quality metrics and cost metadata for comparison.

## Common Pitfalls

- **Optimizing cost before establishing quality**: If the baseline does not achieve acceptable quality, cost optimization only makes it cheaper while remaining inadequate.
- **Single global settings for heterogeneous query classes**: Simple factual queries tolerate small pools (5–10 candidates); complex multi-hop questions need larger pools (25–50). Consider routing queries to different cost tiers based on complexity indicators.
- **No monitoring after cost changes**: Cost reductions applied uniformly may work well on the evaluation set but fail on production edge cases. Monitor quality metrics continuously after any change.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA.

## Next Steps

- Use `reranking/` for quality-first improvements before applying cost controls.
- Use `contextual_compression/` as a standalone feature if token reduction is the primary goal.
- Use `agentic_rag/` for the highest-quality complex reasoning tasks where cost cannot be the primary concern.
