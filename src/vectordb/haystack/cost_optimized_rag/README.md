# Cost-Optimized RAG (Haystack)

Cost-optimized RAG balances retrieval and generation quality against compute and token spend. It provides explicit controls over candidate pool sizes, context length passed to the generator, and model tier selection to achieve predictable cost per query.

## How It Works

Cost optimization operates across multiple pipeline stages:

1. **Retrieval breadth control**: Rather than retrieving the maximum useful candidate pool, cost-optimized RAG retrieves a smaller, tuned number of documents. This reduces the reranking workload and the number of embeddings computed at query time.
2. **Selective compression**: Contextual compression is applied only to the retrieved candidates, reducing the token count passed to the generator. This is the primary cost lever for generation — shorter context = fewer tokens = lower LLM cost.
3. **Model tiering**: The pipeline uses cheaper, faster models for lower-stakes stages (for example, Llama 3.1 8B for reranking decisions) and more capable models only for final answer generation when the query requires it.
4. **Context budget enforcement**: A `context_budget` parameter caps the maximum number of tokens passed to the generator, regardless of how many documents were retrieved.

The `cost_optimized_rag/base/` directory contains base pipeline classes, `cost_optimized_rag/evaluation/` contains cost-quality tradeoff tracking, and `cost_optimized_rag/utils/` contains token counting and budget utilities.

## When to Use It

- Production systems with explicit cost per query targets or monthly budget caps.
- High-query-volume workloads where small per-query savings multiply to significant monthly savings.
- Services with latency SLOs where both cost and latency must be controlled simultaneously.

## When Not to Use It

- Research and benchmarking runs where absolute quality matters more than cost.
- Very small-scale workloads where the optimization effort is not economically justified.
- Early experimentation phases before a quality baseline has been established — optimize cost only after quality is acceptable.

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Near-baseline quality when tuned well; quality degrades if compression is too aggressive or pool too small |
| Latency | Can improve or worsen depending on whether retrieval breadth reduction or compression overhead dominates |
| Cost | Primary goal — significant reductions achievable (30–70%) with careful tuning |

## Configuration

```yaml
search:
  candidate_pool_size: 15        # First-pass retrieval breadth (smaller = cheaper)
  top_k: 5                       # Final result count

cost_optimization:
  context_budget: 2000           # Maximum tokens for the generator context
  model_tiering:
    routing: "llama-3.1-8b-instant"     # Fast/cheap model for non-critical decisions
    generation: "llama-3.3-70b-versatile"  # Capable model for final answer
  compression:
    enabled: true
    strategy: "extractive"
    num_sentences: 5

rag:
  enabled: true
  model: "${cost_optimization.model_tiering.generation}"
  api_key: "${GROQ_API_KEY}"
```

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `candidate_pool_size` | Primary retrieval cost lever; cutting from 50 to 15 significantly reduces embedding and ranking cost |
| `cost_optimization.context_budget` | Directly caps generation token spend; set based on your LLM's per-token pricing |
| `model_tiering` | Using a fast 8B model for routing decisions vs a 70B model for generation creates significant cost differentiation |

## Cost-Quality Evaluation Pattern

The `evaluation/` subdirectory contains scripts that run retrieval evaluation while tracking cost metrics. This enables explicit cost-quality curves — run the same evaluation at different `candidate_pool_size` and `context_budget` settings and plot quality vs cost to identify the operating point that meets your targets.

## Common Pitfalls

- **Optimizing cost before establishing quality**: If the baseline pipeline does not achieve acceptable quality, cost optimization will only make it cheaper while remaining inadequate. Fix quality first.
- **Using single global settings for heterogeneous query classes**: Simple factual lookups can tolerate very small pools (5 candidates), while complex multi-hop questions need larger pools (25–50). Consider query routing to different cost tiers.
- **No monitoring for quality drift after cost cuts**: Token budget reductions may work well on the evaluation set but fail on edge cases that appear in production. Monitor quality metrics continuously after any cost reduction change.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA.

## Next Steps

- Use `reranking/` for quality-first ranking improvements before applying cost controls.
- Use `contextual_compression/` as a standalone feature if token reduction is the only goal.
- Use `agentic_rag/` for the highest-quality (and highest-cost) complex reasoning tasks where quality cannot be compromised.
