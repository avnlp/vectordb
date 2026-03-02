# Query Enhancement (LangChain)

Query enhancement uses a large language model to generate improved or expanded retrieval queries from the user's original input before vector search. This increases recall by addressing vocabulary mismatch and query ambiguity.

## How It Works

The `QueryEnhancer` component (from `components/query_enhancer.py`) supports three strategies, accessed via a unified `generate_queries(query, mode)` interface:

### Multi-Query Generation (`mode="multi_query"`)

Generates 5 alternative phrasings of the original query using `ChatGroq` with a structured prompt. Each variation captures a different way the same information could be expressed, addressing vocabulary mismatch between user queries and indexed documents.

```
Original: "What is AI?"
Generated: [
    "Define artificial intelligence",
    "Explain what AI means",
    "What does artificial intelligence refer to",
    "How is AI defined in computer science?",
    "What are the core concepts of AI?"
]
```

All generated queries (plus optionally the original) are searched independently and results are merged.

### HyDE — Hypothetical Document Embeddings (`mode="hyde"`)

Generates a hypothetical 2–3 sentence answer document and returns `[original_query, hypothetical_answer]`. The hypothetical document is embedded and used for retrieval. Because the hypothetical answer is declarative and uses document-like vocabulary, its embedding better matches indexed passages than the short, interrogative original query.

### Step-Back Prompting (`mode="step_back"`)

Generates 3 broader, foundational questions that provide background context for the original query. Returns up to 4 queries: `[step_back_1, step_back_2, step_back_3, original_query]`. Step-back retrieval fetches broader context that helps contextualize and answer the specific question.

```
Specific: "What is backpropagation?"
Step-back: ["How do neural networks learn?", "What is gradient descent?", "What is machine learning optimization?"]
```

## When to Use It

- Ambiguous or short user queries where a single embedding may not capture the full intent.
- Knowledge-intensive QA where recall bottlenecks harm quality more than precision overhead.
- Corpora with significant vocabulary mismatch between user language and document language.

## When Not to Use It

- Strict latency budgets where extra LLM calls per query are unacceptable.
- Simple, precise factual lookups where the query is already specific and unambiguous.

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Higher recall and robustness; each strategy addresses a different recall failure mode |
| Latency | One additional LLM call per query, plus N retrieval calls (one per generated query) |
| Cost | Higher LLM inference cost proportional to number of generated queries |

## Configuration

```yaml
query_enhancement:
  mode: "multi_query"     # "multi_query", "hyde", or "step_back"

llm:
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY}"
  temperature: 0.3        # Low for consistent output; higher for more diverse variants
```

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `mode` | Multi-query is most general; HyDE is strongest for distribution mismatch; step-back is best for context-dependent questions |
| LLM temperature | Higher (0.5–0.7) generates more diverse query variants; lower (0.2–0.3) generates more conservative ones |
| Fusion strategy | How the results from multiple queries are merged affects final recall |

## Common Pitfalls

- **Too many low-value query variants**: Generating 10 variants when 5 suffice multiplies retrieval cost. Evaluate the marginal benefit of each additional query.
- **No deduplication after expansion**: All generated queries retrieve results that must be deduplicated before ranking. Without deduplication, popular documents appear multiple times.
- **Expecting enhancement to fix poor indexing**: Query expansion increases recall from the existing index but cannot retrieve documents that were never indexed or were indexed with poor embeddings.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA.

## Next Steps

- Use `agentic_rag/` when query enhancement is not sufficient and iterative multi-step retrieval is needed.
- Use `reranking/` after query enhancement to prioritize the best candidates from the merged result set.
