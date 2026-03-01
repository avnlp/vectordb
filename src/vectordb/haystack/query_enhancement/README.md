# Query Enhancement (Haystack)

Query enhancement uses a large language model to generate improved or expanded retrieval queries from the user's original input before performing vector search. This increases recall by casting a wider net across different phrasings, angles, and vocabulary.

## How It Works

The `QueryEnhancer` component (from `components/query_enhancer.py`) supports three strategies:

### Multi-Query Generation

Generates N alternative phrasings of the original query. Each alternative captures a different way the same information could be expressed, addressing the vocabulary mismatch problem where the user's words differ from the indexed document's words.

```
User query: "How do transformers learn positional information?"
Generated: [
    "What mechanisms do transformers use to understand word order?",
    "Positional encoding in self-attention models",
    "How does attention-based architecture process sequence positions?"
]
```

All variants are searched independently and results are merged using deduplication or RRF fusion.

### HyDE (Hypothetical Document Embeddings)

Instead of embedding the original query (short, interrogative), the LLM generates a hypothetical document that would answer the query (longer, declarative, matching the vocabulary distribution of indexed documents). The hypothetical document is embedded and used for retrieval alongside the original query.

This bridges the distribution gap between how questions are phrased and how answers are written in source documents. Even if the hypothetical document contains hallucinations, its vocabulary and structure more closely match the indexed corpus.

### Step-Back Prompting

Generates a broader, more abstract version of the query to retrieve foundational background context. The abstract query retrieves broader context, and the specific query retrieves targeted evidence. Results from both are combined.

```
Specific: "What is backpropagation's computational complexity per layer?"
Step-back: "How do neural networks learn from training data?"
```

The step-back approach helps with complex questions where background knowledge is needed to properly contextualize the specific answer.

## When to Use It

- Ambiguous or underspecified user questions where a single embedding may not capture the full intent.
- Knowledge-intensive QA tasks where recall bottlenecks hurt quality more than precision.
- Corpora with domain vocabulary that differs from typical user language.

## When Not to Use It

- Strict latency budgets where extra LLM calls (one per enhancement strategy) are unacceptable.
- Very short, deterministic lookup workflows where the query is already specific and unambiguous.

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Higher recall and robustness; each strategy addresses a different type of recall failure |
| Latency | Higher — one additional LLM call for query generation, plus N retrieval calls (one per generated query) |
| Cost | Higher LLM inference cost for query generation; multiplied by number of generated queries |

## Configuration

```yaml
query_enhancement:
  strategy: "multi_query"   # "multi_query", "hyde", or "step_back"
  num_queries: 3            # Only for multi_query: number of variants to generate
  num_docs: 3               # Only for hyde: number of hypothetical documents

llm:
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY}"
  api_base_url: "https://api.groq.com/openai/v1"
  temperature: 0.7          # Higher than routing (0.0) to encourage diversity
```

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `strategy` | Multi-query is the lightest and most general; HyDE is strongest for distribution mismatch; step-back is best for context retrieval |
| `num_queries` | More variants = higher recall potential but also higher cost and latency. 3–5 is a good starting range. |
| `llm.temperature` | Higher temperature (0.5–0.8) generates more diverse query variants; lower (0.2–0.4) generates more conservative ones |

## Common Pitfalls

- **Generating too many low-value query variants**: Adding 10 variants when 3 suffice multiplies retrieval cost without proportional quality gain. Evaluate the marginal benefit of additional queries.
- **No deduplication or fusion after expansion**: All generated queries are searched and results are merged. Without deduplication, the same document appears multiple times and inflates apparent relevance.
- **Treating enhancement as a replacement for good indexing**: Query expansion increases recall from the existing index but cannot retrieve documents that were not indexed or were indexed with poor embeddings.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA.

## Next Steps

- Use `agentic_rag/` when query enhancement is not sufficient and iterative multi-step retrieval with self-evaluation is needed.
- Use `reranking/` after query enhancement to prioritize the best candidates from the merged result set.
- Use `semantic_search/` as the baseline to measure how much query enhancement improves recall.
