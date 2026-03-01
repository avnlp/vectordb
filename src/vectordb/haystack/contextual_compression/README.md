# Contextual Compression (Haystack)

Contextual compression reduces retrieved documents to their most query-relevant fragments before passing the context to the generator. This improves generation quality by reducing noise and irrelevant content, and reduces token cost by shortening the generator's input.

## How It Works

After standard retrieval, the `ContextCompressor` component (from `components/context_compressor.py`) processes the retrieved documents with an LLM. Three compression strategies are available:

### Abstractive Compression

An LLM reads the full retrieved context and generates a summary focused on information relevant to the query. The summary captures key facts using the LLM's own language.

```
Query: "What causes auroras?"
Retrieved: 3 documents, each 500 words
Compressed: 1 paragraph summarizing the solar wind-magnetic field interaction
```

### Extractive Compression

An LLM selects the most relevant sentences from the retrieved text and returns them verbatim, without rephrasing. This preserves exact wording from source documents — useful when faithfulness to the original is important.

```
Returned: The 5 most relevant sentences, extracted from their original documents
```

### Relevance Filtering

Each paragraph in the retrieved context is scored 0–100 for relevance to the query. Paragraphs below the configured threshold are dropped. The output is a subset of the original paragraphs.

```
Threshold: 50
Input: 6 paragraphs across 2 documents
Output: 4 paragraphs that scored >= 50
```

All compression methods fall back to returning the original context unchanged if the LLM call fails, ensuring the pipeline never breaks due to a compression error.

The `compression_utils.py` file contains shared utilities for token counting and context building used across compression strategies.

## When to Use It

- Long source documents or large candidate sets where the retrieved context substantially exceeds the generator's context window.
- Pipelines with high token cost where reducing the generator's input by 50–80% has meaningful economic impact.
- Cases where retrieved documents contain relevant paragraphs embedded in large amounts of irrelevant background material.
- Generation quality that is degraded by noisy or tangential context; compression can improve answer faithfulness.

## When Not to Use It

- Very short passages (one or two sentences per document) where compression removes needed nuance.
- Pipelines that already operate with minimal clean context; compression overhead is not justified.
- Strict latency requirements — each compression call adds one LLM round-trip.

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Often improves generation grounding by removing noise; may lose nuance with aggressive compression |
| Latency | Extra LLM call adds 200–2000 ms depending on context length and model |
| Cost | Can reduce generation cost despite added compression cost if context shrinks significantly |

## Configuration

```yaml
compression:
  strategy: "abstractive"       # "abstractive", "extractive", or "relevance_filter"
  max_tokens: 2048              # For abstractive: max output tokens
  num_sentences: 5             # For extractive: number of sentences to keep
  relevance_threshold: 0.5     # For relevance_filter: minimum score to keep (0.0–1.0)
  compression_top_k: 3         # Number of retrieved documents to feed into compressor

llm:
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY}"
  api_base_url: "https://api.groq.com/openai/v1"
  temperature: 0.0             # Deterministic compression
  max_tokens: 2048
```

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `compression.strategy` | Extractive is safest (preserves original text); abstractive is most concise; relevance filter is most configurable |
| `compression.compression_top_k` | Controls how many retrieved documents are sent to the compressor; balance quality vs latency |
| `compression.relevance_threshold` | For relevance filter only: too high removes useful context; too low keeps everything |

## Common Pitfalls

- **Over-compressing**: Setting `num_sentences` to 1 or `max_tokens` to 100 may remove the exact evidence needed for the answer. Start conservatively (5–10 sentences or 500–1000 tokens).
- **Compressing before fixing retrieval quality**: If retrieval recall is poor, compression cannot recover missing evidence. Establish strong retrieval first, then add compression.
- **Not auditing answer faithfulness after compression changes**: Abstractive compression in particular may paraphrase source material in ways that subtly alter factual content. Spot-check generated answers against source documents.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

`llm_extraction`, `reranking` (specialized configs for compression evaluation scenarios).

## Next Steps

- Use `reranking/` as an alternative when improving result ranking is the primary goal rather than reducing context length.
- Use `cost_optimized_rag/` for a broader budget-control strategy that combines compression with retrieval breadth controls.
- Combine with `parent_document_retrieval/` when parent documents are retrieved but need trimming before generation.
