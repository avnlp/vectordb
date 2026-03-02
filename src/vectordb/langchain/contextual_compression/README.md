# Contextual Compression (LangChain)

Contextual compression reduces retrieved documents to their most query-relevant fragments before passing the context to the generator. This improves generation quality by removing noise and reduces token cost by shortening the generator's input.

## How It Works

LangChain's `ContextualCompressionRetriever` wraps any base retriever with a compressor that filters or transforms retrieved documents. The compressor processes each retrieved document against the query and returns a compressed or filtered version.

Two main approaches are used in this module:

### LLM-Based Extraction

An LLM reads each retrieved document and extracts only the sentences or paragraphs that are relevant to the query. LangChain's `LLMChainExtractor` implements this by prompting an LLM to return a verbatim excerpt from the document. Documents where no relevant content is found are dropped entirely.

### Embedding-Based Relevance Filtering

LangChain's `EmbeddingsFilter` (from `langchain.retrievers.document_compressors`) computes cosine similarity between each retrieved document embedding and the query embedding. Documents (or sentences) below a similarity threshold are dropped. This approach is faster than LLM-based extraction because it requires no additional LLM calls.

## When to Use It

- Long source documents where only a fraction of each document is relevant to the query.
- Token-cost-sensitive pipelines where reducing generator input by 50–80% has meaningful economic impact.
- Cases where retrieved documents contain relevant facts embedded in large amounts of irrelevant context.

## When Not to Use It

- Very short passages (one or two sentences) where compression removes needed nuance.
- Pipelines with tight latency budgets — each LLM-based compression call adds latency.
- Pipelines where retrieval quality is the primary problem; fix retrieval before adding compression.

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Often improves generation grounding by removing noise; may lose nuance with aggressive settings |
| Latency | LLM-based compression adds 200–2000 ms; embedding-based is faster |
| Cost | Can reduce generation cost despite added compression cost if context shrinks significantly |

## Configuration

```yaml
compression:
  type: "llm_extraction"         # "llm_extraction" or "embedding_filter"
  relevance_threshold: 0.5       # Used for embedding_filter
  compression_top_k: 3           # Number of retrieved documents to compress

llm:
  model: "llama-3.3-70b-versatile"
  api_key: "${GROQ_API_KEY}"
  temperature: 0.0

search:
  top_k: 10
  candidate_pool_size: 20
```

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `compression.type` | LLM extraction is more accurate but slower; embedding filter is faster but less nuanced |
| `compression.relevance_threshold` | For embedding filter: too high drops useful content, too low keeps everything |
| `compression.compression_top_k` | Controls how many documents are sent to the compressor; balance quality vs latency |

## Common Pitfalls

- **Over-aggressive compression**: Setting thresholds too strictly may remove the exact sentence needed to answer the question. Validate compressed context against expected answers.
- **Compressing before fixing retrieval**: If retrieval rarely returns relevant documents, compression cannot recover missing evidence.
- **Not auditing faithfulness after compression**: LLM extraction can occasionally paraphrase content in ways that subtly alter factual meaning. Spot-check compressed outputs against source documents.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA.

## Next Steps

- Use `reranking/` as an alternative when improving result ordering is the primary goal.
- Use `cost_optimized_rag/` for a broader budget-control strategy combining compression with retrieval breadth controls.
- Combine with `parent_document_retrieval/` when returned parent documents are long and need trimming before generation.
