# Semantic Search (LangChain)

Semantic search retrieves documents by meaning rather than exact keyword overlap. Documents and queries are converted into dense vector embeddings by the same model, and similarity is measured by cosine distance in the embedding space.

## How It Works

1. **Indexing**: Each document's text is embedded using `HuggingFaceEmbeddings` (created via `EmbedderHelper.create_embedder(config)` from `utils/embeddings.py`). The resulting float vector and the document metadata are stored in the target backend through the backend's LangChain integration (for example, `Chroma`, `QdrantVectorStore`, `PineconeVectorStore`).
2. **Query embedding**: At search time, the same embedder model is used to embed the query string via `EmbedderHelper.embed_query(embedder, query)`.
3. **Nearest-neighbor retrieval**: The LangChain retriever performs approximate nearest-neighbor search and returns the top-k most similar documents.
4. **Optional generation**: If `rag.enabled: true`, the retrieved documents are formatted into a prompt using `RAGHelper.format_prompt()` and passed to a `ChatGroq` LLM for answer generation.

## When to Use It

- Natural-language questions where query phrasing differs from document vocabulary.
- General-purpose RAG baseline before specializing with advanced features.
- Any corpus where exact keyword overlap between query and documents is unreliable.

## When Not to Use It

- Strict compliance or legal workflows where exact terms must appear verbatim.
- Very small corpora where BM25 already saturates quality.
- Keyword-heavy technical workloads where semantic generalization is unhelpful.

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Strong semantic recall; may miss exact terminology |
| Latency | Low to moderate; dominated by embedding inference |
| Cost | Embedding compute + vector search cost per query |

## Configuration

```yaml
pinecone:
  api_key: "${PINECONE_API_KEY}"
  index_name: "semantic-search"

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"
  batch_size: 32

dataloader:
  dataset: "triviaqa"
  split: "test"
  limit: 500

search:
  top_k: 10

rag:
  enabled: false
```

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `embeddings.model` | The primary quality lever; the model determines how semantically meaningful similarity scores are |
| `search.top_k` | Controls the number of returned candidates; too small misses evidence, too large increases downstream cost |
| `dataloader.limit` | Corpus size for experiments; start small to validate pipeline, then scale up |

## Common Pitfalls

- **Mismatched embedding models**: Using a different model for indexing and querying produces meaningless similarity scores.
- **Oversized chunks**: Large text chunks blur the embedding signal. Shorter, focused chunks typically produce better retrieval.
- **Too small `top_k`**: If relevant evidence is rarely in the top 3 results, increase `top_k` and apply reranking rather than only tuning the embedding model.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA.

## Next Steps

- Add `reranking/` for better final-result precision.
- Switch to `hybrid_indexing/` if queries mix natural language with domain keywords.
- Add `metadata_filtering/` if the corpus has reliable structured attributes to constrain results.
