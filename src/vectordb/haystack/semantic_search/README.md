# Semantic Search (Haystack)

Semantic search retrieves documents by meaning rather than exact keyword overlap. Documents and queries are converted into dense vector embeddings by the same model, and similarity is measured by cosine distance in the embedding space.

## How It Works

1. **Indexing**: Each document's text is passed through a SentenceTransformers model (`SentenceTransformersDocumentEmbedder`) to produce a dense float vector. The vector and the document metadata are stored in the target vector database.
2. **Query embedding**: At search time, the query string is embedded with the same model (`SentenceTransformersTextEmbedder`) to produce a query vector.
3. **Nearest-neighbor retrieval**: The database performs approximate nearest-neighbor (ANN) search over indexed embeddings and returns the top-k most similar documents ranked by cosine similarity score.
4. **Optional filtering**: Metadata filters can be applied to restrict the candidate set before similarity scoring, using the backend's native filter syntax.

The `EmbedderFactory` (from `utils/embeddings.py`) creates and warms up both the document and text embedders from the config file. Warm-up pre-loads the model weights so the first real call does not incur cold-start latency.

## When to Use It

- Natural-language questions where the phrasing in the question may differ from the phrasing in documents.
- General-purpose RAG starting points before specializing with advanced features.
- Any corpus where exact keyword overlap between query and documents is unreliable.

## When Not to Use It

- Strict compliance or legal workflows where specific terms must appear verbatim.
- Very small corpora (fewer than a few hundred documents) where BM25 already saturates quality.
- Keyword-heavy technical workloads with domain acronyms and jargon where semantic generalization is unhelpful.

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Strong semantic recall; can miss exact terminology |
| Latency | Low to moderate; dominated by embedding model inference |
| Cost | Embedding model compute + vector search cost per query |

## Configuration

Each backend has a set of config files under `configs/`. A typical config looks like:

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
| `embeddings.model` | The single largest quality lever. Better models produce more meaningful similarity scores. |
| `search.top_k` | Controls how many candidates are returned. Too small misses evidence; too large increases downstream cost. |
| `dataloader.limit` | Controls corpus size for experiments. Start small to validate the pipeline, then scale up. |

## Common Pitfalls

- **Mismatched embedding models**: Using a different model for indexing and querying produces meaningless similarity scores. Always use the same `model` value in both indexing and search configs.
- **Oversized chunks**: Large text chunks blur the embedding signal, making the vector represent too many topics at once. Shorter, focused chunks usually produce better retrieval.
- **Too small `top_k`**: If relevant evidence is rarely in the top 3 results, increasing `top_k` to 10 or 20 and then applying reranking usually helps more than tuning the embedding model.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

Each backend has an indexing script in `indexing/` and a search script in `search/`.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA. Config files are named `{backend}_{dataset}.yaml` inside `configs/`.

## Next Steps

After establishing a semantic search baseline:

- Add `reranking/` for better final-result precision.
- Switch to `hybrid_indexing/` if your queries mix natural language with domain keywords.
- Add `metadata_filtering/` if your corpus has reliable structured attributes to constrain results.
