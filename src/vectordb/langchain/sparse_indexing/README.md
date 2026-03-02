# Sparse Indexing (LangChain)

Sparse retrieval encodes documents and queries as high-dimensional token-weight vectors. Retrieval is based on lexical overlap between query and document tokens, weighted by a learned importance model (SPLADE).

## How It Works

1. **Sparse encoding**: Documents are encoded by a sparse embedding model (typically SPLADE) into token-weight vectors where each dimension corresponds to a vocabulary token. Non-zero dimensions are stored efficiently.
2. **Sparse index**: The sparse vectors are stored in the backend's sparse index alongside (or instead of) dense vectors.
3. **Sparse query**: At query time, the same sparse encoder processes the query string into a sparse query vector.
4. **Inner-product retrieval**: Documents are ranked by inner product between the query sparse vector and each document sparse vector. SPLADE's learned weights allow limited vocabulary expansion beyond exact token matches.

The `utils/sparse_embeddings.py` helper creates and applies the sparse embedding model within LangChain pipelines.

## When to Use It

- Keyword-heavy workloads: product codes, legal citations, medical terminology, technical identifiers.
- Domain jargon where exact terms matter and semantic paraphrasing is unhelpful.
- As the sparse leg of a hybrid pipeline alongside dense retrieval.

## When Not to Use It

- Open-domain QA with varied natural-language phrasing where query and documents rarely share vocabulary.
- Tasks where document vocabulary is substantially different from user query vocabulary.

## Tradeoffs

| Dimension | What to Expect |
|---|---|
| Quality | Strong lexical precision; limited semantic generalization beyond SPLADE learned expansion |
| Latency | Backend-dependent; often competitive with ANN search |
| Cost | Sparse embedding inference is usually faster than dense; storage depends on vector sparsity |

## Configuration

```yaml
sparse:
  model: "naver/splade-cocondenser-ensembledistil"

search:
  top_k: 10
```

## Settings to Tune First

| Setting | Why It Matters |
|---|---|
| `sparse.model` | SPLADE model determines vocabulary expansion behavior and lexical matching coverage |
| `search.top_k` | Sparse retrieval has sharp relevance cutoffs; too small a `top_k` misses relevant documents |

## Common Pitfalls

- **Using sparse-only for paraphrase-heavy queries**: If queries use different terminology from documents, sparse retrieval fails without hybrid pairing.
- **Ignoring tokenization quirks**: Compound words, camelCase, and domain abbreviations may tokenize unexpectedly in SPLADE models.
- **Not pairing with dense in mixed workloads**: Sparse retrieval alone rarely outperforms hybrid retrieval for general RAG use cases.

## Backends Supported

Chroma, Milvus, Pinecone, Qdrant, Weaviate.

## Dataset Configs Provided

ARC, Earnings Calls, FActScore, PopQA, TriviaQA.

## Next Steps

- Use `hybrid_indexing/` to pair sparse with dense for more robust coverage.
- Use `semantic_search/` as the baseline to determine whether your workload benefits more from semantic or lexical retrieval.
